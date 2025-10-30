//! Postprocessing utilities for military target detection results

use crate::error::{DetectionError, Result};
use crate::types::{BoundingBox, Detection, DetectionResult, TargetClass};
use ndarray::Array2;
use std::collections::HashMap;

/// Postprocessor for converting model outputs to detection results
pub struct Postprocessor {
    /// Confidence threshold for filtering detections
    confidence_threshold: f32,
    /// NMS threshold for removing duplicate detections  
    nms_threshold: f32,
    /// Maximum number of detections to return
    max_detections: usize,
    /// Input image size used during preprocessing
    input_size: (u32, u32),
}

impl Postprocessor {
    /// Create new postprocessor
    pub fn new(
        confidence_threshold: f32,
        nms_threshold: f32,
        max_detections: usize,
        input_size: (u32, u32),
    ) -> Self {
        Self {
            confidence_threshold,
            nms_threshold,
            max_detections,
            input_size,
        }
    }

    /// Process model output to extract detections
    pub fn process(
        &self,
        output: &Array2<f32>,
        original_size: (u32, u32),
        inference_time_ms: f32,
    ) -> Result<DetectionResult> {
        // Parse raw detections from model output
        let raw_detections = self.parse_yolo_output(output)?;

        // Filter by confidence threshold
        let filtered_detections = self.filter_by_confidence(raw_detections);

        // Apply non-maximum suppression
        let nms_detections = self.apply_nms(filtered_detections);

        // Scale bounding boxes to original image size
        let scaled_detections = self.scale_detections(nms_detections, original_size);

        // Limit number of detections
        let final_detections = self.limit_detections(scaled_detections);

        Ok(DetectionResult::new(
            final_detections,
            inference_time_ms,
            original_size.0,
            original_size.1,
        ))
    }

    /// Parse YOLO model output format
    /// Expected format: [num_detections, 6] where each detection is [x, y, w, h, confidence, class_id]
    fn parse_yolo_output(&self, output: &Array2<f32>) -> Result<Vec<RawDetection>> {
        let mut detections = Vec::new();

        // Check output shape
        if output.ncols() < 6 {
            return Err(DetectionError::inference(format!(
                "Invalid output shape: expected at least 6 columns, got {}",
                output.ncols()
            )));
        }

        for row in output.rows() {
            // Extract values from row
            let x_center = row[0];
            let y_center = row[1];
            let width = row[2];
            let height = row[3];
            let confidence = row[4];
            let class_id = row[5] as u32;

            // Skip low confidence detections early
            if confidence < self.confidence_threshold {
                continue;
            }

            // Validate class ID
            let class = TargetClass::from_id(class_id).ok_or_else(|| {
                DetectionError::inference(format!("Invalid class ID: {}", class_id))
            })?;

            // Convert center coordinates to top-left coordinates
            let x = x_center - width / 2.0;
            let y = y_center - height / 2.0;

            // Clamp coordinates to [0, 1]
            let x = x.max(0.0).min(1.0);
            let y = y.max(0.0).min(1.0);
            let width = width.max(0.0).min(1.0 - x);
            let height = height.max(0.0).min(1.0 - y);

            detections.push(RawDetection {
                class,
                confidence,
                bbox: BoundingBox::new(x, y, width, height),
            });
        }

        Ok(detections)
    }

    /// Filter detections by confidence threshold
    fn filter_by_confidence(&self, detections: Vec<RawDetection>) -> Vec<RawDetection> {
        detections
            .into_iter()
            .filter(|det| det.confidence >= self.confidence_threshold)
            .collect()
    }

    /// Apply Non-Maximum Suppression to remove duplicate detections
    fn apply_nms(&self, mut detections: Vec<RawDetection>) -> Vec<RawDetection> {
        // Sort by confidence (descending)
        detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        let mut keep = Vec::new();
        let mut suppress = vec![false; detections.len()];

        for i in 0..detections.len() {
            if suppress[i] {
                continue;
            }

            keep.push(detections[i].clone());

            // Check all subsequent detections
            for j in (i + 1)..detections.len() {
                if suppress[j] {
                    continue;
                }

                // Only apply NMS within the same class
                if detections[i].class == detections[j].class {
                    let iou = detections[i].bbox.iou(&detections[j].bbox);
                    if iou > self.nms_threshold {
                        suppress[j] = true;
                    }
                }
            }
        }

        keep
    }

    /// Scale detection coordinates from model input size to original image size
    fn scale_detections(
        &self,
        detections: Vec<RawDetection>,
        original_size: (u32, u32),
    ) -> Vec<Detection> {
        let (orig_width, orig_height) = original_size;
        let (input_width, input_height) = self.input_size;

        // Calculate scaling factors
        let scale_x = orig_width as f32 / input_width as f32;
        let scale_y = orig_height as f32 / input_height as f32;

        detections
            .into_iter()
            .map(|raw_det| {
                // Scale coordinates back to original image size
                let scaled_bbox = BoundingBox::new(
                    raw_det.bbox.x * scale_x / orig_width as f32,
                    raw_det.bbox.y * scale_y / orig_height as f32,
                    raw_det.bbox.width * scale_x / orig_width as f32,
                    raw_det.bbox.height * scale_y / orig_height as f32,
                );

                Detection::new(raw_det.class, raw_det.confidence, scaled_bbox)
            })
            .collect()
    }

    /// Limit number of detections to maximum allowed
    fn limit_detections(&self, mut detections: Vec<Detection>) -> Vec<Detection> {
        if detections.len() > self.max_detections {
            // Sort by confidence and take top N
            detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
            detections.truncate(self.max_detections);
        }
        detections
    }

    /// Get detection statistics
    pub fn get_class_statistics(
        &self,
        detections: &[Detection],
    ) -> HashMap<TargetClass, ClassStats> {
        let mut stats = HashMap::new();

        for class in TargetClass::all() {
            let class_detections: Vec<_> =
                detections.iter().filter(|det| det.class == class).collect();

            let count = class_detections.len();
            let avg_confidence = if count > 0 {
                class_detections
                    .iter()
                    .map(|det| det.confidence)
                    .sum::<f32>()
                    / count as f32
            } else {
                0.0
            };

            let max_confidence = class_detections
                .iter()
                .map(|det| det.confidence)
                .fold(0.0f32, |a, b| a.max(b));

            stats.insert(
                class,
                ClassStats {
                    count,
                    avg_confidence,
                    max_confidence,
                },
            );
        }

        stats
    }

    /// Update configuration
    pub fn update_config(
        &mut self,
        confidence_threshold: Option<f32>,
        nms_threshold: Option<f32>,
        max_detections: Option<usize>,
    ) {
        if let Some(conf) = confidence_threshold {
            self.confidence_threshold = conf.clamp(0.0, 1.0);
        }
        if let Some(nms) = nms_threshold {
            self.nms_threshold = nms.clamp(0.0, 1.0);
        }
        if let Some(max_det) = max_detections {
            self.max_detections = max_det;
        }
    }
}

/// Raw detection from model output (before NMS and scaling)
#[derive(Debug, Clone)]
struct RawDetection {
    class: TargetClass,
    confidence: f32,
    bbox: BoundingBox,
}

/// Statistics for a specific target class
#[derive(Debug, Clone)]
pub struct ClassStats {
    pub count: usize,
    pub avg_confidence: f32,
    pub max_confidence: f32,
}

/// Utility functions for postprocessing
pub mod utils {
    use super::*;

    /// Apply class-specific confidence thresholds
    pub fn apply_class_thresholds(
        detections: Vec<Detection>,
        thresholds: &HashMap<TargetClass, f32>,
    ) -> Vec<Detection> {
        detections
            .into_iter()
            .filter(|det| {
                let threshold = thresholds.get(&det.class).unwrap_or(&0.5);
                det.confidence >= *threshold
            })
            .collect()
    }

    /// Group detections by class
    pub fn group_by_class(detections: &[Detection]) -> HashMap<TargetClass, Vec<&Detection>> {
        let mut groups = HashMap::new();

        for detection in detections {
            groups
                .entry(detection.class)
                .or_insert_with(Vec::new)
                .push(detection);
        }

        groups
    }

    /// Calculate detection density (detections per unit area)
    pub fn calculate_density(detections: &[Detection], image_area: f32) -> f32 {
        detections.len() as f32 / image_area
    }

    /// Find detections within a specific region
    pub fn find_in_region<'a>(
        detections: &'a [Detection],
        region: &BoundingBox,
    ) -> Vec<&'a Detection> {
        detections
            .iter()
            .filter(|det| det.bbox.intersects(region))
            .collect()
    }

    /// Merge overlapping detections of the same class
    pub fn merge_overlapping(detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
        let mut merged = Vec::new();
        let mut used = vec![false; detections.len()];

        for i in 0..detections.len() {
            if used[i] {
                continue;
            }

            let mut group = vec![i];
            used[i] = true;

            // Find all overlapping detections of the same class
            for j in (i + 1)..detections.len() {
                if used[j] || detections[i].class != detections[j].class {
                    continue;
                }

                let iou = detections[i].bbox.iou(&detections[j].bbox);
                if iou > iou_threshold {
                    group.push(j);
                    used[j] = true;
                }
            }

            // Merge detections in group
            if group.len() == 1 {
                merged.push(detections[i].clone());
            } else {
                merged.push(merge_detection_group(&detections, &group));
            }
        }

        merged
    }

    fn merge_detection_group(detections: &[Detection], indices: &[usize]) -> Detection {
        let class = detections[indices[0]].class;

        // Average confidence
        let avg_confidence = indices
            .iter()
            .map(|&i| detections[i].confidence)
            .sum::<f32>()
            / indices.len() as f32;

        // Average bounding box coordinates
        let avg_x =
            indices.iter().map(|&i| detections[i].bbox.x).sum::<f32>() / indices.len() as f32;

        let avg_y =
            indices.iter().map(|&i| detections[i].bbox.y).sum::<f32>() / indices.len() as f32;

        let avg_width = indices
            .iter()
            .map(|&i| detections[i].bbox.width)
            .sum::<f32>()
            / indices.len() as f32;

        let avg_height = indices
            .iter()
            .map(|&i| detections[i].bbox.height)
            .sum::<f32>()
            / indices.len() as f32;

        Detection::new(
            class,
            avg_confidence,
            BoundingBox::new(avg_x, avg_y, avg_width, avg_height),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_postprocessor_creation() {
        let postprocessor = Postprocessor::new(0.5, 0.45, 100, (640, 640));
        assert_eq!(postprocessor.confidence_threshold, 0.5);
        assert_eq!(postprocessor.nms_threshold, 0.45);
        assert_eq!(postprocessor.max_detections, 100);
    }

    #[test]
    fn test_confidence_filtering() {
        let postprocessor = Postprocessor::new(0.5, 0.45, 100, (640, 640));

        let detections = vec![
            RawDetection {
                class: TargetClass::ArmedPersonnel,
                confidence: 0.6,
                bbox: BoundingBox::new(0.1, 0.1, 0.2, 0.2),
            },
            RawDetection {
                class: TargetClass::RocketLauncher,
                confidence: 0.3, // Below threshold
                bbox: BoundingBox::new(0.5, 0.5, 0.2, 0.2),
            },
        ];

        let filtered = postprocessor.filter_by_confidence(detections);
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].class, TargetClass::ArmedPersonnel);
    }

    #[test]
    fn test_bounding_box_iou() {
        let bbox1 = BoundingBox::new(0.0, 0.0, 0.5, 0.5);
        let bbox2 = BoundingBox::new(0.25, 0.25, 0.5, 0.5);

        let iou = bbox1.iou(&bbox2);
        assert!(iou > 0.0 && iou < 1.0); // Should have some overlap
    }
}
