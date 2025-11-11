/// ByteTrack tracker implementation using ioutrack crate
///
/// ByteTrack is a more modern tracking algorithm compared to Kalman filtering.
/// It uses motion prediction and data association with high/low confidence detection handling.
///
/// Reference: https://arxiv.org/abs/2110.06864
use crate::frame_pipeline::TileDetection;
use ioutrack::ByteTrack;
use ndarray::{Array2, CowArray};
use std::time::Instant;

/// Configuration for ByteTrack algorithm
#[derive(Clone, Debug)]
pub struct ByteTrackConfig {
    /// Maximum frames to keep a track alive without matching detections
    pub max_age: u32,
    /// Minimum consecutive hits before track is confirmed
    pub min_hits: u32,
    /// IOU threshold for matching
    pub iou_threshold: f32,
    /// Minimum score to create a new tracklet from unmatched detection
    pub init_tracker_min_score: f32,
    /// High confidence threshold for first round of association
    pub high_score_threshold: f32,
    /// Low confidence threshold for second round of association  
    pub low_score_threshold: f32,
    /// Measurement noise covariance diagonal
    pub measurement_noise: [f32; 4],
    /// Process noise covariance diagonal
    pub process_noise: [f32; 7],
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            max_age: 1,
            min_hits: 3,
            iou_threshold: 0.3,
            init_tracker_min_score: 0.8,
            high_score_threshold: 0.7,
            low_score_threshold: 0.1,
            measurement_noise: [1.0, 1.0, 10.0, 10.0],
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        }
    }
}

/// Multi-object tracker using ByteTrack algorithm
pub struct MultiObjectTracker {
    tracker: ByteTrack,
    config: ByteTrackConfig,
    frame_id: u32,
    last_tracks: Vec<TileDetection>,
}

impl MultiObjectTracker {
    /// Create new ByteTrack tracker
    pub fn new(config: ByteTrackConfig) -> Self {
        let (tracker, _base) = ByteTrack::new(
            config.max_age,
            config.min_hits,
            config.iou_threshold,
            config.init_tracker_min_score,
            config.high_score_threshold,
            config.low_score_threshold,
            config.measurement_noise,
            config.process_noise,
        );

        Self {
            tracker,
            config,
            frame_id: 0,
            last_tracks: Vec::new(),
        }
    }

    /// Update tracker with new detections
    pub fn update(&mut self, detections: &[TileDetection], _dt: f32) -> Vec<TileDetection> {
        self.frame_id += 1;

        // Convert TileDetections to the expected format (n_boxes, 5) array: [x1, y1, x2, y2, score]
        let detection_data: Vec<f32> = detections
            .iter()
            .flat_map(|det| {
                vec![
                    det.x,          // x1 (left)
                    det.y,          // y1 (top)
                    det.x + det.w,  // x2 (right)
                    det.y + det.h,  // y2 (bottom)
                    det.confidence, // score
                ]
            })
            .collect();

        if detection_data.is_empty() {
            self.last_tracks.clear();
            return Vec::new();
        }

        // Create ndarray from detection data
        let n_detections = detections.len();
        let detection_array = Array2::from_shape_vec((n_detections, 5), detection_data)
            .expect("Failed to create detection array");

        // Update tracker
        let track_array =
            match self
                .tracker
                .update(CowArray::from(detection_array.view()), false, false)
            {
                Ok(tracks) => tracks,
                Err(e) => {
                    eprintln!("ByteTrack update error: {}", e);
                    self.last_tracks.clear();
                    return Vec::new();
                }
            };

        // Convert tracks back to TileDetections
        let mut tracked_detections = Vec::new();
        for track_row in track_array.outer_iter() {
            if let Some(detection) = self.track_to_tile_detection(&track_row, detections) {
                tracked_detections.push(detection);
            }
        }

        self.last_tracks = tracked_detections.clone();
        tracked_detections
    }

    /// Get current predictions (for ByteTrack, this is the same as the last update result)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        // Return the last tracked detections as predictions
        self.last_tracks.clone()
    }

    /// Get number of active tracks
    pub fn num_tracks(&self) -> usize {
        self.last_tracks.len()
    }

    /// Get track by ID (simplified implementation)
    pub fn get_track(&self, track_id: u32) -> Option<&TileDetection> {
        self.last_tracks
            .iter()
            .find(|det| det.track_id == Some(track_id))
    }

    /// Evict stale tracks (ByteTrack handles this internally)
    pub fn evict_stale_tracks(&mut self, _max_age_ms: u64) {
        // ByteTracker handles track lifecycle internally
        // No manual eviction needed
    }

    /// Convert ByteTrack track result to TileDetection
    fn track_to_tile_detection(
        &self,
        track_row: &ndarray::ArrayView1<f32>,
        original_detections: &[TileDetection],
    ) -> Option<TileDetection> {
        // Track row format: [x1, y1, x2, y2, track_id]
        if track_row.len() < 5 {
            return None;
        }

        let x1 = track_row[0];
        let y1 = track_row[1];
        let x2 = track_row[2];
        let y2 = track_row[3];
        let track_id = track_row[4] as u32;

        let x = x1;
        let y = y1;
        let w = x2 - x1;
        let h = y2 - y1;

        // Find the closest original detection to get class information
        let mut best_match: Option<&TileDetection> = None;
        let mut best_iou = 0.0;

        for orig_det in original_detections {
            let iou =
                calculate_iou_boxes(x, y, w, h, orig_det.x, orig_det.y, orig_det.w, orig_det.h);
            if iou > best_iou {
                best_iou = iou;
                best_match = Some(orig_det);
            }
        }

        let (class_id, class_name, confidence) = if let Some(matched_det) = best_match {
            (
                matched_det.class_id,
                matched_det.class_name.clone(),
                matched_det.confidence,
            )
        } else {
            (0, "unknown".to_string(), 0.5)
        };

        Some(TileDetection {
            x,
            y,
            w,
            h,
            confidence,
            class_id,
            class_name,
            tile_idx: 0,
            vx: None, // ByteTrack doesn't expose velocity directly
            vy: None,
            track_id: Some(track_id),
        })
    }
}

/// Helper function to calculate IoU for box matching
fn calculate_iou_boxes(
    x1: f32,
    y1: f32,
    w1: f32,
    h1: f32,
    x2: f32,
    y2: f32,
    w2: f32,
    h2: f32,
) -> f32 {
    let x1_max = x1 + w1;
    let y1_max = y1 + h1;
    let x2_max = x2 + w2;
    let y2_max = y2 + h2;

    let inter_x_min = x1.max(x2);
    let inter_y_min = y1.max(y2);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
        return 0.0;
    }

    let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    let area1 = w1 * h1;
    let area2 = w2 * h2;
    let union_area = area1 + area2 - inter_area;

    inter_area / union_area
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytetrack_tracker() {
        let config = ByteTrackConfig::default();
        let mut tracker = MultiObjectTracker::new(config);

        let detections = vec![TileDetection {
            x: 100.0,
            y: 100.0,
            w: 50.0,
            h: 50.0,
            confidence: 0.9,
            class_id: 0,
            class_name: "person".to_string(),
            tile_idx: 0,
            vx: None,
            vy: None,
            track_id: None,
        }];

        let tracked = tracker.update(&detections, 0.033);
        assert!(!tracked.is_empty());

        // Check that track ID was assigned
        assert!(tracked[0].track_id.is_some());
    }

    #[test]
    fn test_iou_calculation() {
        let iou = calculate_iou_boxes(0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0);
        assert!(iou > 0.0 && iou < 1.0);
    }
}
