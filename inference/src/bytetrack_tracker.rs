/// ByteTrack tracker implementation using ioutrack_rs
///
/// ByteTrack is a more modern tracking algorithm compared to Kalman filtering.
/// It uses motion prediction and data association with high/low confidence detection handling.
///
/// Reference: https://arxiv.org/abs/2110.06864
use crate::frame_pipeline::TileDetection;
use ioutrack_rs::{ByteTracker, Detection as IOUDetection, Track};
use std::time::Instant;

/// Configuration for ByteTrack algorithm
#[derive(Clone, Debug)]
pub struct ByteTrackConfig {
    /// High confidence threshold for detections (0-1)
    pub high_conf_threshold: f32,
    /// Low confidence threshold for detections (0-1)
    pub low_conf_threshold: f32,
    /// Matching threshold for high confidence detections
    pub match_thresh: f32,
    /// Frame rate of the input video (used for motion prediction)
    pub frame_rate: f32,
    /// Number of frames to keep a track without detections before deletion
    pub track_buffer: u32,
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            high_conf_threshold: 0.5,
            low_conf_threshold: 0.1,
            match_thresh: 0.8,
            frame_rate: 30.0,
            track_buffer: 30,
        }
    }
}

/// Multi-object tracker using ByteTrack algorithm
pub struct MultiObjectTracker {
    tracker: ByteTracker,
    config: ByteTrackConfig,
    frame_id: u32,
}

impl MultiObjectTracker {
    /// Create new ByteTrack tracker
    pub fn new(config: ByteTrackConfig) -> Self {
        let tracker = ByteTracker::new(config.frame_rate, config.track_buffer);

        Self {
            tracker,
            config,
            frame_id: 0,
        }
    }

    /// Update tracker with new detections
    pub fn update(&mut self, detections: &[TileDetection], _dt: f32) -> Vec<TileDetection> {
        self.frame_id += 1;

        // Convert TileDetections to IOUDetections
        let iou_detections: Vec<IOUDetection> = detections
            .iter()
            .enumerate()
            .map(|(idx, det)| {
                IOUDetection::new(
                    idx as u32, // Use index as detection ID
                    det.x,
                    det.y,
                    det.x + det.w,
                    det.y + det.h,
                    det.confidence,
                    det.class_id as i32,
                )
            })
            .collect();

        // Update tracker
        let tracks = self.tracker.update(&iou_detections);

        // Convert tracks back to TileDetections
        let mut tracked_detections = Vec::new();
        for track in tracks {
            if let Some(detection) = self.track_to_tile_detection(&track, detections) {
                tracked_detections.push(detection);
            }
        }

        tracked_detections
    }

    /// Get current predictions (for ByteTrack, this is the same as the last update result)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        // ByteTrack doesn't have a separate prediction step like Kalman
        // The tracks from the last update are the current predictions
        Vec::new() // Return empty for now, as we need to store the last result
    }

    /// Get number of active tracks
    pub fn num_tracks(&self) -> usize {
        // This would need to be stored from the last update
        0 // Placeholder
    }

    /// Get track by ID
    pub fn get_track(&self, _track_id: u32) -> Option<&Track> {
        // This would require internal track storage
        None // Placeholder
    }

    /// Evict stale tracks (ByteTrack handles this internally)
    pub fn evict_stale_tracks(&mut self, _max_age_ms: u64) {
        // ByteTracker handles track lifecycle internally
        // No manual eviction needed
    }

    /// Convert ByteTrack Track to TileDetection
    fn track_to_tile_detection(
        &self,
        track: &Track,
        original_detections: &[TileDetection],
    ) -> Option<TileDetection> {
        let bbox = track.get_rect();

        // Extract track information
        let x = bbox.x;
        let y = bbox.y;
        let w = bbox.width;
        let h = bbox.height;
        let track_id = track.get_track_id();

        // Try to find the corresponding original detection to get class info
        // For now, we'll use a default class if we can't match
        let (class_id, class_name, confidence) = if let Some(det_idx) = track.get_detection_id() {
            if let Some(original_det) = original_detections.get(det_idx as usize) {
                (
                    original_det.class_id,
                    original_det.class_name.clone(),
                    original_det.confidence,
                )
            } else {
                (0, "unknown".to_string(), 0.5)
            }
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

/// Helper function to calculate IoU for debugging
#[allow(dead_code)]
fn calculate_iou(det1: &TileDetection, det2: &TileDetection) -> f32 {
    let x1_min = det1.x;
    let y1_min = det1.y;
    let x1_max = det1.x + det1.w;
    let y1_max = det1.y + det1.h;

    let x2_min = det2.x;
    let y2_min = det2.y;
    let x2_max = det2.x + det2.w;
    let y2_max = det2.y + det2.h;

    let inter_x_min = x1_min.max(x2_min);
    let inter_y_min = y1_min.max(y2_min);
    let inter_x_max = x1_max.min(x2_max);
    let inter_y_max = y1_max.min(y2_max);

    if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
        return 0.0;
    }

    let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
    let area1 = det1.w * det1.h;
    let area2 = det2.w * det2.h;
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
        let det1 = TileDetection {
            x: 0.0,
            y: 0.0,
            w: 10.0,
            h: 10.0,
            confidence: 1.0,
            class_id: 0,
            class_name: "test".to_string(),
            tile_idx: 0,
            vx: None,
            vy: None,
            track_id: None,
        };

        let det2 = TileDetection {
            x: 5.0,
            y: 5.0,
            w: 10.0,
            h: 10.0,
            confidence: 1.0,
            class_id: 0,
            class_name: "test".to_string(),
            tile_idx: 0,
            vx: None,
            vy: None,
            track_id: None,
        };

        let iou = calculate_iou(&det1, &det2);
        assert!(iou > 0.0 && iou < 1.0);
    }
}
