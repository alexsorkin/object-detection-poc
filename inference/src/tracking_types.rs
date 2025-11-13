/// Configuration types for different tracking algorithms
///
/// This module contains configuration structs for various tracking methods
/// extracted from the operator modules to eliminate code duplication.

/// Configuration for Kalman filter tracking
#[derive(Clone, Debug)]
pub struct KalmanConfig {
    /// Maximum frames to keep a track alive without matching detections
    pub max_age: u32,
    /// Minimum consecutive hits before track is confirmed
    pub min_hits: u32,
    /// IoU threshold for associating detections to tracks
    pub iou_threshold: f32,
    /// Minimum score to create a new tracklet from unmatched detection
    pub init_tracker_min_score: f32,
    /// Measurement noise covariance diagonal
    pub measurement_noise: [f32; 4],
    /// Process noise covariance diagonal  
    pub process_noise: [f32; 7],
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            max_age: 10, // Allow 300ms without detection at 50Hz prediction rate (15 * 20ms)
            min_hits: 1, // Allow immediate track creation (lowered from 2)
            iou_threshold: 0.3,
            init_tracker_min_score: 0.3, // Match RT-DETR confidence levels (~30-40%)
            measurement_noise: [1.0, 1.0, 10.0, 10.0],
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        }
    }
}

/// Configuration for ByteTrack algorithm
#[derive(Clone, Debug)]
pub struct ByteTrackConfig {
    /// Maximum frames to keep a track alive without matching detections
    pub max_age: u32,
    /// Minimum consecutive hits before track is confirmed
    pub min_hits: u32,
    /// IoU threshold for associating detections to tracks
    pub iou_threshold: f32,
    /// Minimum score to create a new tracklet from unmatched detection
    pub init_tracker_min_score: f32,
    /// High confidence threshold for first association stage
    pub high_score_threshold: f32,
    /// Low confidence threshold for second association stage (recovery)
    pub low_score_threshold: f32,
    /// Measurement noise covariance diagonal
    pub measurement_noise: [f32; 4],
    /// Process noise covariance diagonal  
    pub process_noise: [f32; 7],
}

impl Default for ByteTrackConfig {
    fn default() -> Self {
        Self {
            max_age: 10, // Allow 300ms without detection at 50Hz prediction rate (15 * 20ms)
            min_hits: 1, // Allow immediate track creation (lowered from 3)
            iou_threshold: 0.3,
            init_tracker_min_score: 0.3, // Match RT-DETR confidence levels (~30-40%)
            high_score_threshold: 0.5,
            low_score_threshold: 0.1,
            measurement_noise: [1.0, 1.0, 10.0, 10.0],
            process_noise: [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        }
    }
}
