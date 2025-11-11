/// Tracking method selection for video pipeline
use crate::frame_pipeline::TileDetection;

/// Available tracking algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrackingMethod {
    /// Kalman filter-based tracking with Hungarian data association
    Kalman,
    /// ByteTrack algorithm (more modern, handles occlusions better)
    ByteTrack,
}

impl Default for TrackingMethod {
    fn default() -> Self {
        Self::Kalman
    }
}

impl std::fmt::Display for TrackingMethod {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Kalman => write!(f, "Kalman Filter"),
            Self::ByteTrack => write!(f, "ByteTrack"),
        }
    }
}

/// Configuration for tracking methods
#[derive(Debug, Clone)]
pub enum TrackingConfig {
    Kalman(crate::kalman_tracker::KalmanConfig),
    ByteTrack(crate::bytetrack_tracker::ByteTrackConfig),
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self::Kalman(crate::kalman_tracker::KalmanConfig::default())
    }
}

impl TrackingConfig {
    /// Get the tracking method from config
    pub fn method(&self) -> TrackingMethod {
        match self {
            Self::Kalman(_) => TrackingMethod::Kalman,
            Self::ByteTrack(_) => TrackingMethod::ByteTrack,
        }
    }
}

/// Unified tracker interface that can use either Kalman or ByteTrack
pub enum UnifiedTracker {
    Kalman(crate::kalman_tracker::MultiObjectTracker),
    ByteTrack(crate::bytetrack_tracker::MultiObjectTracker),
}

impl UnifiedTracker {
    /// Create new tracker from config
    pub fn new(config: TrackingConfig) -> Self {
        match config {
            TrackingConfig::Kalman(kalman_config) => Self::Kalman(
                crate::kalman_tracker::MultiObjectTracker::new(kalman_config),
            ),
            TrackingConfig::ByteTrack(bytetrack_config) => Self::ByteTrack(
                crate::bytetrack_tracker::MultiObjectTracker::new(bytetrack_config),
            ),
        }
    }

    /// Update tracker with new detections
    pub fn update(&mut self, detections: &[TileDetection], dt: f32) -> Vec<TileDetection> {
        match self {
            Self::Kalman(tracker) => {
                tracker.update(detections, dt);
                tracker.get_predictions()
            }
            Self::ByteTrack(tracker) => tracker.update(detections, dt),
        }
    }

    /// Get current predictions
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        match self {
            Self::Kalman(tracker) => tracker.get_predictions(),
            Self::ByteTrack(tracker) => tracker.get_predictions(),
        }
    }

    /// Get number of active tracks
    pub fn num_tracks(&self) -> usize {
        match self {
            Self::Kalman(tracker) => tracker.num_tracks(),
            Self::ByteTrack(tracker) => tracker.num_tracks(),
        }
    }

    /// Get tracking method
    pub fn method(&self) -> TrackingMethod {
        match self {
            Self::Kalman(_) => TrackingMethod::Kalman,
            Self::ByteTrack(_) => TrackingMethod::ByteTrack,
        }
    }
}
