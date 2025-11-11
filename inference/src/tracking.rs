/// Tracking method selection for video pipeline
use crate::frame_pipeline::TileDetection;
use crate::tracking_utils::{tile_detection_to_tracker_format, tracker_output_to_tile_detection};
use ioutrack::{ByteMultiTracker, KalmanMultiTracker, MultiObjectTracker};

/// Commands that can be sent to any tracking operator
#[derive(Debug)]
pub enum TrackingCommand {
    /// Update tracker with new detections
    Update {
        detections: Vec<TileDetection>,
        dt: f32,
    },
    /// Predict forward in time without detections
    Predict { dt: f32 },
    /// Shutdown the tracker thread
    Shutdown,
}

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
    Kalman(crate::kalman_operator::KalmanConfig),
    ByteTrack(crate::bytetrack_operator::ByteTrackConfig),
}

impl Default for TrackingConfig {
    fn default() -> Self {
        Self::Kalman(crate::kalman_operator::KalmanConfig::default())
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

/// Unified tracker interface that can use any tracking algorithm
pub struct UnifiedTracker {
    tracker: Box<dyn MultiObjectTracker>,
    method: TrackingMethod,
    /// Store original detection information by track ID
    track_metadata: std::collections::HashMap<u32, (u32, String, f32)>, // (class_id, class_name, last_confidence)
    /// Track update count for debugging
    update_count: u64,
}

impl UnifiedTracker {
    /// Create new tracker from config
    pub fn new(config: TrackingConfig) -> Self {
        let (tracker, method): (Box<dyn MultiObjectTracker>, TrackingMethod) = match config {
            TrackingConfig::Kalman(kalman_config) => {
                log::info!("Creating KalmanMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}", 
                    kalman_config.max_age, kalman_config.min_hits, kalman_config.iou_threshold, kalman_config.init_tracker_min_score);
                let tracker = KalmanMultiTracker::new(
                    kalman_config.max_age,
                    kalman_config.min_hits,
                    kalman_config.iou_threshold,
                    kalman_config.init_tracker_min_score,
                    kalman_config.measurement_noise,
                    kalman_config.process_noise,
                );
                (Box::new(tracker), TrackingMethod::Kalman)
            }
            TrackingConfig::ByteTrack(bytetrack_config) => {
                log::info!("Creating ByteMultiTracker with config: max_age={}, min_hits={}, iou_threshold={:.3}, init_tracker_min_score={:.3}", 
                    bytetrack_config.max_age, bytetrack_config.min_hits, bytetrack_config.iou_threshold, bytetrack_config.init_tracker_min_score);
                let tracker = ByteMultiTracker::new(
                    bytetrack_config.max_age,
                    bytetrack_config.min_hits,
                    bytetrack_config.iou_threshold,
                    bytetrack_config.init_tracker_min_score,
                    bytetrack_config.high_score_threshold,
                    bytetrack_config.low_score_threshold,
                    bytetrack_config.measurement_noise,
                    bytetrack_config.process_noise,
                );
                (Box::new(tracker), TrackingMethod::ByteTrack)
            }
        };
        Self {
            tracker,
            method,
            track_metadata: std::collections::HashMap::new(),
            update_count: 0,
        }
    }

    /// Update tracker with new detections
    pub fn update(&mut self, detections: &[TileDetection], _dt: f32) -> Vec<TileDetection> {
        self.update_count += 1;
        log::debug!(
            "UnifiedTracker::update called with {} detections (update #{})",
            detections.len(),
            self.update_count
        );

        // Convert TileDetection to tracker format
        let detection_array = tile_detection_to_tracker_format(detections);
        log::debug!(
            "Converted to tracker array with shape: {:?}",
            detection_array.shape()
        );

        // Detailed validation and debugging
        if !detection_array.is_empty() {
            let mut valid_detections = 0;
            for (i, row) in detection_array.outer_iter().enumerate() {
                let x1 = row[0];
                let y1 = row[1];
                let x2 = row[2];
                let y2 = row[3];
                let conf = row[4];

                let is_valid = x1.is_finite()
                    && y1.is_finite()
                    && x2.is_finite()
                    && y2.is_finite()
                    && conf.is_finite()
                    && x1 >= 0.0
                    && y1 >= 0.0
                    && x2 > x1
                    && y2 > y1
                    && conf >= 0.0
                    && conf <= 1.0;

                if is_valid {
                    valid_detections += 1;
                }

                log::debug!(
                    "Detection {}: x1={:.3}, y1={:.3}, x2={:.3}, y2={:.3}, conf={:.3} [{}]",
                    i,
                    x1,
                    y1,
                    x2,
                    y2,
                    conf,
                    if is_valid { "VALID" } else { "INVALID" }
                );
            }
            log::debug!(
                "Total valid detections: {}/{}",
                valid_detections,
                detection_array.nrows()
            );
        }

        // Use unified interface
        log::debug!(
            "Calling tracker.update() with {} detections",
            detection_array.nrows()
        );

        // Log the exact detection data we're sending to the tracker
        for (i, detection) in detection_array.outer_iter().enumerate() {
            if detection.len() >= 5 {
                log::debug!(
                    "INPUT Detection {}: x1={:.3}, y1={:.3}, x2={:.3}, y2={:.3}, conf={:.3}",
                    i,
                    detection[0],
                    detection[1],
                    detection[2],
                    detection[3],
                    detection[4]
                );
            }
        }

        match self.tracker.update(detection_array.view(), true, false) {
            Ok(tracks) => {
                log::debug!("Tracker returned {} tracks", tracks.nrows());
                if !tracks.is_empty() {
                    for (i, track) in tracks.outer_iter().enumerate() {
                        log::debug!(
                            "OUTPUT Track {}: x1={:.3}, y1={:.3}, x2={:.3}, y2={:.3}, id={}",
                            i,
                            track[0],
                            track[1],
                            track[2],
                            track[3],
                            track[4] as u32
                        );
                    }
                } else {
                    log::warn!(
                        "Tracker returned empty tracks array despite {} valid input detections",
                        detection_array.nrows()
                    );
                    log::debug!(
                        "Tracker config: method={}, num_tracks={}",
                        self.method,
                        self.num_tracks()
                    );
                }
                // Convert back to TileDetection and preserve metadata
                let mut tracked_detections = tracker_output_to_tile_detection(&tracks);

                // Update metadata for active tracks using IoU matching
                log::debug!(
                    "Starting IoU matching with {} detections and {} tracks",
                    detections.len(),
                    tracked_detections.len()
                );
                for (det_idx, detection) in detections.iter().enumerate() {
                    let det_x1 = detection.x;
                    let det_y1 = detection.y;
                    let det_x2 = detection.x + detection.w;
                    let det_y2 = detection.y + detection.h;

                    // Find the closest track by IoU
                    let mut best_iou = 0.0;
                    let mut best_track_id = None;

                    for tracked_det in &tracked_detections {
                        if let Some(track_id) = tracked_det.track_id {
                            let track_x1 = tracked_det.x;
                            let track_y1 = tracked_det.y;
                            let track_x2 = tracked_det.x + tracked_det.w;
                            let track_y2 = tracked_det.y + tracked_det.h;

                            let iou = calculate_iou(
                                det_x1, det_y1, det_x2, det_y2, track_x1, track_y1, track_x2,
                                track_y2,
                            );

                            log::debug!(
                                "Detection {}: [{:.1},{:.1},{:.1},{:.1}] vs Track {}: [{:.1},{:.1},{:.1},{:.1}] -> IoU={:.3}",
                                det_idx, det_x1, det_y1, det_x2, det_y2,
                                track_id, track_x1, track_y1, track_x2, track_y2, iou
                            );

                            if iou > best_iou && iou > 0.3 {
                                // IoU threshold for association
                                best_iou = iou;
                                best_track_id = Some(track_id);
                            }
                        }
                    }

                    // Update metadata for the matched track
                    if let Some(track_id) = best_track_id {
                        log::debug!(
                            "Detection {} matched to track {} with IoU={:.3}",
                            det_idx,
                            track_id,
                            best_iou
                        );
                        self.track_metadata.insert(
                            track_id,
                            (
                                detection.class_id,
                                detection.class_name.clone(),
                                detection.confidence,
                            ),
                        );
                    } else {
                        log::debug!(
                            "Detection {} unmatched (best IoU: {:.3})",
                            det_idx,
                            best_iou
                        );
                    }
                }

                // Apply metadata to tracked detections
                for tracked_det in &mut tracked_detections {
                    if let Some(track_id) = tracked_det.track_id {
                        if let Some((class_id, class_name, confidence)) =
                            self.track_metadata.get(&track_id)
                        {
                            tracked_det.class_id = *class_id;
                            tracked_det.class_name = class_name.clone();
                            tracked_det.confidence = *confidence;
                        }
                    }
                }

                tracked_detections
            }
            Err(e) => {
                log::error!("Tracker update failed: {:?}", e);
                Vec::new()
            }
        }
    }

    /// Get current predictions (not directly supported by unified interface)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        // The unified interface doesn't expose direct prediction access
        // For now, return empty vec. In practice, use update() to get results
        Vec::new()
    }

    /// Get number of active tracks
    pub fn num_tracks(&self) -> usize {
        self.tracker.num_tracklets()
    }

    /// Get tracking method
    pub fn method(&self) -> TrackingMethod {
        self.method
    }
}

/// Calculate IoU (Intersection over Union) between two bounding boxes
fn calculate_iou(
    x1_1: f32,
    y1_1: f32,
    x2_1: f32,
    y2_1: f32,
    x1_2: f32,
    y1_2: f32,
    x2_2: f32,
    y2_2: f32,
) -> f32 {
    let x1 = x1_1.max(x1_2);
    let y1 = y1_1.max(y1_2);
    let x2 = x2_1.min(x2_2);
    let y2 = y2_1.min(y2_2);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area1 = (x2_1 - x1_1) * (y2_1 - y1_1);
    let area2 = (x2_2 - x1_2) * (y2_2 - y1_2);
    let union = area1 + area2 - intersection;

    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}
