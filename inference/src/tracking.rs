/// Tracking method selection for video pipeline
use crate::frame_pipeline::TileDetection;
use crate::tracking_utils::{
    calculate_iou, tile_detection_to_tracker_format, tracker_output_to_tile_detection, BoundingBox,
};
use ioutrack::{ByteMultiTracker, KalmanMultiTracker, MultiObjectTracker};
use rayon::prelude::*;
use std::collections::HashMap;

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
    /// Store original detection information by track ID with pre-allocated capacity
    track_metadata: HashMap<u32, TrackMetadata>,
    /// Track ages by track ID (number of frames since last detection)
    track_ages: HashMap<u32, u32>,
    /// Track update count for debugging
    update_count: u64,
    /// Pre-allocated vectors for IoU matching performance
    temp_iou_results: Vec<IoUMatch>,
}

/// Metadata for each track
#[derive(Clone, Debug)]
struct TrackMetadata {
    class_id: u32,
    class_name: String,
    last_confidence: f32,
}

/// IoU matching result for batch processing
#[derive(Clone, Copy, Debug)]
struct IoUMatch {
    detection_idx: usize,
    track_id: u32,
    iou_score: f32,
}

impl std::fmt::Debug for UnifiedTracker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UnifiedTracker")
            .field("method", &self.method)
            .field("track_metadata", &self.track_metadata)
            .field("track_ages", &self.track_ages)
            .field("update_count", &self.update_count)
            .field("num_tracks", &self.tracker.num_tracklets())
            .finish_non_exhaustive()
    }
}

impl Clone for UnifiedTracker {
    fn clone(&self) -> Self {
        // We need to recreate the tracker since Box<dyn Trait> can't be cloned
        // For now, create a new tracker with default config of the same method
        let config = match self.method {
            TrackingMethod::Kalman => {
                TrackingConfig::Kalman(crate::kalman_operator::KalmanConfig::default())
            }
            TrackingMethod::ByteTrack => {
                TrackingConfig::ByteTrack(crate::bytetrack_operator::ByteTrackConfig::default())
            }
        };

        let new_tracker = Self::new(config);

        // Note: We lose the internal state of the tracker when cloning
        // This is a limitation of the trait object approach
        Self {
            tracker: new_tracker.tracker,
            method: self.method,
            track_metadata: self.track_metadata.clone(),
            track_ages: self.track_ages.clone(),
            update_count: self.update_count,
            temp_iou_results: Vec::with_capacity(128),
        }
    }
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
            track_metadata: HashMap::with_capacity(64), // Pre-allocate for common use cases
            track_ages: HashMap::with_capacity(64),     // Pre-allocate for track age tracking
            update_count: 0,
            temp_iou_results: Vec::with_capacity(128), // Pre-allocate for IoU matching
        }
    }

    /// Update tracker with new detections
    pub fn update(&mut self, detections: &[TileDetection], _dt: f32) -> Vec<TileDetection> {
        self.update_count += 1;

        // For real detections (non-empty), purge tracks older than 1 frames
        /*if !detections.is_empty() {
            self.purge_old_tracks(1);
        }
        */

        if detections.is_empty() {
            log::debug!(
                "UnifiedTracker::update called with 0 detections (update #{})",
                self.update_count
            );

            // Age all tracks by 1 frame when no detections
            self.age_all_tracks();
            return Vec::new();
        }

        log::debug!(
            "UnifiedTracker::update called with {} detections (update #{})",
            detections.len(),
            self.update_count
        );

        // Fast path for single detection
        if detections.len() == 1 {
            return self.update_single_detection(&detections[0]);
        }

        // Convert TileDetection to tracker format
        let detection_array = tile_detection_to_tracker_format(detections);

        log::debug!(
            "Converted to tracker array with shape: {:?}",
            detection_array.shape()
        );

        // Validation of detections using static validation function
        let valid_count = detections
            .par_iter()
            .map(|det| is_detection_valid(det))
            .filter(|&is_valid| is_valid)
            .count();

        log::debug!("Valid detections: {}/{}", valid_count, detections.len());

        // Use unified interface
        match self.tracker.update(detection_array.view(), true, false) {
            Ok(tracks) => {
                log::debug!("Tracker returned {} tracks", tracks.nrows());

                if tracks.is_empty() {
                    log::warn!(
                        "Tracker returned empty tracks array despite {} valid input detections",
                        valid_count
                    );
                    return Vec::new();
                }

                // Convert back to TileDetection
                let mut tracked_detections = tracker_output_to_tile_detection(&tracks);

                // IoU matching for metadata assignment
                self.update_track_metadata(detections, &mut tracked_detections);

                // Update track ages based on active tracks
                let active_track_ids: Vec<u32> = tracked_detections
                    .iter()
                    .filter_map(|det| det.track_id)
                    .collect();
                self.update_track_ages(&active_track_ids);

                tracked_detections
            }
            Err(e) => {
                log::error!("Tracker update failed: {:?}", e);
                Vec::new()
            }
        }
    }

    /// Fast path for single detection update
    fn update_single_detection(&mut self, detection: &TileDetection) -> Vec<TileDetection> {
        if !is_detection_valid(detection) {
            return Vec::new();
        }

        // Create single-element array
        let detection_data = vec![
            detection.x,
            detection.y,
            detection.x + detection.w,
            detection.y + detection.h,
            detection.confidence,
        ];

        let detection_array = ndarray::Array2::from_shape_vec((1, 5), detection_data)
            .unwrap_or_else(|_| ndarray::Array2::zeros((0, 5)));

        match self.tracker.update(detection_array.view(), true, false) {
            Ok(tracks) => {
                let mut tracked_detections = tracker_output_to_tile_detection(&tracks);

                // Simple metadata update for single detection
                if let Some(tracked_det) = tracked_detections.first_mut() {
                    if let Some(track_id) = tracked_det.track_id {
                        self.track_metadata.insert(
                            track_id,
                            TrackMetadata {
                                class_id: detection.class_id,
                                class_name: detection.class_name.clone(),
                                last_confidence: detection.confidence,
                            },
                        );

                        tracked_det.class_id = detection.class_id;
                        tracked_det.class_name = detection.class_name.clone();
                        tracked_det.confidence = detection.confidence;

                        // Update track ages for single detection
                        self.update_track_ages(&[track_id]);
                    }
                }

                tracked_detections
            }
            Err(_) => Vec::new(),
        }
    }

    /// IoU matching and metadata update
    fn update_track_metadata(
        &mut self,
        detections: &[TileDetection],
        tracked_detections: &mut [TileDetection],
    ) {
        if detections.is_empty() || tracked_detections.is_empty() {
            return;
        }

        // Pre-compute bounding boxes for all detections and tracks
        let det_boxes: Vec<BoundingBox> = detections
            .par_iter()
            .map(|det| BoundingBox {
                x1: det.x,
                y1: det.y,
                x2: det.x + det.w,
                y2: det.y + det.h,
            })
            .collect();

        let track_boxes: Vec<(u32, BoundingBox)> = tracked_detections
            .par_iter()
            .filter_map(|tracked_det| {
                tracked_det.track_id.map(|id| {
                    (
                        id,
                        BoundingBox {
                            x1: tracked_det.x,
                            y1: tracked_det.y,
                            x2: tracked_det.x + tracked_det.w,
                            y2: tracked_det.y + tracked_det.h,
                        },
                    )
                })
            })
            .collect();

        // IoU computation
        self.temp_iou_results.clear();
        self.temp_iou_results
            .reserve(detections.len() * tracked_detections.len());

        let iou_results: Vec<IoUMatch> = det_boxes
            .par_iter()
            .enumerate()
            .flat_map(|(det_idx, det_box)| {
                track_boxes
                    .par_iter()
                    .filter_map(move |(track_id, track_box)| {
                        let iou = calculate_iou(det_box, track_box);
                        if iou > 0.3 {
                            Some(IoUMatch {
                                detection_idx: det_idx,
                                track_id: *track_id,
                                iou_score: iou,
                            })
                        } else {
                            None
                        }
                    })
            })
            .collect();

        // Find best matches using parallel grouping and reduction
        let best_matches: HashMap<u32, (usize, f32)> = iou_results
            .into_par_iter()
            .fold(HashMap::<u32, (usize, f32)>::new, |mut acc, iou_match| {
                let current_best = acc.get(&iou_match.track_id);
                if current_best.map_or(true, |(_, current_iou)| iou_match.iou_score > *current_iou)
                {
                    acc.insert(
                        iou_match.track_id,
                        (iou_match.detection_idx, iou_match.iou_score),
                    );
                }
                acc
            })
            .reduce(HashMap::<u32, (usize, f32)>::new, |mut acc1, acc2| {
                for (track_id, (det_idx, iou_score)) in acc2 {
                    let current_best = acc1.get(&track_id);
                    if current_best.map_or(true, |(_, current_iou)| iou_score > *current_iou) {
                        acc1.insert(track_id, (det_idx, iou_score));
                    }
                }
                acc1
            });

        // Update metadata based on best matches
        for (track_id, (det_idx, iou_score)) in best_matches {
            if let Some(detection) = detections.get(det_idx) {
                log::debug!(
                    "Detection {} matched to track {} with IoU={:.3}",
                    det_idx,
                    track_id,
                    iou_score
                );

                self.track_metadata.insert(
                    track_id,
                    TrackMetadata {
                        class_id: detection.class_id,
                        class_name: detection.class_name.clone(),
                        last_confidence: detection.confidence,
                    },
                );
            }
        }

        // Apply metadata to tracked detections
        for tracked_det in tracked_detections {
            if let Some(track_id) = tracked_det.track_id {
                if let Some(metadata) = self.track_metadata.get(&track_id) {
                    tracked_det.class_id = metadata.class_id;
                    tracked_det.class_name = metadata.class_name.clone();
                    tracked_det.confidence = metadata.last_confidence;
                }
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

    /// Purge tracks older than the specified max age (in frames)
    fn _purge_old_tracks(&mut self, max_age: u32) {
        // Parallel filtering to find old tracks
        let old_track_ids: Vec<u32> = self
            .track_ages
            .par_iter()
            .filter_map(
                |(&track_id, &age)| {
                    if age > max_age {
                        Some(track_id)
                    } else {
                        None
                    }
                },
            )
            .collect();

        for track_id in old_track_ids {
            log::debug!(
                "Purging old track {} with age {} frames",
                track_id,
                self.track_ages.get(&track_id).unwrap_or(&0)
            );

            // Remove from tracker
            self.tracker.remove_tracker(track_id);

            // Remove from our metadata
            self.track_metadata.remove(&track_id);
            self.track_ages.remove(&track_id);
        }
    }

    /// Age all tracks by 1 frame (called when no detections are present)
    fn age_all_tracks(&mut self) {
        // Convert to parallel iterator for aging all tracks
        let updated_ages: HashMap<u32, u32> = self
            .track_ages
            .par_iter()
            .map(|(&track_id, &age)| (track_id, age + 1))
            .collect();

        self.track_ages = updated_ages;
    }

    /// Update track ages based on current active tracks
    fn update_track_ages(&mut self, active_track_ids: &[u32]) {
        // Parallel aging of all tracks by 1 frame
        let mut updated_ages: HashMap<u32, u32> = self
            .track_ages
            .par_iter()
            .map(|(&track_id, &age)| (track_id, age + 1))
            .collect();

        // Sequential update for matched tracks (HashMap mutation not thread-safe)
        for &track_id in active_track_ids {
            updated_ages.insert(track_id, 0);
        }

        // Parallel filtering to keep only existing tracks
        let existing_track_ids: std::collections::HashSet<u32> =
            active_track_ids.par_iter().copied().collect();

        let filtered_ages: HashMap<u32, u32> = updated_ages
            .par_iter()
            .filter_map(|(&track_id, &age)| {
                if existing_track_ids.contains(&track_id) {
                    Some((track_id, age))
                } else {
                    None
                }
            })
            .collect();

        self.track_ages = filtered_ages;
    }
}

/// Static detection validation function
#[inline]
fn is_detection_valid(detection: &TileDetection) -> bool {
    detection.x.is_finite()
        && detection.y.is_finite()
        && detection.w.is_finite()
        && detection.h.is_finite()
        && detection.confidence.is_finite()
        && detection.x >= 0.0
        && detection.y >= 0.0
        && detection.w > 0.0
        && detection.h > 0.0
        && detection.confidence >= 0.0
        && detection.confidence <= 1.0
}
