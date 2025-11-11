/// Kalman filter-based multi-object tracker with Hungarian matching
use crate::frame_pipeline::TileDetection;
use nalgebra::{Matrix4, Matrix4x6, Matrix6, Vector4, Vector6};
use rayon::prelude::*;
use std::sync::Mutex;
use std::time::Instant;

/// Configuration for Kalman filter noise parameters
#[derive(Clone, Debug)]
pub struct KalmanConfig {
    /// Process noise for position (pixels) - how much position can change unexpectedly
    pub process_noise_pos: f32,
    /// Process noise for velocity (pixels/sec) - acceleration uncertainty
    pub process_noise_vel: f32,
    /// Measurement noise (pixels) - detector bbox accuracy
    pub measurement_noise: f32,
    /// Initial state covariance
    pub initial_covariance: f32,
    /// Maximum time without measurement before track is dropped (milliseconds)
    pub max_age_ms: u64,
    /// IoU threshold for associating detections to tracks
    pub iou_threshold: f32,
    /// Maximum centroid distance for matching (pixels) - fallback when IoU=0
    pub max_centroid_distance: f32,
}

impl Default for KalmanConfig {
    fn default() -> Self {
        Self {
            process_noise_pos: 0.5,       // Moderate position uncertainty
            process_noise_vel: 2.0,       // Higher velocity uncertainty (acceleration)
            measurement_noise: 2.0,       // 2 pixels of detector jitter
            initial_covariance: 10.0,     // Initial uncertainty
            max_age_ms: 500,              // Drop tracks after 500ms
            iou_threshold: 0.3,           // Match if IoU > 30%
            max_centroid_distance: 100.0, // Match if centroids within 100px when IoU=0
        }
    }
}

/// Kalman filter for tracking single object with constant velocity model
/// State: [x, y, w, h, vx, vy] - center position, size, velocity
#[derive(Clone)]
pub struct KalmanFilter {
    /// State vector: [x, y, w, h, vx, vy]
    state: Vector6<f32>,
    /// State covariance matrix (uncertainty)
    covariance: Matrix6<f32>,
    /// Process noise covariance
    process_noise: Matrix6<f32>,
    /// Measurement noise covariance
    measurement_noise: Matrix4<f32>,
}

impl KalmanFilter {
    /// Create new Kalman filter initialized with detection
    pub fn new(detection: &TileDetection, config: &KalmanConfig) -> Self {
        // Initial state: center position, size, zero velocity
        let x = detection.x + detection.w / 2.0;
        let y = detection.y + detection.h / 2.0;
        let state = Vector6::new(x, y, detection.w, detection.h, 0.0, 0.0);

        // Initial covariance (high uncertainty for velocity)
        let mut covariance = Matrix6::identity() * config.initial_covariance;
        covariance[(4, 4)] = config.initial_covariance * 10.0; // Higher velocity uncertainty
        covariance[(5, 5)] = config.initial_covariance * 10.0;

        // Process noise (position and velocity can change)
        let mut process_noise = Matrix6::zeros();
        process_noise[(0, 0)] = config.process_noise_pos;
        process_noise[(1, 1)] = config.process_noise_pos;
        process_noise[(2, 2)] = config.process_noise_pos * 0.1; // Width changes slowly
        process_noise[(3, 3)] = config.process_noise_pos * 0.1; // Height changes slowly
        process_noise[(4, 4)] = config.process_noise_vel;
        process_noise[(5, 5)] = config.process_noise_vel;

        // Measurement noise (detector bbox accuracy)
        let measurement_noise = Matrix4::identity() * config.measurement_noise;

        Self {
            state,
            covariance,
            process_noise,
            measurement_noise,
        }
    }

    /// Predict next state based on constant velocity model
    pub fn predict(&mut self, dt: f32) {
        // State transition matrix (constant velocity model)
        // x' = x + vx * dt
        // y' = y + vy * dt
        // w' = w (size constant)
        // h' = h
        // vx' = vx (velocity constant)
        // vy' = vy
        let mut f = Matrix6::identity();
        f[(0, 4)] = dt; // x += vx * dt
        f[(1, 5)] = dt; // y += vy * dt

        // Predict state
        self.state = f * self.state;

        // Predict covariance: P' = F * P * F^T + Q
        self.covariance = f * self.covariance * f.transpose() + self.process_noise;
    }

    /// Update state with new measurement (detection)
    pub fn update(&mut self, detection: &TileDetection) {
        // Measurement: [x_center, y_center, width, height]
        let x_center = detection.x + detection.w / 2.0;
        let y_center = detection.y + detection.h / 2.0;
        let measurement = Vector4::new(x_center, y_center, detection.w, detection.h);

        // Measurement matrix H (we observe position and size, not velocity)
        let h = Matrix4x6::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, // x
            0.0, 1.0, 0.0, 0.0, 0.0, 0.0, // y
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // w
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // h
        );

        // Innovation (measurement residual)
        let predicted_measurement = h * self.state;
        let innovation = measurement - predicted_measurement;

        // Innovation covariance: S = H * P * H^T + R
        let innovation_covariance = h * self.covariance * h.transpose() + self.measurement_noise;

        // Kalman gain: K = P * H^T * S^-1
        let kalman_gain =
            self.covariance * h.transpose() * innovation_covariance.try_inverse().unwrap();

        // Update state: x = x + K * innovation
        self.state += kalman_gain * innovation;

        // Update covariance: P = (I - K * H) * P
        let identity = Matrix6::identity();
        self.covariance = (identity - kalman_gain * h) * self.covariance;
    }

    /// Get current predicted detection
    pub fn get_detection(&self, class_id: u32, class_name: &str, confidence: f32) -> TileDetection {
        let x_center = self.state[0];
        let y_center = self.state[1];
        let w = self.state[2];
        let h = self.state[3];
        let vx = self.state[4];
        let vy = self.state[5];

        TileDetection {
            x: x_center - w / 2.0,
            y: y_center - h / 2.0,
            w,
            h,
            confidence,
            class_id,
            class_name: class_name.to_string(),
            tile_idx: 0,  // Not applicable for tracked objects
            vx: Some(vx), // Include velocity from Kalman state
            vy: Some(vy),
            track_id: None, // Will be set by TrackedObject
        }
    }
}

/// Tracked object with Kalman filter state
#[derive(Clone)]
pub struct TrackedObject {
    pub track_id: u32,
    pub class_id: u32,
    pub class_name: String,
    pub kalman: KalmanFilter,
    // OPTIMIZATION: Store only essential fields instead of full detection
    pub last_x: f32,
    pub last_y: f32,
    pub last_w: f32,
    pub last_h: f32,
    pub last_seen: Instant,
    pub confidence: f32,
    pub hits: u32,   // Number of successful measurements
    pub misses: u32, // Number of prediction-only frames
}

impl TrackedObject {
    pub fn new(track_id: u32, detection: &TileDetection, config: &KalmanConfig) -> Self {
        Self {
            track_id,
            class_id: detection.class_id,
            class_name: detection.class_name.clone(),
            kalman: KalmanFilter::new(detection, config),
            last_x: detection.x,
            last_y: detection.y,
            last_w: detection.w,
            last_h: detection.h,
            last_seen: Instant::now(),
            confidence: detection.confidence,
            hits: 1,
            misses: 0,
        }
    }

    /// Predict next state
    pub fn predict(&mut self, dt: f32) {
        self.kalman.predict(dt);
        self.misses += 1;
        // Decay confidence for extrapolated frames
        self.confidence *= 0.95;
    }

    /// Update with new detection
    pub fn update(&mut self, detection: &TileDetection) {
        // Compute velocity from position change if we have a previous detection
        let dt = self.last_seen.elapsed().as_secs_f32();

        if dt > 0.001 {
            // Manually compute and set velocity in Kalman state from position difference
            let dx = (detection.x + detection.w / 2.0) - (self.last_x + self.last_w / 2.0);
            let dy = (detection.y + detection.h / 2.0) - (self.last_y + self.last_h / 2.0);
            let vx = dx / dt;
            let vy = dy / dt;

            // Directly set velocity in Kalman state (indices 4 and 5)
            self.kalman.state[4] = vx;
            self.kalman.state[5] = vy;
        }

        self.kalman.update(detection);
        // OPTIMIZATION: Store only essential fields instead of cloning entire detection
        self.last_x = detection.x;
        self.last_y = detection.y;
        self.last_w = detection.w;
        self.last_h = detection.h;
        self.last_seen = Instant::now();
        self.confidence = detection.confidence;
        self.hits += 1;
        self.misses = 0;
    }

    /// Get current detection (predicted or measured) with track ID
    pub fn get_detection(&self) -> TileDetection {
        let mut det = self
            .kalman
            .get_detection(self.class_id, &self.class_name, self.confidence);
        det.track_id = Some(self.track_id); // Add persistent track ID
        det
    }

    /// Check if track is stale (too old)
    pub fn is_stale(&self, max_age_ms: u64) -> bool {
        self.last_seen.elapsed().as_millis() > max_age_ms as u128
    }
}

/// Multi-object tracker with data association
pub struct MultiObjectTracker {
    tracks: Vec<TrackedObject>,
    next_track_id: u32,
    config: KalmanConfig,
}

impl MultiObjectTracker {
    pub fn new(config: KalmanConfig) -> Self {
        Self {
            tracks: Vec::new(),
            next_track_id: 0,
            config,
        }
    }

    /// Calculate IoU between detection and tracked object
    #[inline]
    fn calculate_iou(det: &TileDetection, track: &TrackedObject) -> f32 {
        let track_det = track.get_detection();

        let x1_min = det.x;
        let y1_min = det.y;
        let x1_max = det.x + det.w;
        let y1_max = det.y + det.h;

        let x2_min = track_det.x;
        let y2_min = track_det.y;
        let x2_max = track_det.x + track_det.w;
        let y2_max = track_det.y + track_det.h;

        let inter_x_min = x1_min.max(x2_min);
        let inter_y_min = y1_min.max(y2_min);
        let inter_x_max = x1_max.min(x2_max);
        let inter_y_max = y1_max.min(y2_max);

        if inter_x_max <= inter_x_min || inter_y_max <= inter_y_min {
            return 0.0;
        }

        let inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min);
        let area1 = det.w * det.h;
        let area2 = track_det.w * track_det.h;
        let union_area = area1 + area2 - inter_area;

        inter_area / union_area
    }

    /// Calculate centroid distance between detection and tracked object (pixels)
    #[inline]
    fn calculate_centroid_distance(det: &TileDetection, track: &TrackedObject) -> f32 {
        let track_det = track.get_detection();

        let det_cx = det.x + det.w / 2.0;
        let det_cy = det.y + det.h / 2.0;

        let track_cx = track_det.x + track_det.w / 2.0;
        let track_cy = track_det.y + track_det.h / 2.0;

        let dx = det_cx - track_cx;
        let dy = det_cy - track_cy;

        (dx * dx + dy * dy).sqrt()
    }

    /// Associate detections to existing tracks using Hungarian algorithm
    fn associate_detections(&self, detections: &[TileDetection]) -> Vec<(usize, usize)> {
        if self.tracks.is_empty() || detections.is_empty() {
            return Vec::new();
        }

        let num_detections = detections.len();
        let num_tracks = self.tracks.len();

        // Hungarian algorithm requires rows <= columns, so transpose if needed
        let transpose = num_detections > num_tracks;
        let (rows, cols) = if transpose {
            (num_tracks, num_detections)
        } else {
            (num_detections, num_tracks)
        };

        // Build cost matrix using pathfinding::matrix::Matrix with integer costs
        // Scale IoU by 10000 to preserve precision as integers
        let scale = 10000_i32;
        let max_cost = scale * 100; // Very high cost for unmatched pairs
        let cost_matrix = Mutex::new(pathfinding::matrix::Matrix::new(rows, cols, max_cost));

        // PARALLEL: Compute costs for all detection-track pairs
        let class_mismatches = std::sync::atomic::AtomicUsize::new(0);
        let costs_set = std::sync::atomic::AtomicUsize::new(0);

        detections
            .par_iter()
            .enumerate()
            .for_each(|(det_idx, det)| {
                for (track_idx, track) in self.tracks.iter().enumerate() {
                    // Only match same class
                    if det.class_id == track.class_id {
                        let iou = Self::calculate_iou(det, track);
                        let centroid_dist = Self::calculate_centroid_distance(det, track);

                        // Hybrid matching: use IoU primarily, but allow distance-based matching
                        // for cases where Kalman prediction has drifted (low IoU but close centroids)
                        let cost = if iou >= self.config.iou_threshold {
                            // Good IoU overlap - use IoU-based cost (lower is better)
                            ((1.0 - iou) * scale as f32) as i32
                        } else if centroid_dist < self.config.max_centroid_distance {
                            // Low IoU but close centroids - use distance-based cost
                            // Add penalty to prefer IoU matches over distance matches
                            let normalized_dist = centroid_dist / self.config.max_centroid_distance;
                            ((normalized_dist + 0.5) * scale as f32) as i32 // Range: 5000-15000
                        } else {
                            // Both IoU and distance too high - don't match
                            max_cost
                        };

                        // Debug: log first few costs
                        if det_idx < 2 && track_idx < 2 {
                            log::debug!(
                                "      det[{}] vs track[{}]: IoU={:.2}, dist={:.1}px, cost={}",
                                det_idx,
                                track_idx,
                                iou,
                                centroid_dist,
                                cost
                            );
                        }

                        // Place cost in correct position depending on transpose
                        let mut matrix = cost_matrix.lock().unwrap();
                        if transpose {
                            matrix[(track_idx, det_idx)] = cost;
                        } else {
                            matrix[(det_idx, track_idx)] = cost;
                        }
                        costs_set.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        class_mismatches.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                }
            });

        let costs_set_count = costs_set.load(std::sync::atomic::Ordering::Relaxed);
        let class_mismatch_count = class_mismatches.load(std::sync::atomic::Ordering::Relaxed);

        log::debug!(
            "  Cost computation: {} costs set, {} class mismatches (total pairs: {})",
            costs_set_count,
            class_mismatch_count,
            num_detections * num_tracks
        );

        let cost_matrix = cost_matrix.into_inner().unwrap();

        // Use Hungarian algorithm for optimal assignment
        let (_total_cost, assignments) = pathfinding::kuhn_munkres::kuhn_munkres(&cost_matrix);

        log::debug!(
            "  Hungarian: {} rows, {} cols, {} assignments (transpose={})",
            rows,
            cols,
            assignments.len(),
            transpose
        );

        // Debug: print cost matrix for first few items
        if log::log_enabled!(log::Level::Debug) && !self.tracks.is_empty() && !detections.is_empty()
        {
            log::debug!("  Cost matrix (first 3x3):");
            for r in 0..rows.min(3) {
                let mut row_str = String::new();
                for c in 0..cols.min(3) {
                    row_str.push_str(&format!("{:6} ", cost_matrix[(r, c)]));
                }
                log::debug!("    [{}]", row_str);
            }
        }

        // PARALLEL: Filter matches by cost threshold and convert from transpose if needed
        let matches: Vec<(usize, usize)> = assignments
            .par_iter()
            .enumerate()
            .filter_map(|(row_idx, col_idx)| {
                let (det_idx, track_idx) = if transpose {
                    (*col_idx, row_idx) // Transpose back
                } else {
                    (row_idx, *col_idx)
                };

                // Check if this is a valid match within bounds
                if det_idx >= num_detections || track_idx >= num_tracks {
                    log::debug!(
                        "    Reject: out of bounds (det={}, track={}, max_det={}, max_track={})",
                        det_idx,
                        track_idx,
                        num_detections,
                        num_tracks
                    );
                    return None;
                }

                let cost = if transpose {
                    cost_matrix[(track_idx, det_idx)]
                } else {
                    cost_matrix[(det_idx, track_idx)]
                };

                // Accept match if cost is reasonable (< max_cost)
                // This filters out dummy assignments and IoU-too-low matches
                if cost < max_cost {
                    log::debug!(
                        "    ✓ Accept: det {} → track {} (cost={})",
                        det_idx,
                        track_idx,
                        cost
                    );
                    Some((det_idx, track_idx))
                } else {
                    log::debug!(
                        "    ✗ Reject: det {} → track {} (cost={} >= max_cost)",
                        det_idx,
                        track_idx,
                        cost
                    );
                    None
                }
            })
            .collect();

        matches
    }

    /// Update tracks with new detections
    pub fn update(&mut self, detections: &[TileDetection], dt: f32) {
        // First, predict all tracks forward (sequential due to &mut requirement)
        // This is fast enough - prediction is ~50 FLOPS per track
        for track in &mut self.tracks {
            track.predict(dt);
        }

        // Associate detections to tracks (uses parallel cost matrix computation)
        let matches = self.associate_detections(detections);

        log::debug!(
            "🔍 Kalman Update: {} detections, {} tracks, {} matches (dt={:.3}s)",
            detections.len(),
            self.tracks.len(),
            matches.len(),
            dt
        );

        let mut matched_detections = vec![false; detections.len()];
        let mut matched_tracks = vec![false; self.tracks.len()];

        // Update matched tracks
        for (det_idx, track_idx) in &matches {
            let iou = Self::calculate_iou(&detections[*det_idx], &self.tracks[*track_idx]);
            let dist =
                Self::calculate_centroid_distance(&detections[*det_idx], &self.tracks[*track_idx]);
            log::debug!(
                "  ✓ Matched detection {} → track_id={} (IoU={:.2}, dist={:.1}px)",
                det_idx,
                self.tracks[*track_idx].track_id,
                iou,
                dist
            );
            self.tracks[*track_idx].update(&detections[*det_idx]);
            matched_detections[*det_idx] = true;
            matched_tracks[*track_idx] = true;
        }

        // Create new tracks for unmatched detections
        for (det_idx, det) in detections.iter().enumerate() {
            if !matched_detections[det_idx] {
                // Log why this detection didn't match any existing tracks
                let mut closest_match_info = String::from("no existing tracks");
                if !self.tracks.is_empty() {
                    let mut min_dist = f32::MAX;
                    let mut best_iou = 0.0f32;
                    for track in &self.tracks {
                        if track.class_id == det.class_id {
                            let iou = Self::calculate_iou(det, track);
                            let dist = Self::calculate_centroid_distance(det, track);
                            if dist < min_dist {
                                min_dist = dist;
                                best_iou = iou;
                            }
                        }
                    }
                    if min_dist < f32::MAX {
                        closest_match_info = format!(
                            "closest: IoU={:.2}, dist={:.1}px (threshold: IoU>{:.2}, dist<{:.1}px)",
                            best_iou,
                            min_dist,
                            self.config.iou_threshold,
                            self.config.max_centroid_distance
                        );
                    }
                }

                log::debug!(
                    "  ✨ NEW track_id={} for detection {} ({}, conf={:.2}) - {}",
                    self.next_track_id,
                    det_idx,
                    det.class_name,
                    det.confidence,
                    closest_match_info
                );
                let track = TrackedObject::new(self.next_track_id, det, &self.config);
                self.tracks.push(track);
                self.next_track_id += 1;
            }
        }

        // Remove stale tracks ONLY if we had actual detections to match against
        // (prediction-only updates should not age out tracks)
        if !detections.is_empty() {
            let before_count = self.tracks.len();
            self.tracks
                .retain(|track| !track.is_stale(self.config.max_age_ms));
            let removed_count = before_count - self.tracks.len();
            if removed_count > 0 {
                log::debug!(
                    "🗑️  Removed {} stale tracks ({} remaining)",
                    removed_count,
                    self.tracks.len()
                );
            }
        }
    }

    /// Get all current track predictions (for extrapolation)
    pub fn get_predictions(&self) -> Vec<TileDetection> {
        self.tracks
            .par_iter()
            .map(|track| track.get_detection())
            .collect()
    }

    /// Get number of active tracks
    pub fn num_tracks(&self) -> usize {
        self.tracks.len()
    }

    /// Get track by ID
    pub fn get_track(&self, track_id: u32) -> Option<&TrackedObject> {
        self.tracks.iter().find(|t| t.track_id == track_id)
    }

    /// Evict stale tracks that haven't been updated within max_age_ms
    /// This is called by the maintenance thread, not by update()
    pub fn evict_stale_tracks(&mut self, max_age_ms: u64) {
        self.tracks.retain(|track| !track.is_stale(max_age_ms));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_predict() {
        let det = TileDetection {
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
        };

        let config = KalmanConfig::default();
        let mut kf = KalmanFilter::new(&det, &config);

        // Predict forward 1 second (should stay in place with zero velocity)
        kf.predict(1.0);
        let pred = kf.get_detection(0, "person", 0.9);

        // Position should be roughly the same (within uncertainty)
        assert!((pred.x - det.x).abs() < 10.0);
        assert!((pred.y - det.y).abs() < 10.0);
    }

    #[test]
    fn test_multi_object_tracker() {
        let mut tracker = MultiObjectTracker::new(KalmanConfig::default());

        let det1 = TileDetection {
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
        };

        // First update: create new track
        tracker.update(&[det1.clone()], 0.033);
        assert_eq!(tracker.num_tracks(), 1);

        // Second update: match existing track
        let det2 = TileDetection {
            x: 105.0,
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
        };
        tracker.update(&[det2], 0.033);
        assert_eq!(tracker.num_tracks(), 1); // Should match, not create new

        // Prediction without measurement
        tracker.update(&[], 0.033);
        assert_eq!(tracker.num_tracks(), 1); // Track still alive

        let predictions = tracker.get_predictions();
        assert_eq!(predictions.len(), 1);

        // Verify track ID is assigned
        assert!(predictions[0].track_id.is_some());

        // Should have learned some velocity (moved from 100 to 105)
        // With Kalman filter, prediction should be at or slightly ahead of last measurement
        assert!(predictions[0].x >= 104.0); // At least near the last measurement
    }
}
