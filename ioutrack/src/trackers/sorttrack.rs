//! SORT (Simple Online Real-time Tracking) implementation
//! Based on the original Python SORT algorithm adapted for pure Rust

use crate::bbox::Bbox;
use crate::box_tracker::{KalmanBoxTracker, KalmanBoxTrackerParams};
use crate::hungarian::HungarianSolver;
use crate::spatial::SpatialTracker;
use ndarray::prelude::*;
use num::cast;
use rayon::prelude::*;
use std::collections::BTreeMap;

type ScoreBoxes = Vec<(f32, Bbox<f32>, u32)>;

/// SORT (Simple Online Real-time Tracking) tracker
#[derive(Debug, Clone)]
pub struct SortMultiTracker {
    pub max_age: u32,
    pub min_hits: u32,
    pub iou_threshold: f32,
    pub init_tracker_min_score: f32,
    next_track_id: u32,
    measurement_noise: [f32; 4],
    process_noise: [f32; 7],
    pub tracklets: BTreeMap<u32, KalmanBoxTracker>,
    pub n_steps: u32,
    // Memory pools for reused allocations
    temp_track_data: Vec<Vec<f32>>,
    temp_assignments: Vec<(u32, Bbox<f32>, u32)>,
    temp_unmatched: Vec<(f32, Bbox<f32>, u32)>,
    // Spatial indexing for dense scenarios
    spatial_tracker: SpatialTracker,
    spatial_threshold: u32,
}

impl SortMultiTracker {
    pub fn new(
        max_age: u32,
        min_hits: u32,
        iou_threshold: f32,
        init_tracker_min_score: f32,
        measurement_noise: [f32; 4],
        process_noise: [f32; 7],
    ) -> Self {
        SortMultiTracker {
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            next_track_id: 1,
            measurement_noise,
            process_noise,
            tracklets: BTreeMap::new(),
            n_steps: 0,
            temp_track_data: Vec::new(),
            temp_assignments: Vec::new(),
            temp_unmatched: Vec::new(),
            spatial_tracker: SpatialTracker::new(100.0),
            spatial_threshold: 100,
        }
    }

    /// Predict new positions for all tracklets and cleanup invalid ones
    pub fn predict_and_cleanup(&mut self) -> Array2<f32> {
        // Clear and reuse temporary storage
        self.temp_track_data.clear();

        // Parallel prediction and validation
        let track_data: Vec<(u32, Vec<f32>)> = self
            .tracklets
            .par_iter_mut()
            .filter_map(|(track_id, tracklet)| {
                let b = tracklet.predict();
                let bounds = b.to_bounds();

                // Validate bounds
                if b.xmin >= b.xmax || b.ymin >= b.ymax || bounds.iter().any(|x| !x.is_normal()) {
                    None
                } else {
                    let mut data = bounds.to_vec();
                    data.push(cast(tracklet.id).unwrap());
                    Some((*track_id, data))
                }
            })
            .collect();

        // Update tracklets map (remove invalid ones)
        let valid_ids: std::collections::HashSet<u32> =
            track_data.iter().map(|(id, _)| *id).collect();
        self.tracklets.retain(|id, _| valid_ids.contains(id));

        // Flatten data for array creation using pre-allocated capacity
        let total_elements = track_data.len() * 5;
        let mut data = Vec::with_capacity(total_elements);
        for (_, track_data) in track_data {
            data.extend(track_data);
        }

        if data.is_empty() {
            Array2::zeros((0, 5))
        } else {
            Array2::from_shape_vec((data.len() / 5, 5), data).unwrap()
        }
    }

    /// Get current tracklet boxes
    pub fn get_tracklet_boxes(&self, return_all: bool, return_indices: bool) -> Array2<f32> {
        // Parallel processing of tracklets
        let track_data: Vec<Vec<f32>> = self
            .tracklets
            .par_iter()
            .filter_map(|(_, tracklet)| {
                if return_all
                    || (tracklet.steps_since_update < 1
                        && (tracklet.hit_streak >= self.min_hits || self.n_steps <= self.min_hits))
                {
                    let mut data = tracklet.bbox().to_bounds().to_vec();
                    data.push(cast(tracklet.id).unwrap());

                    if return_indices {
                        data.push(tracklet.det_idx as f32);
                    }
                    Some(data)
                } else {
                    None
                }
            })
            .collect();

        // Flatten data
        let dim = if return_indices { 6 } else { 5 };
        let mut data = Vec::with_capacity(track_data.len() * dim);
        for track_data in track_data {
            data.extend(track_data);
        }

        if data.is_empty() {
            Array2::zeros((0, dim))
        } else {
            Array2::from_shape_vec((data.len() / dim, dim), data).unwrap()
        }
    }

    /// Create new tracklets from unmatched detections
    pub fn create_tracklets(&mut self, score_boxes: ScoreBoxes) {
        for (score, bbox, det_idx) in score_boxes {
            if score >= self.init_tracker_min_score {
                self.tracklets.insert(
                    self.next_track_id,
                    KalmanBoxTracker::new(
                        KalmanBoxTrackerParams {
                            id: self.next_track_id,
                            bbox,
                            meas_var: Some(self.measurement_noise),
                            proc_var: Some(self.process_noise),
                        },
                        det_idx,
                    ),
                );
                self.next_track_id += 1;
            }
        }
    }

    /// Update tracklets with detections using pooled memory
    pub fn update_tracklets(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        tracklet_boxes: ArrayView2<f32>,
    ) -> anyhow::Result<ScoreBoxes> {
        // Clear reused vectors
        self.temp_assignments.clear();
        self.temp_unmatched.clear();

        // Use spatial indexing for dense scenarios
        let num_tracklets = tracklet_boxes.nrows() as u32;
        if num_tracklets >= self.spatial_threshold {
            self.update_tracklets_spatial(detection_boxes, tracklet_boxes)
        } else {
            self.update_tracklets_standard(detection_boxes, tracklet_boxes)
        }
    }

    fn update_tracklets_standard(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        tracklet_boxes: ArrayView2<f32>,
    ) -> anyhow::Result<ScoreBoxes> {
        // Use optimized IOU calculation if available
        let det_track_ious = crate::bbox::ious_optimized(detection_boxes, tracklet_boxes);
        let assignment_result =
            HungarianSolver::solve_iou(det_track_ious.view(), self.iou_threshold);

        // Process assignments using pooled memory
        for (det_idx, track_idx) in assignment_result.assignments {
            if det_idx >= detection_boxes.nrows() || track_idx >= tracklet_boxes.nrows() {
                continue;
            }

            let det_row = detection_boxes.row(det_idx);
            if det_row.len() < 5 {
                continue;
            }

            let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
            let score = det_row[4];
            let iou = det_track_ious[(det_idx, track_idx)];

            if iou > self.iou_threshold {
                let track_row = tracklet_boxes.row(track_idx);
                if track_row.len() >= 5 {
                    let track_id = track_row[4] as u32;
                    self.temp_assignments
                        .push((track_id, det_box, det_idx as u32));
                }
            } else {
                self.temp_unmatched.push((score, det_box, det_idx as u32));
            }
        }

        // Process unassigned detections
        for det_idx in assignment_result.unassigned_detections {
            if det_idx >= detection_boxes.nrows() {
                continue;
            }

            let det_row = detection_boxes.row(det_idx);
            if det_row.len() < 5 {
                continue;
            }

            let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
            let score = det_row[4];
            self.temp_unmatched.push((score, det_box, det_idx as u32));
        }

        // Update tracklets
        for (track_id, bbox, det_idx) in &self.temp_assignments {
            if let Some(tracklet) = self.tracklets.get_mut(track_id) {
                tracklet.det_idx = *det_idx;
                if tracklet.update(bbox.clone()).is_err() {
                    // Failed to update Kalman filter, remove tracklet
                    self.tracklets.remove(track_id);
                }
            }
        }

        // Return cloned unmatched detections (avoiding move)
        Ok(self.temp_unmatched.clone())
    }

    fn update_tracklets_spatial(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        tracklet_boxes: ArrayView2<f32>,
    ) -> anyhow::Result<ScoreBoxes> {
        // Convert arrays to spatial format
        let mut detections = Vec::new();
        for i in 0..detection_boxes.nrows() {
            let row = detection_boxes.row(i);
            if row.len() >= 5 {
                detections.push([row[0], row[1], row[2], row[3]]);
            }
        }

        let mut tracks = Vec::new();
        for i in 0..tracklet_boxes.nrows() {
            let row = tracklet_boxes.row(i);
            if row.len() >= 5 {
                tracks.push([row[0], row[1], row[2], row[3]]);
            }
        }

        // Update spatial tracker with current frame data
        self.spatial_tracker.update(&detections, &tracks);

        // Generate candidate pairs using spatial indexing
        let candidate_pairs = self.spatial_tracker.get_candidate_pairs();

        // Calculate IOUs only for candidate pairs - create sparse matrix
        let mut iou_matrix = Array2::zeros((detection_boxes.nrows(), tracklet_boxes.nrows()));
        for (det_idx, track_idx) in candidate_pairs {
            if det_idx >= detection_boxes.nrows() || track_idx >= tracklet_boxes.nrows() {
                continue;
            }

            let det_row = detection_boxes.row(det_idx);
            let track_row = tracklet_boxes.row(track_idx);

            if det_row.len() < 5 || track_row.len() < 5 {
                continue;
            }

            let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
            let track_box = Bbox::new(track_row[0], track_row[1], track_row[2], track_row[3]);

            let iou = crate::bbox::calculate_iou(&det_box, &track_box);
            iou_matrix[[det_idx, track_idx]] = iou;
        }

        // Use Hungarian algorithm with sparse matrix
        let assignment_result = HungarianSolver::solve_iou(iou_matrix.view(), self.iou_threshold);

        // Process assignments
        for (det_idx, track_idx) in assignment_result.assignments {
            if det_idx >= detection_boxes.nrows() || track_idx >= tracklet_boxes.nrows() {
                continue;
            }

            let det_row = detection_boxes.row(det_idx);
            if det_row.len() < 5 {
                continue;
            }

            let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
            let iou = iou_matrix[(det_idx, track_idx)];

            if iou > self.iou_threshold {
                let track_row = tracklet_boxes.row(track_idx);
                if track_row.len() >= 5 {
                    let track_id = track_row[4] as u32;
                    self.temp_assignments
                        .push((track_id, det_box, det_idx as u32));
                }
            } else {
                let score = det_row[4];
                self.temp_unmatched.push((score, det_box, det_idx as u32));
            }
        }

        // Process unassigned detections
        for det_idx in assignment_result.unassigned_detections {
            if det_idx >= detection_boxes.nrows() {
                continue;
            }

            let det_row = detection_boxes.row(det_idx);
            if det_row.len() < 5 {
                continue;
            }

            let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
            let score = det_row[4];
            self.temp_unmatched.push((score, det_box, det_idx as u32));
        }

        // Update tracklets
        for (track_id, bbox, det_idx) in &self.temp_assignments {
            if let Some(tracklet) = self.tracklets.get_mut(track_id) {
                tracklet.det_idx = *det_idx;
                if tracklet.update(bbox.clone()).is_err() {
                    // Failed to update Kalman filter, remove tracklet
                    self.tracklets.remove(track_id);
                }
            }
        }

        // Return cloned unmatched detections (avoiding move)
        Ok(self.temp_unmatched.clone())
    }

    /// Remove tracklets that haven't been updated for too long
    pub fn remove_stale_tracklets(&mut self) {
        self.tracklets
            .retain(|_, tracklet| tracklet.steps_since_update <= self.max_age);
    }

    /// Main update function - the core SORT algorithm
    pub fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        // Step 1: Predict new positions for all tracklets
        let tracklet_boxes = self.predict_and_cleanup();

        // Step 2: Associate detections with tracklets using Hungarian algorithm
        let unmatched_detections = self.update_tracklets(detection_boxes, tracklet_boxes.view())?;

        // Step 3: Remove stale tracklets
        self.remove_stale_tracklets();

        // Step 4: Create new tracklets from unmatched detections
        self.create_tracklets(unmatched_detections);

        // Step 5: Increment step counter
        self.n_steps += 1;

        // Step 6: Return current tracklet boxes
        Ok(self.get_tracklet_boxes(return_all, return_indices))
    }

    /// Clear all trackers
    pub fn clear_trackers(&mut self) {
        self.tracklets.clear();
        self.next_track_id = 1;
        self.n_steps = 0;
    }

    /// Remove specific tracker
    pub fn remove_tracker(&mut self, track_id: u32) {
        self.tracklets.remove(&track_id);
    }

    /// Get current number of trackers
    pub fn num_trackers(&self) -> usize {
        self.tracklets.len()
    }

    /// Set the threshold for switching to spatial indexing
    pub fn set_spatial_threshold(&mut self, threshold: u32) {
        self.spatial_threshold = threshold;
    }

    /// Get spatial indexing statistics
    pub fn get_spatial_stats(&self) -> (usize, usize, f64) {
        let stats = self.spatial_tracker.efficiency_stats();
        (
            stats.candidate_pairs,
            stats.total_possible_pairs,
            stats.reduction_ratio as f64,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::array;

    #[test]
    fn test_sort_basic() {
        let mut tracker = SortMultiTracker::new(
            5,    // max_age
            2,    // min_hits
            0.3,  // iou_threshold
            0.25, // init_tracker_min_score
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        let detections = array![
            [10.0, 10.0, 50.0, 50.0, 0.9],
            [60.0, 60.0, 100.0, 100.0, 0.8],
        ];

        let tracks = tracker.update(detections.view(), false, false).unwrap();
        assert_eq!(tracks.ncols(), 5);

        // Test clear
        tracker.clear_trackers();
        assert_eq!(tracker.num_trackers(), 0);
    }

    #[test]
    fn test_sort_filter() {
        let mut tracker = SortMultiTracker::new(
            1,
            3,
            0.3,
            0.3,
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        // Test with invalid detection (infinity)
        let detections = array![
            [f32::INFINITY, 1.5, 12.6, 25.0, 0.9],
            [-5.5, 18.0, 1.0, 20.0, 0.15]
        ];

        tracker.update(detections.view(), false, false).unwrap();
        let res = tracker.predict_and_cleanup();
        assert!(res.shape()[0] == 0); // Should filter out invalid detections
    }

    #[test]
    fn test_first_update() {
        let mut tracker = SortMultiTracker::new(
            1,
            3,
            0.3,
            0.3,
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        let detections = array![[0.0, 1.5, 12.6, 25.0, 0.9], [-5.5, 18.0, 1.0, 20.0, 0.15]];

        let tracks = tracker.update(detections.view(), false, false).unwrap();

        // Only one detection should pass the score threshold
        assert_eq!(tracks.nrows(), 1);
        assert_abs_diff_eq!(tracks[[0, 4]], 1.0, epsilon = 0.00001); // Track ID should be 1
    }
}
