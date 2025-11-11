//! SORT (Simple Online Real-time Tracking) implementation
//! Based on the original Python SORT algorithm adapted for pure Rust

use crate::bbox::{ious, Bbox};
use crate::box_tracker::{KalmanBoxTracker, KalmanBoxTrackerParams};
use crate::hungarian::HungarianSolver;
use ndarray::prelude::*;
use num::cast;
use rayon::prelude::*;
use std::collections::BTreeMap;

type TrackidBoxes = Vec<(u32, Bbox<f32>, u32)>;
type ScoreBoxes = Vec<(f32, Bbox<f32>, u32)>;

/// Assign detection boxes to track boxes using Hungarian algorithm
fn assign_detections_to_tracks(
    detections: ArrayView2<f32>,
    tracks: ArrayView2<f32>,
    iou_threshold: f32,
) -> anyhow::Result<(TrackidBoxes, ScoreBoxes)> {
    let det_track_ious = ious(detections, tracks);
    let assignment_result = HungarianSolver::solve_iou(det_track_ious.view(), iou_threshold);

    let mut match_updates = Vec::new();
    let mut unmatched_dets = Vec::new();

    // Process assigned detections
    for (det_idx, track_idx) in assignment_result.assignments {
        if det_idx >= detections.nrows() || track_idx >= tracks.nrows() {
            continue;
        }

        let det_row = detections.row(det_idx);
        if det_row.len() < 5 {
            continue;
        }

        let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
        let score = det_row[4];
        let iou = det_track_ious[(det_idx, track_idx)];

        if iou > iou_threshold {
            let track_row = tracks.row(track_idx);
            if track_row.len() >= 5 {
                let track_id = track_row[4] as u32;
                match_updates.push((track_id, det_box, det_idx as u32));
            }
        } else {
            unmatched_dets.push((score, det_box, det_idx as u32));
        }
    }

    // Process unassigned detections
    for det_idx in assignment_result.unassigned_detections {
        if det_idx >= detections.nrows() {
            continue;
        }

        let det_row = detections.row(det_idx);
        if det_row.len() < 5 {
            continue;
        }

        let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
        let score = det_row[4];
        unmatched_dets.push((score, det_box, det_idx as u32));
    }

    Ok((match_updates, unmatched_dets))
}

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
        }
    }

    /// Predict new positions for all tracklets and cleanup invalid ones
    pub fn predict_and_cleanup(&mut self) -> Array2<f32> {
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

        // Flatten data for array creation
        let mut data = Vec::with_capacity(track_data.len() * 5);
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

    /// Update tracklets with detections
    pub fn update_tracklets(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        tracklet_boxes: ArrayView2<f32>,
    ) -> anyhow::Result<ScoreBoxes> {
        let (matched_boxes, unmatched_detections) =
            assign_detections_to_tracks(detection_boxes, tracklet_boxes, self.iou_threshold)?;

        for (track_id, bbox, det_idx) in matched_boxes {
            if let Some(tracklet) = self.tracklets.get_mut(&track_id) {
                tracklet.det_idx = det_idx;
                if tracklet.update(bbox).is_err() {
                    // Failed to update Kalman filter, remove tracklet
                    self.tracklets.remove(&track_id);
                }
            }
        }
        Ok(unmatched_detections)
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
