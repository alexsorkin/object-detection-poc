//! Individual bounding box tracker using Kalman filter and Multi-Object tracker implementation

use crate::bbox::{ious, Bbox};
use crate::hungarian::HungarianSolver;
use crate::kalman::{KalmanFilter, KalmanFilterParams};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use num::cast;
use rayon::prelude::*;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct KalmanBoxTrackerParams {
    pub id: u32,
    pub bbox: Bbox<f32>,
    /// Diagonal of the measurement noise covariance matrix
    /// i.e. uncertainties of (x, y, s, r) measurements
    /// default = [1., 1., 10., 10.]
    pub meas_var: Option<[f32; 4]>,
    /// Diagonal of the process noise covariance matrix
    /// i.e. uncertainties of (x, y, s, r, dx, dy, ds) during transition
    /// default = [1., 1., 1., 1., 0.01, 0.01, 0.0001]
    pub proc_var: Option<[f32; 7]>,
}

#[derive(Debug, Clone)]
pub struct KalmanBoxTracker {
    /// track id
    pub id: u32,
    pub det_idx: u32,
    /// Kalman filter tracking bbox state
    kf: KalmanFilter<f32>,
    /// number of steps tracker has been run for (each predict() is one step)
    pub age: u32,
    /// number of steps with matching detection box
    pub hits: u32,
    /// number of consecutive steps with matched box
    pub hit_streak: u32,
    /// number of consecutive steps predicted without receiving box
    pub steps_since_update: u32,
}

impl KalmanBoxTracker {
    /// Create new Kalman filter-based bbox tracker
    pub fn new(p: KalmanBoxTrackerParams, det_idx: u32) -> Self {
        let meas_var = p.meas_var.unwrap_or([1.0, 1.0, 10.0, 10.0]);
        let proc_var = p
            .proc_var
            .unwrap_or([1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001]);

        // State: [center_x, center_y, area, aspect_ratio, vel_x, vel_y, vel_area]
        let initial_state = {
            let z = p.bbox.to_z();
            DVector::from_vec(vec![z[0], z[1], z[2], z[3], 0.0, 0.0, 0.0])
        };

        let params = KalmanFilterParams {
            dim_x: 7, // center_x, center_y, area, aspect_ratio, vel_x, vel_y, vel_area
            dim_z: 4, // center_x, center_y, area, aspect_ratio
            x: initial_state,
            p: DMatrix::from_diagonal(&DVector::from_vec(vec![
                10.0, 10.0, 10.0, 10.0, 10000.0, 10000.0, 10000.0,
            ])),
            f: DMatrix::from_row_slice(
                7,
                7,
                &[
                    1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // center_x' = center_x + vel_x
                    0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, // center_y' = center_y + vel_y
                    0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, // area' = area + vel_area
                    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, // aspect_ratio' = aspect_ratio
                    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, // vel_x' = vel_x
                    0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, // vel_y' = vel_y
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, // vel_area' = vel_area
                ],
            ),
            h: DMatrix::from_row_slice(
                4,
                7,
                &[
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                ],
            ),
            r: DMatrix::from_diagonal(&DVector::from_vec(meas_var.to_vec())),
            q: DMatrix::from_diagonal(&DVector::from_vec(proc_var.to_vec())),
        };

        KalmanBoxTracker {
            id: p.id,
            det_idx,
            kf: KalmanFilter::new(params),
            age: 0,
            hits: 0,
            hit_streak: 0,
            steps_since_update: 0,
        }
    }

    /// Update tracker with detected box
    pub fn update(&mut self, bbox: Bbox<f32>) -> Result<()> {
        // Don't increase hits/hit_streak if we get
        // several updates in the same step
        if self.steps_since_update > 0 {
            self.hits += 1;
            self.hit_streak += 1;
        }
        self.steps_since_update = 0;

        let z = bbox.to_z();
        self.kf.update(DVector::from_vec(z.to_vec()))?;
        Ok(())
    }

    /// Predict box position in next step
    pub fn predict(&mut self) -> Bbox<f32> {
        // Predict area velocity close to zero to avoid negative areas
        if self.kf.x[6] + self.kf.x[2] <= 0.0 {
            self.kf.x[6] *= 0.0;
        }

        self.kf.predict();
        self.age += 1;

        if self.steps_since_update > 0 {
            self.hit_streak = 0;
        }
        self.steps_since_update += 1;

        self.bbox()
    }

    /// Get current bbox from Kalman filter state
    pub fn bbox(&self) -> Bbox<f32> {
        let state = self.kf.get_state();
        let z = [state[0], state[1], state[2], state[3]];
        Bbox::from_z(&z)
    }
}

/// Multi-object tracker using multiple KalmanBoxTrackers with parallel processing
#[derive(Debug, Clone)]
pub struct KalmanMultiTracker {
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

impl KalmanMultiTracker {
    pub fn new(
        max_age: u32,
        min_hits: u32,
        iou_threshold: f32,
        init_tracker_min_score: f32,
        measurement_noise: [f32; 4],
        process_noise: [f32; 7],
    ) -> Self {
        KalmanMultiTracker {
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            measurement_noise,
            process_noise,
            next_track_id: 1,
            tracklets: BTreeMap::new(),
            n_steps: 0,
        }
    }

    /// Parallel prediction and cleanup
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

    /// Parallel assignment of detections to tracks using Hungarian algorithm
    fn assign_detections_to_tracks(
        &self,
        detections: ArrayView2<f32>,
        tracks: ArrayView2<f32>,
    ) -> anyhow::Result<(Vec<(u32, Bbox<f32>, u32)>, Vec<(f32, Bbox<f32>, u32)>)> {
        let det_track_ious = ious(detections, tracks);
        let assignment_result =
            HungarianSolver::solve_iou(det_track_ious.view(), self.iou_threshold);

        let mut match_updates = Vec::new();
        let mut unmatched_dets = Vec::new();

        // Parallel processing of assigned detections
        let assigned_results: Vec<Option<(u32, Bbox<f32>, u32)>> = assignment_result
            .assignments
            .par_iter()
            .map(|(det_idx, track_idx)| {
                if *det_idx >= detections.nrows() || *track_idx >= tracks.nrows() {
                    return None;
                }

                let det_row = detections.row(*det_idx);
                if det_row.len() < 5 {
                    return None;
                }

                let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
                let iou = det_track_ious[(*det_idx, *track_idx)];

                if iou > self.iou_threshold {
                    let track_row = tracks.row(*track_idx);
                    if track_row.len() >= 5 {
                        let track_id = track_row[4] as u32;
                        return Some((track_id, det_box, *det_idx as u32));
                    }
                }
                None
            })
            .collect();

        // Parallel processing of unassigned detections
        let unassigned_results: Vec<Option<(f32, Bbox<f32>, u32)>> = assignment_result
            .unassigned_detections
            .par_iter()
            .map(|det_idx| {
                if *det_idx >= detections.nrows() {
                    return None;
                }

                let det_row = detections.row(*det_idx);
                if det_row.len() < 5 {
                    return None;
                }

                let det_box = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);
                let score = det_row[4];
                Some((score, det_box, *det_idx as u32))
            })
            .collect();

        // Collect results
        match_updates.extend(assigned_results.into_iter().flatten());
        unmatched_dets.extend(unassigned_results.into_iter().flatten());
        Ok((match_updates, unmatched_dets))
    }

    /// Create new tracklets from unmatched detections
    pub fn create_tracklets(&mut self, score_boxes: Vec<(f32, Bbox<f32>, u32)>) {
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

    /// Update matched tracklets
    pub fn update_tracklets(
        &mut self,
        matched_boxes: Vec<(u32, Bbox<f32>, u32)>,
    ) -> anyhow::Result<()> {
        for (track_id, bbox, det_idx) in matched_boxes {
            if let Some(tracklet) = self.tracklets.get_mut(&track_id) {
                tracklet.det_idx = det_idx;
                if tracklet.update(bbox).is_err() {
                    self.tracklets.remove(&track_id);
                }
            }
        }
        Ok(())
    }

    /// Remove stale tracklets
    pub fn remove_stale_tracklets(&mut self) {
        self.tracklets
            .retain(|_, tracklet| tracklet.steps_since_update <= self.max_age);
    }

    /// Get tracklet boxes with parallel processing
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

    /// Update tracker with new detections (unified interface)
    pub fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        let tracklet_boxes = self.predict_and_cleanup();
        let (matched_boxes, unmatched_detections) =
            self.assign_detections_to_tracks(detection_boxes, tracklet_boxes.view())?;

        self.update_tracklets(matched_boxes)?;
        self.remove_stale_tracklets();
        self.create_tracklets(unmatched_detections);

        self.n_steps += 1;
        Ok(self.get_tracklet_boxes(return_all, return_indices))
    }

    /// Clear all trackers
    pub fn clear_trackers(&mut self) {
        self.tracklets.clear();
    }

    /// Remove specific tracker
    pub fn remove_tracker(&mut self, track_id: u32) {
        self.tracklets.remove(&track_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bbox_tracker() {
        let mut tracker = KalmanBoxTracker::new(
            KalmanBoxTrackerParams {
                id: 0,
                bbox: Bbox {
                    xmin: 0.0,
                    xmax: 10.0,
                    ymin: 0.0,
                    ymax: 5.0,
                },
                meas_var: None,
                proc_var: None,
            },
            0,
        );

        let pred1 = tracker.predict();

        tracker
            .update(Bbox {
                xmin: 5.0,
                xmax: 15.0,
                ymin: 0.0,
                ymax: 4.5,
            })
            .unwrap();

        let pred2 = tracker.predict();

        // Check that prediction changed after update
        assert!(pred2.center_x() != pred1.center_x());
        assert!(tracker.hits == 1);
        assert!(tracker.age == 2);
    }

    #[test]
    fn test_bbox_conversion_consistency() {
        let bbox = Bbox::new(10.0, 20.0, 30.0, 40.0);
        let z = bbox.to_z();
        let bbox2 = Bbox::from_z(&z);

        assert_abs_diff_eq!(bbox.xmin, bbox2.xmin, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.ymin, bbox2.ymin, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.xmax, bbox2.xmax, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.ymax, bbox2.ymax, epsilon = 0.001);
    }

    #[test]
    fn test_box_multi_tracker() {
        let mut tracker = KalmanMultiTracker::new(
            5,    // max_age
            2,    // min_hits
            0.3,  // iou_threshold
            0.25, // init_tracker_min_score
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        let detections = ndarray::array![
            [10.0, 10.0, 50.0, 50.0, 0.9],
            [60.0, 60.0, 100.0, 100.0, 0.8],
        ];

        let tracks = tracker.update(detections.view(), false, false).unwrap();
        assert_eq!(tracks.ncols(), 5);

        // Test with movement
        let detections2 = ndarray::array![
            [12.0, 10.0, 52.0, 50.0, 0.9],
            [62.0, 60.0, 102.0, 100.0, 0.8],
        ];

        let tracks2 = tracker.update(detections2.view(), false, false).unwrap();
        assert_eq!(tracks2.ncols(), 5);

        // Test clear
        tracker.clear_trackers();
        assert!(tracker.tracklets.is_empty());
    }

    #[test]
    fn test_parallel_assignment() {
        let mut tracker = KalmanMultiTracker::new(
            3,
            1,
            0.3,
            0.5,
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        // Create multiple detections to test parallel processing
        let detections = ndarray::array![
            [10.0, 10.0, 30.0, 30.0, 0.9],
            [40.0, 40.0, 60.0, 60.0, 0.8],
            [70.0, 70.0, 90.0, 90.0, 0.7],
            [100.0, 100.0, 120.0, 120.0, 0.6],
        ];

        let tracks1 = tracker.update(detections.view(), true, false).unwrap(); // return_all = true
        assert_eq!(tracks1.nrows(), 4); // All should be created

        // Move detections slightly to test assignment
        let detections2 = ndarray::array![
            [12.0, 12.0, 32.0, 32.0, 0.9],
            [42.0, 42.0, 62.0, 62.0, 0.8],
            [72.0, 72.0, 92.0, 92.0, 0.7],
        ];

        let tracks2 = tracker.update(detections2.view(), true, false).unwrap(); // return_all = true
        assert!(tracks2.nrows() >= 3); // Should have at least 3 tracks (may have stale ones)
        assert!(tracks2.nrows() <= 7); // But not more than total possible

        // Check that track IDs are preserved (tracks should match)
        assert_eq!(tracks2.ncols(), 5);
    }
}
