//! ByteMultiTracker implementation using SortTracker as base tracker
//! ByteMultiTracker extends SORT with a two-pass association strategy

use crate::trackers::SortMultiTracker;
use ndarray::prelude::*;
use rayon::prelude::*;

#[derive(Debug, Clone)]
pub struct ByteMultiTracker {
    pub high_score_threshold: f32,
    pub low_score_threshold: f32,
    sort_tracker: SortMultiTracker,
}

impl ByteMultiTracker {
    pub fn new(
        max_age: u32,
        min_hits: u32,
        iou_threshold: f32,
        init_tracker_min_score: f32,
        high_score_threshold: f32,
        low_score_threshold: f32,
        measurement_noise: [f32; 4],
        process_noise: [f32; 7],
    ) -> Self {
        let sort_tracker = SortMultiTracker::new(
            max_age,
            min_hits,
            iou_threshold,
            init_tracker_min_score,
            measurement_noise,
            process_noise,
        );

        ByteMultiTracker {
            high_score_threshold,
            low_score_threshold,
            sort_tracker,
        }
    }

    fn split_detections(&self, detection_boxes: ArrayView2<f32>) -> (Array2<f32>, Array2<f32>) {
        // Parallel filtering of detections by score threshold
        let detection_data: Vec<(Vec<f32>, bool)> = detection_boxes
            .outer_iter()
            .collect::<Vec<_>>()
            .par_iter()
            .filter_map(|det_row| {
                if det_row.len() < 5 {
                    return None;
                }

                let score = det_row[4];
                let det_data = det_row.to_vec();

                if score >= self.high_score_threshold {
                    Some((det_data, true)) // High score
                } else if score >= self.low_score_threshold {
                    Some((det_data, false)) // Low score
                } else {
                    None // Below threshold
                }
            })
            .collect();

        // Separate high and low score detections
        let mut high_score_dets = Vec::new();
        let mut low_score_dets = Vec::new();

        for (det_data, is_high_score) in detection_data {
            if is_high_score {
                high_score_dets.extend(det_data);
            } else {
                low_score_dets.extend(det_data);
            }
        }

        let high_count = high_score_dets.len() / 5;
        let low_count = low_score_dets.len() / 5;

        let high_array = if high_count > 0 {
            Array2::from_shape_vec((high_count, 5), high_score_dets).unwrap()
        } else {
            Array2::zeros((0, 5))
        };

        let low_array = if low_count > 0 {
            Array2::from_shape_vec((low_count, 5), low_score_dets).unwrap()
        } else {
            Array2::zeros((0, 5))
        };

        (high_array, low_array)
    }

    pub fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        let (high_score_detections, low_score_detections) = self.split_detections(detection_boxes);

        // ByteMultiTracker two-pass algorithm:

        // Pass 1: Predict positions for all tracklets
        let tracklet_boxes = self.sort_tracker.predict_and_cleanup();

        // Pass 2: Associate high confidence detections with all tracklets
        let unmatched_high_dets = self
            .sort_tracker
            .update_tracklets(high_score_detections.view(), tracklet_boxes.view())?;

        // Pass 3: Get unmatched tracklets (those that didn't get associated)
        let unmatched_track_boxes: Array2<f32> = {
            let unmatched_data: Vec<f32> = tracklet_boxes
                .outer_iter()
                .zip(self.sort_tracker.tracklets.iter())
                .filter_map(|(track_box, (_, tracklet))| {
                    // If tracklet wasn't updated in this step, it's unmatched
                    if tracklet.steps_since_update == 0 {
                        None // Was matched
                    } else {
                        Some(track_box.to_vec())
                    }
                })
                .flatten()
                .collect();

            if unmatched_data.is_empty() {
                Array2::zeros((0, 5))
            } else {
                Array2::from_shape_vec((unmatched_data.len() / 5, 5), unmatched_data)?
            }
        };

        // Pass 4: Associate low confidence detections with unmatched tracklets
        let _unmatched_low_dets = self
            .sort_tracker
            .update_tracklets(low_score_detections.view(), unmatched_track_boxes.view())?;

        // Pass 5: Remove stale tracklets
        self.sort_tracker.remove_stale_tracklets();

        // Pass 6: Create new tracklets only from unmatched high confidence detections
        self.sort_tracker.create_tracklets(unmatched_high_dets);

        // Pass 7: Increment step counter
        self.sort_tracker.n_steps += 1;

        // Return current tracklet boxes
        Ok(self
            .sort_tracker
            .get_tracklet_boxes(return_all, return_indices))
    }

    pub fn clear_trackers(&mut self) {
        self.sort_tracker.clear_trackers();
    }

    pub fn remove_tracker(&mut self, track_id: u32) {
        self.sort_tracker.remove_tracker(track_id);
    }

    pub fn get_current_track_boxes(&self, return_all: bool, return_indices: bool) -> Array2<f32> {
        self.sort_tracker
            .get_tracklet_boxes(return_all, return_indices)
    }

    pub fn num_tracklets(&self) -> usize {
        self.sort_tracker.tracklets.len()
    }

    pub fn get_step_count(&self) -> u32 {
        self.sort_tracker.n_steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_bytetrack() {
        let mut tracker = ByteMultiTracker::new(
            5,    // max_age
            2,    // min_hits
            0.3,  // iou_threshold
            0.25, // init_tracker_min_score
            0.7,  // high_score_threshold
            0.1,  // low_score_threshold
            [1.0, 1.0, 10.0, 10.0],
            [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001],
        );

        let detections = array![
            [270.71, 1.6277, 374.85, 276.68, 0.85113],
            [376.79, 13.419, 464.71, 250.11, 0.79943],
            [198.16, 102.59, 243.19, 214.45, 0.71286],
        ];

        let tracks = tracker.update(detections.view(), false, false).unwrap();
        assert_eq!(tracks.ncols(), 5);

        // Test clear
        tracker.clear_trackers();
        let empty_tracks = tracker.get_current_track_boxes(false, false);
        assert_eq!(empty_tracks.nrows(), 0);
    }
}
