//! Multi-object tracking implementations
//!
//! This module provides three tracking algorithms with a unified interface:
//! - SortMultiTracker: Simple Online Real-time Tracking with Kalman filters
//! - ByteMultiTracker: SortMultiTracker with high/low confidence detection handling  
//! - KalmanMultiTracker: Parallelized KalmanBoxTracker-based multi-object tracking

use ndarray::{Array2, ArrayView2};

mod bytetrack;
mod sorttrack;

pub use crate::box_tracker::KalmanMultiTracker;
pub use bytetrack::ByteMultiTracker;
pub use sorttrack::SortMultiTracker;

/// Common interface for multi-object trackers
///
/// All tracker implementations (SORT, ByteTrack, KalmanTracker) support this interface
/// for consistent usage across different tracking algorithms.
pub trait MultiObjectTracker: Send {
    /// Update tracker with new detections
    ///
    /// # Arguments
    /// * `detection_boxes` - Nx5 array where each row is [x1, y1, x2, y2, confidence]
    /// * `return_all` - If true, return all tracks; if false, filter by min_hits
    /// * `return_indices` - If true, return 6 columns including detection indices
    ///
    /// # Returns
    /// Array of tracked objects where each row is [x1, y1, x2, y2, track_id] or
    /// [x1, y1, x2, y2, track_id, det_idx] if return_indices is true
    fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>>;

    /// Clear all trackers
    fn clear_trackers(&mut self);

    /// Remove specific tracker by ID
    fn remove_tracker(&mut self, track_id: u32);

    /// Get number of active tracklets
    fn num_tracklets(&self) -> usize;

    /// Get current step count
    fn get_step_count(&self) -> u32;
}

// Implement the trait for SortMultiTracker
impl MultiObjectTracker for SortMultiTracker {
    fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        self.update(detection_boxes, return_all, return_indices)
    }

    fn clear_trackers(&mut self) {
        self.clear_trackers()
    }

    fn remove_tracker(&mut self, track_id: u32) {
        self.remove_tracker(track_id)
    }

    fn num_tracklets(&self) -> usize {
        self.tracklets.len()
    }

    fn get_step_count(&self) -> u32 {
        self.n_steps
    }
}

// Implement the trait for ByteMultiTracker
impl MultiObjectTracker for ByteMultiTracker {
    fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        self.update(detection_boxes, return_all, return_indices)
    }

    fn clear_trackers(&mut self) {
        self.clear_trackers()
    }

    fn remove_tracker(&mut self, track_id: u32) {
        self.remove_tracker(track_id)
    }

    fn num_tracklets(&self) -> usize {
        self.num_tracklets()
    }

    fn get_step_count(&self) -> u32 {
        self.get_step_count()
    }
}

// Implement the trait for KalmanMultiTracker
impl MultiObjectTracker for KalmanMultiTracker {
    fn update(
        &mut self,
        detection_boxes: ArrayView2<f32>,
        return_all: bool,
        return_indices: bool,
    ) -> anyhow::Result<Array2<f32>> {
        self.update(detection_boxes, return_all, return_indices)
    }

    fn clear_trackers(&mut self) {
        self.clear_trackers()
    }

    fn remove_tracker(&mut self, track_id: u32) {
        self.remove_tracker(track_id)
    }

    fn num_tracklets(&self) -> usize {
        self.tracklets.len()
    }

    fn get_step_count(&self) -> u32 {
        self.n_steps
    }
}
