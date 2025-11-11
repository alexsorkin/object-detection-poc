//! Pure Rust IOU-based tracking library
//!
//! This crate provides implementations of SortMultiTracker, ByteMultiTracker, and KalmanMultiTracker algorithms
//! without Python bindings, suitable for pure Rust applications.
//!
//! # Unified Interface
//!
//! All tracking algorithms implement the `MultiObjectTracker` trait for consistent usage:
//!
//! ```rust,ignore
//! use ioutrack::{SortMultiTracker, ByteMultiTracker, KalmanMultiTracker, MultiObjectTracker};
//! use ndarray::array;
//!
//! // Create any tracker
//! let mut tracker: Box<dyn MultiObjectTracker> = Box::new(SortMultiTracker::new(/*...*/));
//!
//! // Use unified interface
//! let detections = array![[10.0, 10.0, 50.0, 50.0, 0.9]];
//! let tracks = tracker.update(detections.view(), false, false)?;
//! ```

pub mod bbox;
pub mod box_tracker;
pub mod hungarian; // Hungarian algorithm for optimal assignment
pub mod kalman;
pub mod trackers;

pub use box_tracker::KalmanBoxTracker;
pub use hungarian::{AssignmentResult, HungarianSolver};
pub use trackers::{ByteMultiTracker, KalmanMultiTracker, MultiObjectTracker, SortMultiTracker};
