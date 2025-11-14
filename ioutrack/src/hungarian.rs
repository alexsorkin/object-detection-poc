/// Hungarian algorithm implementation for optimal assignment
///
/// This module provides a unified Hungarian algorithm implementation
/// for optimal detection-to-track assignment in tracking algorithms.
use ndarray::ArrayView2;
use pathfinding::prelude::{kuhn_munkres, Matrix};
use rayon::prelude::*;

/// Result of Hungarian assignment algorithm
#[derive(Debug, Clone)]
pub struct AssignmentResult {
    /// Assignments as (detection_idx, track_idx) pairs
    pub assignments: Vec<(usize, usize)>,
    /// Indices of unassigned detections
    pub unassigned_detections: Vec<usize>,
    /// Indices of unassigned tracks
    pub unassigned_tracks: Vec<usize>,
    /// Total cost of the assignment
    pub total_cost: i32,
}

/// Hungarian assignment solver
pub struct HungarianSolver;

impl HungarianSolver {
    /// Solve assignment problem using Hungarian algorithm
    ///
    /// # Arguments
    /// * `cost_matrix` - Cost matrix where cost_matrix\[i\]\[j\] is the cost of assigning detection i to track j
    /// * `threshold` - Maximum allowed cost for a valid assignment (assignments above this are rejected)
    ///
    /// # Returns
    /// AssignmentResult containing optimal assignments and unassigned indices
    pub fn solve(cost_matrix: ArrayView2<f32>, threshold: f32) -> AssignmentResult {
        let num_detections = cost_matrix.nrows();
        let num_tracks = cost_matrix.ncols();

        if num_detections == 0 || num_tracks == 0 {
            return AssignmentResult {
                assignments: Vec::new(),
                unassigned_detections: (0..num_detections).collect(),
                unassigned_tracks: (0..num_tracks).collect(),
                total_cost: 0,
            };
        }

        // Check sparsity and use optimized algorithm for sparse matrices
        let valid_count = cost_matrix.iter().filter(|&&x| x < threshold).count();
        let total_entries = num_detections * num_tracks;

        // Use greedy assignment for very sparse matrices (< 25% valid entries)
        if valid_count < total_entries / 4 {
            return Self::solve_greedy(cost_matrix, threshold);
        }

        Self::solve_hungarian(cost_matrix, threshold)
    }

    /// Full Hungarian algorithm implementation
    fn solve_hungarian(cost_matrix: ArrayView2<f32>, threshold: f32) -> AssignmentResult {
        let num_detections = cost_matrix.nrows();
        let num_tracks = cost_matrix.ncols();

        // Convert to integer cost matrix for pathfinding crate with parallel processing
        let max_cost = 1_000_000i32; // Large value for invalid assignments
        let threshold_int = (threshold * 1000.0) as i32;

        // Ensure matrix is square by padding with dummy entries
        let size = num_detections.max(num_tracks);
        let mut int_cost_matrix = Matrix::new(size, size, max_cost);

        // Parallel filling of the actual costs
        let cost_data: Vec<(usize, usize, i32)> = (0..num_detections)
            .into_par_iter()
            .flat_map(|i| {
                (0..num_tracks).into_par_iter().map(move |j| {
                    let cost = (cost_matrix[[i, j]] * 1000.0) as i32;
                    (i, j, cost)
                })
            })
            .collect();

        // Apply costs to matrix (sequential, as Matrix is not thread-safe for writes)
        for (i, j, cost) in cost_data {
            int_cost_matrix[(i, j)] = cost;
        }

        // Solve using Hungarian algorithm
        let (total_cost, raw_assignments) = kuhn_munkres(&int_cost_matrix);

        // Process assignment results and filter by threshold
        let assignments: Vec<(usize, usize)> = raw_assignments
            .par_iter()
            .enumerate()
            .filter_map(|(det_idx, &track_idx)| {
                if det_idx < num_detections
                    && track_idx < num_tracks
                    && int_cost_matrix[(det_idx, track_idx)] <= threshold_int
                {
                    Some((det_idx, track_idx))
                } else {
                    None
                }
            })
            .collect();

        // Parallel determination of assigned indices
        let assigned_detections: Vec<bool> = (0..num_detections)
            .into_par_iter()
            .map(|det_idx| assignments.iter().any(|(d, _)| *d == det_idx))
            .collect();

        let assigned_tracks: Vec<bool> = (0..num_tracks)
            .into_par_iter()
            .map(|track_idx| assignments.iter().any(|(_, t)| *t == track_idx))
            .collect();

        // Collect unassigned indices
        let unassigned_detections: Vec<usize> = (0..num_detections)
            .into_par_iter()
            .filter(|&i| !assigned_detections[i])
            .collect();

        let unassigned_tracks: Vec<usize> = (0..num_tracks)
            .into_par_iter()
            .filter(|&i| !assigned_tracks[i])
            .collect();

        AssignmentResult {
            assignments,
            unassigned_detections,
            unassigned_tracks,
            total_cost,
        }
    }

    /// Greedy assignment algorithm for sparse cost matrices
    /// Much faster than Hungarian when most assignments are invalid
    fn solve_greedy(cost_matrix: ArrayView2<f32>, threshold: f32) -> AssignmentResult {
        let num_detections = cost_matrix.nrows();
        let num_tracks = cost_matrix.ncols();

        // Collect all valid assignments with their costs
        let mut candidates: Vec<(f32, usize, usize)> = Vec::new();

        for i in 0..num_detections {
            for j in 0..num_tracks {
                let cost = cost_matrix[[i, j]];
                if cost < threshold {
                    candidates.push((cost, i, j));
                }
            }
        }

        // Sort by cost (ascending - best assignments first)
        candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Greedily assign
        let mut assignments = Vec::new();
        let mut used_detections = vec![false; num_detections];
        let mut used_tracks = vec![false; num_tracks];

        for (_cost, det_idx, track_idx) in candidates {
            if !used_detections[det_idx] && !used_tracks[track_idx] {
                assignments.push((det_idx, track_idx));
                used_detections[det_idx] = true;
                used_tracks[track_idx] = true;
            }
        }

        // Collect unassigned indices
        let unassigned_detections: Vec<usize> = (0..num_detections)
            .filter(|&i| !used_detections[i])
            .collect();

        let unassigned_tracks: Vec<usize> = (0..num_tracks).filter(|&i| !used_tracks[i]).collect();

        // Calculate total cost
        let total_cost = assignments
            .iter()
            .map(|(det_idx, track_idx)| (cost_matrix[[*det_idx, *track_idx]] * 1000.0) as i32)
            .sum();

        AssignmentResult {
            assignments,
            unassigned_detections,
            unassigned_tracks,
            total_cost,
        }
    }

    /// Solve assignment problem with IoU cost matrix
    ///
    /// # Arguments
    /// * `iou_matrix` - IoU matrix where iou_matrix\[i\]\[j\] is the IoU between detection i and track j
    /// * `iou_threshold` - Minimum IoU for a valid assignment
    ///
    /// # Returns
    /// AssignmentResult containing optimal assignments based on IoU
    pub fn solve_iou(iou_matrix: ArrayView2<f32>, iou_threshold: f32) -> AssignmentResult {
        // Convert IoU to cost (higher IoU = lower cost)
        let cost_matrix = iou_matrix.mapv(|iou| 1.0 - iou);
        let cost_threshold = 1.0 - iou_threshold;

        Self::solve(cost_matrix.view(), cost_threshold)
    }
}

/// Convert assignment result to legacy format for backward compatibility
impl AssignmentResult {
    /// Convert to (assignment_vector, unassigned_detections) format
    /// Used by SORT algorithm
    pub fn to_sort_format(&self, num_detections: usize) -> (Vec<Option<usize>>, Vec<usize>) {
        let mut assignment_vector = vec![None; num_detections];

        for &(det_idx, track_idx) in &self.assignments {
            assignment_vector[det_idx] = Some(track_idx);
        }

        (assignment_vector, self.unassigned_detections.clone())
    }
}
