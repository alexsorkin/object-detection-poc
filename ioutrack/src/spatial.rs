//! Spatial indexing structures for efficient nearest neighbor search in dense tracking scenarios

use std::collections::HashMap;

/// Spatial grid for efficient bounding box queries in dense scenes
/// Particularly useful when tracking many objects where most don't overlap
#[derive(Debug, Clone)]
pub struct SpatialGrid {
    grid_size: f32,
    grid: HashMap<(i32, i32), Vec<usize>>,
}

impl SpatialGrid {
    /// Create a new spatial grid with the specified cell size
    ///
    /// # Arguments
    /// * `grid_size` - Size of each grid cell. Should be roughly the size of typical objects
    pub fn new(grid_size: f32) -> Self {
        Self {
            grid_size,
            grid: HashMap::new(),
        }
    }

    /// Insert a bounding box into the grid
    ///
    /// # Arguments
    /// * `bbox` - Bounding box as [xmin, ymin, xmax, ymax]
    /// * `id` - Identifier for this bounding box
    pub fn insert(&mut self, bbox: &[f32; 4], id: usize) {
        let x_min = (bbox[0] / self.grid_size).floor() as i32;
        let y_min = (bbox[1] / self.grid_size).floor() as i32;
        let x_max = (bbox[2] / self.grid_size).ceil() as i32;
        let y_max = (bbox[3] / self.grid_size).ceil() as i32;

        for x in x_min..=x_max {
            for y in y_min..=y_max {
                self.grid.entry((x, y)).or_default().push(id);
            }
        }
    }

    /// Query the grid for potential overlapping bounding boxes
    /// Returns a list of IDs that might overlap with the query bbox
    ///
    /// # Arguments
    /// * `bbox` - Query bounding box as [xmin, ymin, xmax, ymax]
    pub fn query(&self, bbox: &[f32; 4]) -> Vec<usize> {
        let x_min = (bbox[0] / self.grid_size).floor() as i32;
        let y_min = (bbox[1] / self.grid_size).floor() as i32;
        let x_max = (bbox[2] / self.grid_size).ceil() as i32;
        let y_max = (bbox[3] / self.grid_size).ceil() as i32;

        let mut candidates = Vec::new();
        for x in x_min..=x_max {
            for y in y_min..=y_max {
                if let Some(ids) = self.grid.get(&(x, y)) {
                    candidates.extend(ids);
                }
            }
        }

        // Remove duplicates while preserving order
        candidates.sort_unstable();
        candidates.dedup();
        candidates
    }

    /// Clear all entries from the grid (reuse for next frame)
    pub fn clear(&mut self) {
        for bucket in self.grid.values_mut() {
            bucket.clear();
        }
    }

    /// Get statistics about grid usage
    pub fn stats(&self) -> SpatialGridStats {
        let total_cells = self.grid.len();
        let total_entries: usize = self.grid.values().map(|v| v.len()).sum();
        let max_entries = self.grid.values().map(|v| v.len()).max().unwrap_or(0);
        let avg_entries = if total_cells > 0 {
            total_entries as f32 / total_cells as f32
        } else {
            0.0
        };

        SpatialGridStats {
            total_cells,
            total_entries,
            max_entries_per_cell: max_entries,
            avg_entries_per_cell: avg_entries,
        }
    }
}

/// Statistics about spatial grid usage
#[derive(Debug, Clone)]
pub struct SpatialGridStats {
    pub total_cells: usize,
    pub total_entries: usize,
    pub max_entries_per_cell: usize,
    pub avg_entries_per_cell: f32,
}

/// Spatial index for efficient IOU computation in dense scenes
/// Reduces the number of IOU calculations needed
pub struct SpatialTracker {
    detection_grid: SpatialGrid,
    track_grid: SpatialGrid,
    detection_bboxes: Vec<[f32; 4]>,
    track_bboxes: Vec<[f32; 4]>,
}

impl SpatialTracker {
    /// Create a new spatial tracker with the specified grid size
    pub fn new(grid_size: f32) -> Self {
        Self {
            detection_grid: SpatialGrid::new(grid_size),
            track_grid: SpatialGrid::new(grid_size),
            detection_bboxes: Vec::new(),
            track_bboxes: Vec::new(),
        }
    }

    /// Update the spatial index with new detections and tracks
    ///
    /// # Arguments
    /// * `detections` - Array of detection bounding boxes [N, 4]
    /// * `tracks` - Array of track bounding boxes [M, 4]
    pub fn update(&mut self, detections: &[[f32; 4]], tracks: &[[f32; 4]]) {
        // Clear previous frame data
        self.detection_grid.clear();
        self.track_grid.clear();
        self.detection_bboxes.clear();
        self.track_bboxes.clear();

        // Store bounding boxes
        self.detection_bboxes.extend_from_slice(detections);
        self.track_bboxes.extend_from_slice(tracks);

        // Build spatial index for detections
        for (i, bbox) in detections.iter().enumerate() {
            self.detection_grid.insert(bbox, i);
        }

        // Build spatial index for tracks
        for (i, bbox) in tracks.iter().enumerate() {
            self.track_grid.insert(bbox, i);
        }
    }

    /// Get potential detection-track pairs that might have non-zero IOU
    /// This significantly reduces the number of IOU calculations needed
    ///
    /// Returns: Vec of (detection_idx, track_idx) pairs to compute IOU for
    pub fn get_candidate_pairs(&self) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        for (det_idx, det_bbox) in self.detection_bboxes.iter().enumerate() {
            let track_candidates = self.track_grid.query(det_bbox);
            for &track_idx in &track_candidates {
                pairs.push((det_idx, track_idx));
            }
        }

        pairs
    }

    /// Get statistics about the spatial indexing efficiency
    pub fn efficiency_stats(&self) -> SpatialEfficiencyStats {
        let total_possible_pairs = self.detection_bboxes.len() * self.track_bboxes.len();
        let candidate_pairs = self.get_candidate_pairs().len();
        let reduction_ratio = if total_possible_pairs > 0 {
            1.0 - (candidate_pairs as f32 / total_possible_pairs as f32)
        } else {
            0.0
        };

        SpatialEfficiencyStats {
            total_possible_pairs,
            candidate_pairs,
            reduction_ratio,
            detection_grid_stats: self.detection_grid.stats(),
            track_grid_stats: self.track_grid.stats(),
        }
    }
}

/// Statistics about spatial indexing efficiency
#[derive(Debug, Clone)]
pub struct SpatialEfficiencyStats {
    pub total_possible_pairs: usize,
    pub candidate_pairs: usize,
    pub reduction_ratio: f32,
    pub detection_grid_stats: SpatialGridStats,
    pub track_grid_stats: SpatialGridStats,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spatial_grid_basic() {
        let mut grid = SpatialGrid::new(100.0);

        // Insert a few bounding boxes
        grid.insert(&[0.0, 0.0, 50.0, 50.0], 0);
        grid.insert(&[150.0, 150.0, 200.0, 200.0], 1);

        // Query for overlapping boxes
        let results = grid.query(&[25.0, 25.0, 75.0, 75.0]);
        assert!(results.contains(&0));
        assert!(!results.contains(&1));
    }

    #[test]
    fn test_spatial_tracker() {
        let mut tracker = SpatialTracker::new(100.0);

        let detections = vec![[0.0, 0.0, 50.0, 50.0], [200.0, 200.0, 250.0, 250.0]];

        let tracks = vec![
            [25.0, 25.0, 75.0, 75.0],     // Should overlap with detection 0
            [300.0, 300.0, 350.0, 350.0], // Shouldn't overlap with any detection
        ];

        tracker.update(&detections, &tracks);
        let pairs = tracker.get_candidate_pairs();

        // Should find the overlapping pair but not the non-overlapping one
        assert!(pairs.contains(&(0, 0)));
        assert!(!pairs.contains(&(1, 1)));

        let stats = tracker.efficiency_stats();
        assert!(stats.reduction_ratio > 0.0);
    }
}
