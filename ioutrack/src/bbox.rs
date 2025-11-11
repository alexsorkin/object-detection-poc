//! Bounding box operations and IoU calculations

use ndarray::prelude::*;
use rayon::prelude::*;
use std::fmt;

/// Simple bounding box representation
#[derive(Debug, Clone, PartialEq)]
pub struct Bbox<T = f32> {
    pub xmin: T,
    pub ymin: T,
    pub xmax: T,
    pub ymax: T,
}

impl Bbox<f32> {
    pub fn new(xmin: f32, ymin: f32, xmax: f32, ymax: f32) -> Self {
        Self {
            xmin,
            ymin,
            xmax,
            ymax,
        }
    }

    pub fn width(&self) -> f32 {
        self.xmax - self.xmin
    }

    pub fn height(&self) -> f32 {
        self.ymax - self.ymin
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center_x(&self) -> f32 {
        (self.xmin + self.xmax) / 2.0
    }

    pub fn center_y(&self) -> f32 {
        (self.ymin + self.ymax) / 2.0
    }
}

impl Bbox<f32> {
    /// Convert to bounds array [xmin, ymin, xmax, ymax]
    pub fn to_bounds(&self) -> [f32; 4] {
        [self.xmin, self.ymin, self.xmax, self.ymax]
    }

    /// Convert to center format [center_x, center_y, area, aspect_ratio]
    /// Used for Kalman filter state representation
    pub fn to_z(&self) -> [f32; 4] {
        let w = self.width();
        let h = self.height();
        let area = w * h;
        let aspect_ratio = if h != 0.0 { w / h } else { 1.0 };
        [self.center_x(), self.center_y(), area, aspect_ratio]
    }

    /// Create from center format [center_x, center_y, area, aspect_ratio]
    pub fn from_z(z: &[f32; 4]) -> Self {
        let center_x = z[0];
        let center_y = z[1];
        let area = z[2];
        let aspect_ratio = z[3];

        let h = (area / aspect_ratio).sqrt();
        let w = aspect_ratio * h;

        Self {
            xmin: center_x - w / 2.0,
            ymin: center_y - h / 2.0,
            xmax: center_x + w / 2.0,
            ymax: center_y + h / 2.0,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Bbox<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bbox({}, {}, {}, {})",
            self.xmin, self.ymin, self.xmax, self.ymax
        )
    }
}

/// Calculate IoU between two bounding boxes
pub fn calculate_iou(bbox1: &Bbox<f32>, bbox2: &Bbox<f32>) -> f32 {
    let x1 = bbox1.xmin.max(bbox2.xmin);
    let y1 = bbox1.ymin.max(bbox2.ymin);
    let x2 = bbox1.xmax.min(bbox2.xmax);
    let y2 = bbox1.ymax.min(bbox2.ymax);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let union = bbox1.area() + bbox2.area() - intersection;

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

/// Compute IoU matrix between detections and tracks with parallel processing
/// Returns: (n_detections, n_tracks) IoU matrix
pub fn ious(detections: ArrayView2<f32>, tracks: ArrayView2<f32>) -> Array2<f32> {
    let n_dets = detections.nrows();
    let n_tracks = tracks.nrows();

    if n_dets == 0 || n_tracks == 0 {
        return Array2::zeros((n_dets, n_tracks));
    }

    // Parallel computation of IoU matrix
    let iou_data: Vec<f32> = (0..n_dets)
        .into_par_iter()
        .flat_map(|i| {
            let det_row = detections.row(i);
            if det_row.len() < 4 {
                return vec![0.0; n_tracks];
            }
            let det_bbox = Bbox::new(det_row[0], det_row[1], det_row[2], det_row[3]);

            (0..n_tracks)
                .map(|j| {
                    let track_row = tracks.row(j);
                    if track_row.len() < 4 {
                        return 0.0;
                    }
                    let track_bbox =
                        Bbox::new(track_row[0], track_row[1], track_row[2], track_row[3]);
                    calculate_iou(&det_bbox, &track_bbox)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    Array2::from_shape_vec((n_dets, n_tracks), iou_data).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bbox_creation() {
        let bbox = Bbox::new(0.0, 0.0, 10.0, 10.0);
        assert_eq!(bbox.xmin, 0.0);
        assert_eq!(bbox.ymin, 0.0);
        assert_eq!(bbox.xmax, 10.0);
        assert_eq!(bbox.ymax, 10.0);
    }

    #[test]
    fn test_bbox_properties() {
        let bbox = Bbox::new(0.0, 0.0, 10.0, 5.0);
        assert_eq!(bbox.width(), 10.0);
        assert_eq!(bbox.height(), 5.0);
        assert_eq!(bbox.area(), 50.0);
        assert_eq!(bbox.center_x(), 5.0);
        assert_eq!(bbox.center_y(), 2.5);
    }

    #[test]
    fn test_iou_calculation() {
        let bbox1 = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let bbox2 = Bbox::new(5.0, 5.0, 15.0, 15.0);
        let iou = calculate_iou(&bbox1, &bbox2);
        assert_abs_diff_eq!(iou, 25.0 / 175.0, epsilon = 0.001);
    }

    #[test]
    fn test_bbox_conversion() {
        let bbox = Bbox::new(0.0, 0.0, 10.0, 10.0);
        let z = bbox.to_z();
        let bbox2 = Bbox::from_z(&z);

        assert_abs_diff_eq!(bbox.xmin, bbox2.xmin, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.ymin, bbox2.ymin, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.xmax, bbox2.xmax, epsilon = 0.001);
        assert_abs_diff_eq!(bbox.ymax, bbox2.ymax, epsilon = 0.001);
    }
}
