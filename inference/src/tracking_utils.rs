/// Tracking utilities for format conversions
///
/// Provides conversion functions between TileDetection format and tracker ndarray format
use crate::frame_pipeline::TileDetection;
use ndarray::Array2;
use rayon::prelude::*;

/// Convert TileDetection (x, y, w, h) to tracker format (x1, y1, x2, y2, confidence)
///
/// This is a general-purpose utility function used by other modules.
/// For optimized performance in UnifiedTracker, see the internal implementation.
pub fn tile_detection_to_tracker_format(detections: &[TileDetection]) -> Array2<f32> {
    if detections.is_empty() {
        return Array2::zeros((0, 5));
    }

    // Parallel conversion of detections to flat vector
    let detection_data: Vec<f32> = detections
        .par_iter()
        .enumerate()
        .flat_map(|(i, det)| {
            let x1 = det.x;
            let y1 = det.y;
            let x2 = det.x + det.w;
            let y2 = det.y + det.h;
            log::debug!("Converting detection {}: TileDetection(x={:.3}, y={:.3}, w={:.3}, h={:.3}, conf={:.3}) -> tracker([{:.3}, {:.3}, {:.3}, {:.3}, {:.3}])", 
                i, det.x, det.y, det.w, det.h, det.confidence, x1, y1, x2, y2, det.confidence);
            vec![x1, y1, x2, y2, det.confidence]
        })
        .collect();

    Array2::from_shape_vec((detections.len(), 5), detection_data)
        .unwrap_or_else(|_| Array2::zeros((0, 5)))
}

/// Convert tracker output (x1, y1, x2, y2, track_id) back to TileDetection format  
///
/// Note: This function only provides tracking information. Original detection metadata
/// (class_id, class_name, confidence) should be maintained separately in a real application.
pub fn tracker_output_to_tile_detection(tracks: &ndarray::Array2<f32>) -> Vec<TileDetection> {
    // Convert AxisIter to Vec first, then parallelize
    let track_rows: Vec<_> = tracks.outer_iter().collect();

    track_rows
        .par_iter() // Parallelize the Vec iteration
        .filter_map(|track| {
            if track.len() >= 5 {
                let x1 = track[0];
                let y1 = track[1];
                let x2 = track[2];
                let y2 = track[3];
                let track_id = track[4] as u32;

                Some(TileDetection {
                    x: x1,
                    y: y1,
                    w: x2 - x1,
                    h: y2 - y1,
                    confidence: 0.8, // Track confidence (should be derived from tracker state)
                    class_id: 0, // TODO: Need to maintain class mapping from original detections
                    class_name: "tracked_object".to_string(), // TODO: Map from class_id
                    tile_idx: 0, // Default
                    vx: None,
                    vy: None,
                    track_id: Some(track_id),
                })
            } else {
                None
            }
        })
        .collect()
}

/// Helper struct for bounding box operations
#[derive(Debug, Clone, Copy)]
pub struct BoundingBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

/// IoU calculation with reduced floating point operations
#[inline]
pub fn calculate_iou(box1: &BoundingBox, box2: &BoundingBox) -> f32 {
    // Early exit for non-overlapping boxes
    if box1.x2 <= box2.x1 || box2.x2 <= box1.x1 || box1.y2 <= box2.y1 || box2.y2 <= box1.y1 {
        return 0.0;
    }

    let intersection_x1 = box1.x1.max(box2.x1);
    let intersection_y1 = box1.y1.max(box2.y1);
    let intersection_x2 = box1.x2.min(box2.x2);
    let intersection_y2 = box1.y2.min(box2.y2);

    let intersection_width = intersection_x2 - intersection_x1;
    let intersection_height = intersection_y2 - intersection_y1;

    // No need to check if intersection is positive again due to early exit above
    let intersection_area = intersection_width * intersection_height;

    let box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);

    let union_area = box1_area + box2_area - intersection_area;

    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}
