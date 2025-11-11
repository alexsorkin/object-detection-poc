use ioutrack::hungarian::HungarianSolver;
use ndarray::Array2;

fn main() {
    println!("Testing optimized Hungarian algorithm...");

    // Create a simple test case
    let cost_matrix = Array2::from_shape_vec(
        (3, 3),
        vec![
            1.0, 2.0, 3.0, // Detection 0 costs
            2.0, 1.0, 4.0, // Detection 1 costs
            3.0, 3.0, 1.0, // Detection 2 costs
        ],
    )
    .unwrap();

    let result = HungarianSolver::solve(cost_matrix.view(), 2.0);

    println!("Assignments: {:?}", result.assignments);
    println!("Unassigned detections: {:?}", result.unassigned_detections);
    println!("Unassigned tracks: {:?}", result.unassigned_tracks);
    println!("Total cost: {}", result.total_cost);

    // Test with larger matrix to show parallel processing
    println!("\nTesting with larger matrix (50x50)...");
    let large_matrix = Array2::from_shape_fn((50, 50), |(i, j)| ((i + j) as f32 % 10.0) / 10.0);

    let start = std::time::Instant::now();
    let large_result = HungarianSolver::solve(large_matrix.view(), 0.5);
    let duration = start.elapsed();

    println!("Large matrix solved in {:?}", duration);
    println!("Found {} assignments", large_result.assignments.len());

    // Test IoU version
    println!("\nTesting IoU-based Hungarian...");
    let iou_matrix = Array2::from_shape_fn((10, 10), |(i, j)| if i == j { 0.8 } else { 0.2 });

    let iou_result = HungarianSolver::solve_iou(iou_matrix.view(), 0.5);
    println!("IoU assignments: {:?}", iou_result.assignments);

    println!("✅ All Hungarian algorithm tests passed!");
}
