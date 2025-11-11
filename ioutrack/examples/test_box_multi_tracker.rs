use ioutrack::{KalmanMultiTracker, MultiObjectTracker};
use ndarray::array;

fn main() -> anyhow::Result<()> {
    println!("Testing BoxMultiTracker with parallelization...");

    // Create a new BoxMultiTracker with parallel processing
    let mut tracker = KalmanMultiTracker::new(
        5,                                        // max_age: tracks die after 5 frames without detection
        0,                      // min_hits: tracks are returned immediately (changed from 1 to 0)
        0.3,                    // iou_threshold: minimum IoU for track-detection association
        0.5,                    // init_tracker_min_score: minimum confidence to start new track
        [1.0, 1.0, 10.0, 10.0], // measurement noise
        [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.0001], // process noise
    );

    // Frame 1: Initial detections
    let detections1 = array![
        [10.0, 10.0, 50.0, 50.0, 0.9],     // High confidence detection
        [100.0, 100.0, 150.0, 150.0, 0.8], // Another detection
        [200.0, 200.0, 240.0, 240.0, 0.7], // Third detection
    ];

    let tracks1 = tracker.update(detections1.view(), true, false)?; // return_all=true to see new tracks
    println!("Frame 1: {} tracks created", tracks1.nrows());
    for row in tracks1.outer_iter() {
        println!(
            "  Track ID {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
            row[4], row[0], row[1], row[2], row[3]
        );
    }

    // Frame 2: Objects move slightly
    let detections2 = array![
        [12.0, 12.0, 52.0, 52.0, 0.9],     // Moved detection 1
        [102.0, 98.0, 152.0, 148.0, 0.8],  // Moved detection 2
        [205.0, 195.0, 245.0, 235.0, 0.7], // Moved detection 3
    ];

    let tracks2 = tracker.update(detections2.view(), false, false)?; // Now use normal filtering
    println!("\nFrame 2: {} tracks updated", tracks2.nrows());
    for row in tracks2.outer_iter() {
        println!(
            "  Track ID {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
            row[4], row[0], row[1], row[2], row[3]
        );
    }

    // Frame 3: One object disappears
    let detections3 = array![
        [14.0, 14.0, 54.0, 54.0, 0.9], // Still tracking first object
        [210.0, 190.0, 250.0, 230.0, 0.7], // Still tracking third object
                                       // Second object disappeared
    ];

    let tracks3 = tracker.update(detections3.view(), false, false)?;
    println!(
        "\nFrame 3: {} tracks (one object disappeared)",
        tracks3.nrows()
    );
    for row in tracks3.outer_iter() {
        println!(
            "  Track ID {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
            row[4], row[0], row[1], row[2], row[3]
        );
    }

    // Frame 4: New object appears
    let detections4 = array![
        [16.0, 16.0, 56.0, 56.0, 0.9],     // Still tracking first object
        [300.0, 300.0, 340.0, 340.0, 0.8], // New object appears
    ];

    let tracks4 = tracker.update(detections4.view(), false, false)?;
    println!(
        "\nFrame 4: {} tracks (new object appeared)",
        tracks4.nrows()
    );
    for row in tracks4.outer_iter() {
        println!(
            "  Track ID {}: [{:.1}, {:.1}, {:.1}, {:.1}]",
            row[4], row[0], row[1], row[2], row[3]
        );
    }

    println!("\nBoxMultiTracker successfully demonstrated parallel tracking!");
    println!("Active tracklets: {}", tracker.num_tracklets());
    println!("Total steps: {}", tracker.get_step_count());

    Ok(())
}
