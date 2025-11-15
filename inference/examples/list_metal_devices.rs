/// Utility to list available Metal/GPU devices on macOS
///
/// This shows what Metal-compatible GPUs are available in your system.
///
/// Run with: cargo run --example list_metal_devices
use std::process::Command;

fn main() {
    println!("🔍 Enumerating Metal-compatible GPU devices...\n");

    // Use Swift to query Metal devices directly
    let swift_code = r#"
import Metal

let devices = MTLCopyAllDevices()
print("Metal Devices: \(devices.count)")
for (index, device) in devices.enumerated() {
    print("---")
    print("Index: \(index)")
    print("Name: \(device.name)")
    print("LowPower: \(device.isLowPower)")
    print("Headless: \(device.isHeadless)")
    print("Removable: \(device.isRemovable)")
    print("MaxWorkingSetSize: \(device.recommendedMaxWorkingSetSize / 1024 / 1024)")
}
"#;

    // Write and execute Swift code
    let output = Command::new("swift")
        .arg("-")
        .arg("-suppress-warnings")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(swift_code.as_bytes())?;
            }
            child.wait_with_output()
        });

    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);

            let mut device_count = 0;
            let mut current_device = std::collections::HashMap::new();

            for line in stdout.lines() {
                let trimmed = line.trim();

                if trimmed.starts_with("Metal Devices:") {
                    device_count = trimmed
                        .split(':')
                        .nth(1)
                        .and_then(|s| s.trim().parse::<usize>().ok())
                        .unwrap_or(0);
                } else if trimmed == "---" {
                    if !current_device.is_empty() {
                        print_device(&current_device);
                        current_device.clear();
                    }
                } else if let Some((key, value)) = trimmed.split_once(':') {
                    current_device.insert(key.trim().to_string(), value.trim().to_string());
                }
            }

            // Print last device
            if !current_device.is_empty() {
                print_device(&current_device);
            }

            println!("\n📊 Total Metal devices: {}\n", device_count);

            if device_count > 0 {
                println!("ℹ️  CoreML/Metal Selection:");
                println!("   - CoreML automatically selects the best Metal device");
                println!("   - Prefers high-performance (non-low-power) GPUs for inference");
                println!("   - No manual device selection needed");
            } else {
                println!("❌ No Metal devices found");
            }
        }
        Err(e) => {
            println!("❌ Failed to query Metal devices: {}", e);
            println!("\nTrying fallback method...\n");

            // Fallback to system_profiler
            let sp_output = Command::new("system_profiler")
                .arg("SPDisplaysDataType")
                .output();

            if let Ok(result) = sp_output {
                let stdout = String::from_utf8_lossy(&result.stdout);
                println!("{}", stdout);
            }
        }
    }

    println!("\n💡 To use GPU acceleration:");
    println!("   cargo run --features coreml --example detect_video");
}

fn print_device(device: &std::collections::HashMap<String, String>) {
    println!(
        "\n🎮 Device {}:",
        device.get("Index").unwrap_or(&"?".to_string())
    );
    println!(
        "   Name: {}",
        device.get("Name").unwrap_or(&"Unknown".to_string())
    );

    let is_low_power = device.get("LowPower").map(|s| s == "true").unwrap_or(false);
    let is_headless = device.get("Headless").map(|s| s == "true").unwrap_or(false);

    if is_low_power {
        println!("   Type: 💡 Integrated GPU (power-efficient)");
    } else {
        println!("   Type: ⚡ High-Performance GPU (dedicated)");
    }

    if is_headless {
        println!("   Mode: Headless (compute-only, no display)");
    }

    if let Some(mem) = device.get("MaxWorkingSetSize") {
        println!("   Max Working Set: {} MB", mem);
    }
}
