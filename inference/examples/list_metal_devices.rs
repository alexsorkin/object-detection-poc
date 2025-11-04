/// Utility to list available Metal/GPU devices on macOS
/// 
/// This shows what GPUs are available in your system.
/// CoreML (used by ONNX Runtime) automatically selects the best GPU.
///
/// Run with: cargo run --release --example list_metal_devices

use std::process::Command;

fn main() {
    println!("🔍 Enumerating GPU devices on this Mac...\n");

    // Query system GPUs using system_profiler
    let output = Command::new("system_profiler")
        .arg("SPDisplaysDataType")
        .output();

    match output {
        Ok(result) => {
            let stdout = String::from_utf8_lossy(&result.stdout);
            
            // Parse GPU information
            let mut current_gpu = None;
            let mut gpu_count = 0;
            
            for line in stdout.lines() {
                let trimmed = line.trim();
                
                if trimmed.starts_with("Chipset Model:") {
                    gpu_count += 1;
                    current_gpu = Some(trimmed.replace("Chipset Model:", "").trim().to_string());
                    println!("GPU {}: {}", gpu_count, current_gpu.as_ref().unwrap());
                } else if trimmed.starts_with("VRAM") && current_gpu.is_some() {
                    println!("  VRAM: {}", trimmed.replace("VRAM (Total):", "")
                        .replace("VRAM (Dynamic, Max):", "").trim());
                } else if trimmed.starts_with("Vendor:") && current_gpu.is_some() {
                    println!("  Vendor: {}", trimmed.replace("Vendor:", "").trim());
                } else if trimmed.starts_with("Bus:") && current_gpu.is_some() {
                    let bus = trimmed.replace("Bus:", "").trim().to_string();
                    println!("  Bus: {}", bus);
                    
                    // Indicate which is high-performance
                    if bus.contains("PCIe") {
                        println!("  ⚡ High-Performance GPU (dedicated)");
                    } else if bus.contains("Built-In") {
                        println!("  💡 Integrated GPU (power-efficient)");
                    }
                    println!();
                    current_gpu = None;
                }
            }

            if gpu_count == 0 {
                println!("❌ No GPUs found. This is unexpected on macOS.");
            } else {
                println!("📊 Total GPUs found: {}\n", gpu_count);
                
                println!("ℹ️  CoreML/Metal GPU Selection:");
                println!("   - ONNX Runtime with CoreML backend automatically selects the best GPU");
                println!("   - For ML workloads, it typically prefers the dedicated/high-performance GPU");
                println!("   - On this system: likely using AMD Radeon Pro for inference");
                println!("   - No manual device selection needed (CoreML handles this)");
            }
        }
        Err(e) => {
            println!("❌ Failed to query system GPUs: {}", e);
            println!("\nℹ️  CoreML will still work and automatically select the best GPU.");
        }
    }

    println!("\n💡 To use GPU acceleration:");
    println!("   cargo run --release --features metal --example detect");
}

