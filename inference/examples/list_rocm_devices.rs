/// Utility to list available ROCm/HIP devices
///
/// This shows what ROCm-compatible AMD GPUs are available in your system.
///
/// Run with: cargo run --features rocm --example list_rocm_devices

use std::process::Command;

fn main() {
    println!("🔍 Enumerating ROCm/HIP-compatible AMD GPU devices...\n");

    // Check ROCm installation first
    check_rocm_installation();
    
    // List devices using rocminfo
    list_devices_rocminfo();
    
    // List devices using hipinfo (if available)
    list_devices_hipinfo();
    
    // Check GPU visibility
    check_gpu_visibility();
}

fn check_rocm_installation() {
    println!("📋 Checking ROCm installation...");
    
    // Check rocm-smi
    match Command::new("rocm-smi").arg("--version").output() {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("✓ rocm-smi found: {}", version.trim());
        }
        Err(_) => {
            println!("⚠️  rocm-smi not found - ROCm may not be installed");
        }
    }
    
    // Check ROCm version
    match Command::new("cat").arg("/opt/rocm/.info/version").output() {
        Ok(output) => {
            let version = String::from_utf8_lossy(&output.stdout);
            println!("✓ ROCm version: {}", version.trim());
        }
        Err(_) => {
            println!("⚠️  ROCm version file not found");
        }
    }
    
    println!();
}

fn list_devices_rocminfo() {
    println!("🎯 Using rocminfo to enumerate devices...");
    
    match Command::new("rocminfo").output() {
        Ok(output) => {
            let info = String::from_utf8_lossy(&output.stdout);
            
            // Parse rocminfo output for GPU devices
            let mut device_count = 0;
            let mut in_agent = false;
            
            for line in info.lines() {
                if line.starts_with("Agent ") && line.contains("Name:") {
                    device_count += 1;
                    in_agent = true;
                    println!("\n--- Device {} ---", device_count);
                    println!("{}", line.trim());
                } else if in_agent && (line.trim().starts_with("Uuid:") || 
                                       line.trim().starts_with("Marketing Name:") ||
                                       line.trim().starts_with("Vendor Name:") ||
                                       line.trim().starts_with("Feature:") ||
                                       line.trim().starts_with("Max Queue Size:") ||
                                       line.trim().starts_with("Queue Min Size:") ||
                                       line.trim().starts_with("Queue Max Size:") ||
                                       line.trim().starts_with("Processing Units:") ||
                                       line.trim().starts_with("Max Waves Per CU:") ||
                                       line.trim().starts_with("Max Work-group Size:") ||
                                       line.trim().starts_with("Grid Max Size:") ||
                                       line.trim().starts_with("Local Memory Size:") ||
                                       line.trim().starts_with("Group Memory Size:") ||
                                       line.trim().starts_with("Cache Info:")) {
                    println!("  {}", line.trim());
                } else if line.trim().is_empty() && in_agent {
                    in_agent = false;
                }
            }
            
            if device_count == 0 {
                println!("⚠️  No ROCm GPU devices found");
            } else {
                println!("\n✓ Found {} ROCm-compatible device(s)", device_count);
            }
        }
        Err(e) => {
            println!("❌ Failed to run rocminfo: {}", e);
            println!("   Make sure ROCm is properly installed and rocminfo is in PATH");
        }
    }
    
    println!();
}

fn list_devices_hipinfo() {
    println!("🔧 Using HIP device query...");
    
    // Try to run a simple HIP program to enumerate devices
    let hip_code = r#"
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipError_t err = hipGetDeviceCount(&deviceCount);
    
    if (err != hipSuccess) {
        std::cout << "HIP Error: " << hipGetErrorString(err) << std::endl;
        return 1;
    }
    
    std::cout << "HIP Device Count: " << deviceCount << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        
        std::cout << "--- HIP Device " << i << " ---" << std::endl;
        std::cout << "  Name: " << prop.name << std::endl;
        std::cout << "  Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB" << std::endl;
        std::cout << "  Multiprocessors: " << prop.multiProcessorCount << std::endl;
        std::cout << "  Max Threads Per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "  Warp Size: " << prop.warpSize << std::endl;
        std::cout << "  Memory Clock Rate: " << prop.memoryClockRate << " kHz" << std::endl;
        std::cout << "  Memory Bus Width: " << prop.memoryBusWidth << " bits" << std::endl;
        std::cout << "  PCI Bus ID: " << prop.pciBusID << std::endl;
        std::cout << "  PCI Device ID: " << prop.pciDeviceID << std::endl;
    }
    
    return 0;
}
"#;
    
    // Write temporary HIP program
    match std::fs::write("/tmp/hip_devices.cpp", hip_code) {
        Ok(_) => {
            // Try to compile and run
            let compile_result = Command::new("hipcc")
                .args(&["/tmp/hip_devices.cpp", "-o", "/tmp/hip_devices"])
                .output();
                
            match compile_result {
                Ok(compile_output) => {
                    if compile_output.status.success() {
                        match Command::new("/tmp/hip_devices").output() {
                            Ok(run_output) => {
                                let output = String::from_utf8_lossy(&run_output.stdout);
                                println!("{}", output);
                            }
                            Err(e) => {
                                println!("❌ Failed to run HIP device query: {}", e);
                            }
                        }
                    } else {
                        let compile_error = String::from_utf8_lossy(&compile_output.stderr);
                        println!("❌ Failed to compile HIP program: {}", compile_error);
                    }
                }
                Err(e) => {
                    println!("⚠️  hipcc not found or failed: {}", e);
                    println!("   This is normal if HIP development tools aren't installed");
                }
            }
            
            // Clean up
            let _ = std::fs::remove_file("/tmp/hip_devices.cpp");
            let _ = std::fs::remove_file("/tmp/hip_devices");
        }
        Err(e) => {
            println!("❌ Failed to write temporary file: {}", e);
        }
    }
    
    println!();
}

fn check_gpu_visibility() {
    println!("🔍 Checking GPU visibility...");
    
    // Check /sys/class/drm for AMD GPUs
    match Command::new("find").args(&["/sys/class/drm", "-name", "card*"]).output() {
        Ok(output) => {
            let cards = String::from_utf8_lossy(&output.stdout);
            let mut amd_cards = 0;
            
            for card in cards.lines() {
                if let Ok(vendor) = std::fs::read_to_string(format!("{}/device/vendor", card)) {
                    if vendor.trim() == "0x1002" { // AMD vendor ID
                        amd_cards += 1;
                        if let Ok(device) = std::fs::read_to_string(format!("{}/device/device", card)) {
                            println!("  AMD GPU found: {} (vendor: {}, device: {})", 
                                   card, vendor.trim(), device.trim());
                        }
                    }
                }
            }
            
            if amd_cards == 0 {
                println!("⚠️  No AMD GPUs found in /sys/class/drm");
            } else {
                println!("✓ Found {} AMD GPU(s) in system", amd_cards);
            }
        }
        Err(e) => {
            println!("❌ Failed to check /sys/class/drm: {}", e);
        }
    }
    
    // Check ROCm device visibility
    match Command::new("ls").arg("/dev/kfd").output() {
        Ok(_) => {
            println!("✓ /dev/kfd exists - ROCm kernel driver loaded");
        }
        Err(_) => {
            println!("⚠️  /dev/kfd not found - ROCm kernel driver may not be loaded");
        }
    }
    
    match Command::new("ls").args(&["/dev/dri/", "-la"]).output() {
        Ok(output) => {
            let dri_info = String::from_utf8_lossy(&output.stdout);
            let render_nodes: Vec<&str> = dri_info.lines()
                .filter(|line| line.contains("renderD"))
                .collect();
                
            if render_nodes.is_empty() {
                println!("⚠️  No render nodes found in /dev/dri/");
            } else {
                println!("✓ Found {} render node(s) in /dev/dri/:", render_nodes.len());
                for node in render_nodes {
                    println!("  {}", node.trim());
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to check /dev/dri/: {}", e);
        }
    }
    
    // Check environment variables
    println!("\n📋 ROCm Environment Variables:");
    for var in ["HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "HSA_OVERRIDE_GFX_VERSION"] {
        match std::env::var(var) {
            Ok(value) => println!("  {}: {}", var, value),
            Err(_) => println!("  {}: (not set)", var),
        }
    }
    
    println!();
}