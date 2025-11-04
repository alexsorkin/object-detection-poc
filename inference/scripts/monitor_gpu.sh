#!/bin/bash
# Monitor GPU usage during inference
# Usage: ./scripts/monitor_gpu.sh

echo "🔍 Monitoring GPU usage..."
echo "Run your inference in another terminal: cargo run --release --features metal --example detect"
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

while true; do
    echo "=== GPU Activity @ $(date +%H:%M:%S) ==="
    
    # Check AMD GPU activity
    amd_stats=$(ioreg -r -c "AMDRadeonX6000_AmdRadeonController" -d 1 2>/dev/null | grep -A 5 "PerformanceStatistics")
    if [ -n "$amd_stats" ]; then
        echo "AMD Radeon Pro 5500M: ACTIVE"
    else
        echo "AMD Radeon Pro 5500M: idle"
    fi
    
    # Check Intel GPU activity
    intel_stats=$(ioreg -r -c "IntelAccelerator" -d 1 2>/dev/null | grep -A 5 "PerformanceStatistics")
    if [ -n "$intel_stats" ]; then
        echo "Intel UHD Graphics 630: ACTIVE"
    else
        echo "Intel UHD Graphics 630: idle"
    fi
    
    echo ""
    sleep 1
done
