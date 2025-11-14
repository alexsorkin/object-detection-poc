#!/bin/bash

# Comprehensive benchmarking script for Military Target Detector
# Runs all benchmarks and generates detailed reports

set -e

echo "🚀 Starting comprehensive benchmarking suite for Military Target Detector"
echo "============================================================================"

# Set up environment
export RUST_LOG=warn  # Reduce noise during benchmarking
export RUST_BACKTRACE=1

# Check if models are available
MODEL_DIR="../models"
if [[ -n "${DEFENITY_MODEL_DIR}" ]]; then
    MODEL_DIR="${DEFENITY_MODEL_DIR}"
fi

echo "📁 Model directory: ${MODEL_DIR}"

# Check for required models
R18_MODEL="${MODEL_DIR}/rtdetr_v2_r18vd_fp32.onnx"
R34_MODEL="${MODEL_DIR}/rtdetr_v2_r34vd_fp32.onnx" 
R50_MODEL="${MODEL_DIR}/rtdetr_v2_r50vd_fp32.onnx"

if [[ ! -f "${R18_MODEL}" ]]; then
    echo "⚠️  Warning: R18 model not found at ${R18_MODEL}"
    echo "   Some detection benchmarks will be skipped"
fi

if [[ ! -f "${R34_MODEL}" ]]; then
    echo "⚠️  Warning: R34 model not found at ${R34_MODEL}"
fi

if [[ ! -f "${R50_MODEL}" ]]; then
    echo "⚠️  Warning: R50 model not found at ${R50_MODEL}"
fi

# Create output directory for reports
REPORT_DIR="benchmark_reports"
mkdir -p "${REPORT_DIR}"

echo ""
echo "📊 Running benchmarks..."
echo "   Reports will be saved to: ${REPORT_DIR}/"

# Function to run a benchmark with error handling
run_benchmark() {
    local benchmark_name="$1"
    local description="$2"
    
    echo ""
    echo "⏱️  Running ${description}..."
    echo "   Benchmark: ${benchmark_name}"
    
    if cargo bench --bench "${benchmark_name}"; then
        echo "   ✅ ${description} completed successfully"
        
        # Move HTML report to our reports directory
        if [[ -d "target/criterion" ]]; then
            cp -r target/criterion "${REPORT_DIR}/${benchmark_name}_report" 2>/dev/null || true
        fi
    else
        echo "   ❌ ${description} failed"
        return 1
    fi
}

# Run image processing benchmarks (fastest, no model dependencies)
run_benchmark "image_processing" "Image Processing Benchmarks"

# Run tracking benchmarks (fast, no model dependencies)
run_benchmark "object_tracking" "Object Tracking Benchmarks"

# Run detection benchmarks (slower, requires models)
if [[ -f "${R18_MODEL}" ]]; then
    run_benchmark "rtdetr_detection" "RT-DETR Detection Benchmarks"
else
    echo "⏭️  Skipping detection benchmarks - models not available"
fi

# Run pipeline benchmarks (slowest, requires models)
if [[ -f "${R18_MODEL}" ]]; then
    run_benchmark "frame_pipeline" "Frame Pipeline Benchmarks"
else
    echo "⏭️  Skipping pipeline benchmarks - models not available"
fi

echo ""
echo "📈 Benchmark Summary"
echo "===================="

# Generate summary report
SUMMARY_FILE="${REPORT_DIR}/benchmark_summary.md"
cat > "${SUMMARY_FILE}" << EOF
# Military Target Detector - Benchmark Summary

Generated on: $(date)
System: $(uname -a)
Rust Version: $(rustc --version)

## Environment

- Model Directory: ${MODEL_DIR}
- R18 Model Available: $(if [[ -f "${R18_MODEL}" ]]; then echo "✅"; else echo "❌"; fi)
- R34 Model Available: $(if [[ -f "${R34_MODEL}" ]]; then echo "✅"; else echo "❌"; fi)  
- R50 Model Available: $(if [[ -f "${R50_MODEL}" ]]; then echo "✅"; else echo "❌"; fi)

## Benchmark Categories

### 🖼️ Image Processing Benchmarks
Tests core image manipulation functions including:
- Scale factor calculation
- Annotation data preparation  
- Batch rectangle/text drawing
- Complete annotation pipeline
- Resolution impact analysis

### 🎯 Tracking Algorithm Benchmarks  
Tests multi-object tracking performance:
- Kalman filter tracking
- ByteTrack algorithm  
- Algorithm comparison
- Sequence tracking (multi-frame)
- IoU threshold impact

### 🤖 Detection Model Benchmarks
Tests neural network inference:
- RT-DETR detection across resolutions
- Batch detection performance
- Model variant comparison (R18/R34/R50)
- Confidence threshold impact

### 🔄 Pipeline Integration Benchmarks
Tests complete system integration:
- Frame executor performance
- Detection pipeline processing
- Video pipeline with tracking
- Frame submission rates
- Backpressure handling

## Results

Detailed results are available in the individual report directories:
- \`image_processing_report/\`
- \`object_tracking_report/\`
- \`rtdetr_detection_report/\`
- \`frame_pipeline_report/\`

## Usage

To view HTML reports, open the \`index.html\` file in each report directory.

To re-run specific benchmarks:
\`\`\`bash
cargo bench --bench <benchmark_name>
\`\`\`

To run all benchmarks:
\`\`\`bash
./run_benchmarks.sh
\`\`\`
EOF

echo "📋 Summary report created: ${SUMMARY_FILE}"
echo ""

# Display report locations
echo "📁 Benchmark Reports Generated:"
for report_dir in "${REPORT_DIR}"/*_report; do
    if [[ -d "${report_dir}" ]]; then
        benchmark_name=$(basename "${report_dir}" "_report")
        if [[ -f "${report_dir}/index.html" ]]; then
            echo "   📊 ${benchmark_name}: ${report_dir}/index.html"
        else
            echo "   📊 ${benchmark_name}: ${report_dir}/"
        fi
    fi
done

echo ""
echo "🎉 Benchmarking complete!"
echo ""
echo "📖 To view results:"
echo "   1. Open HTML reports in your browser"
echo "   2. Read the summary: ${SUMMARY_FILE}"
echo "   3. Compare performance across different configurations"
echo ""
echo "💡 Tips:"
echo "   - Run benchmarks on a quiet system for consistent results"
echo "   - Use release builds: cargo bench --release"
echo "   - Monitor system resources during benchmarking"
echo "   - Compare results before and after optimizations"