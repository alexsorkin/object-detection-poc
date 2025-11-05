# RT-DETR Batch Inference Fix

## Problem
RT-DETR model (`rf-detr-medium.onnx`) was exported with **fixed batch size = 1**, causing batch inference to fail:

```
⚠️  Batch inference not supported, falling back to sequential processing
🔥 Sequential Execution: 2 images in 786.8ms (2.5 FPS per image)
```

**Current behavior:**
- Commands ARE enqueued in parallel ✅
- BatchExecutor DOES try to batch them ✅  
- But RT-DETR model rejects batch input ❌
- Falls back to processing tiles sequentially (one at a time)

**Performance impact:**
- Sequential: ~512ms (tile 1) + ~379ms (tile 2) = ~891ms total
- True batch would be: ~450ms for both tiles in parallel (2x speedup)

## Solution
Re-export RT-DETR model with **dynamic batch size** support.

### Step 1: Export RT-DETR with Dynamic Batch
```bash
cd /Users/alexandersorkin/defenity/model/scripts

# Install requirements if needed
pip install transformers torch onnx onnxruntime

# Export RT-DETR-R50vd (medium model, ~100MB)
python export_rtdetr_batch.py --model r50vd

# Or export RT-DETR-R101vd (larger model, more accurate, ~200MB)
python export_rtdetr_batch.py --model r101vd

# Or export both
python export_rtdetr_batch.py --model both
```

**Output:**
- `../models/rtdetr_r50vd_batch.onnx` - Dynamic batch support, 640x640 input
- Model signature will show: `Shape: ['batch', 3, 640, 640]` (batch is dynamic)

### Step 2: Update RT-DETR Detector
The `detect_batch()` method is already implemented correctly in `src/detector_rtdetr.rs`.

**Just need to:**
1. Update `input_size` if the new model uses 640x640 instead of 576x576
2. Update model path to point to `rtdetr_r50vd_batch.onnx`

### Step 3: Test Batch Inference
```bash
cd /Users/alexandersorkin/defenity/model/inference

# Run RT-DETR pipeline
cargo run --release --features metal --example detect_pipeline -- --detector rtdetr
```

**Expected output:**
```
🔥 GPU Batch Execution: 2 images in 450.0ms (4.4 FPS per image)
📊 Per-Tile End-to-End Latency (submission → response):
  Tile #0: 470.0ms (2.1 FPS) - 2 detections
  Tile #1: 465.0ms (2.1 FPS) - 2 detections
  Average latency: 467.5ms (2.1 FPS)
```

**Key indicators of success:**
- ✅ No "⚠️ Batch inference not supported" warning
- ✅ Shows "🔥 GPU Batch Execution" (not Sequential)
- ✅ Both tiles have similar latency (~450-470ms)
- ✅ Total execution ~450-500ms (not ~800-900ms)

## Architecture Verification

### Current (YOLOv8 - Working ✅)
```
2 tiles submitted (44ms) 
    ↓
BatchExecutor collects into batch 
    ↓
🔥 GPU Batch Execution: 2 images in 871ms
    ↓
Both tiles processed in parallel on GPU
```

### Current (RT-DETR - Broken ❌)
```
2 tiles submitted (2.5ms)
    ↓
BatchExecutor tries to batch
    ↓
Model rejects batch (expects batch_size=1)
    ↓
⚠️ Falls back to sequential processing
    ↓
Tile #0: 512ms, then Tile #1: 379ms = 891ms total
```

### After Fix (RT-DETR - Fixed ✅)
```
2 tiles submitted (2.5ms)
    ↓
BatchExecutor collects into batch
    ↓
🔥 GPU Batch Execution: 2 images in 450ms
    ↓
Both tiles processed in parallel on GPU
```

## Technical Details

### Why Fixed Batch Size Fails
ONNX models can be exported with:
- **Static batch**: `[1, 3, 640, 640]` - Only accepts exactly 1 image
- **Dynamic batch**: `['batch', 3, 640, 640]` - Accepts 1, 2, 4, 8, etc.

The downloaded `rf-detr-medium.onnx` has static batch=1, so when we pass `[2, 3, 576, 576]`, ONNX Runtime throws:
```
Error: Got invalid dimensions for input: images. Expected: 1, Got: 2
```

### Export Script Key Points
```python
torch.onnx.export(
    model,
    dummy_input,
    output_path,
    dynamic_axes={
        'images': {0: 'batch'},      # ← Makes batch dimension dynamic
        'pred_logits': {0: 'batch'}, # ← Output batch matches input
        'pred_boxes': {0: 'batch'}
    }
)
```

### Input Size Differences
- **YOLOv8**: 640×640 input
- **RT-DETR (rf-detr)**: 576×576 input  
- **RT-DETR (standard)**: 640×640 input

If we use the standard RT-DETR from Hugging Face, it will use 640×640 like YOLO, which may simplify configuration.

## Performance Expectations

### Before Fix (Sequential)
- 2 tiles × ~450ms/tile = ~900ms total
- Limited by sequential processing
- GPU not fully utilized

### After Fix (Parallel Batch)
- 2 tiles in parallel = ~450-500ms total (2x speedup)
- GPU processes both tiles simultaneously
- Better GPU utilization
- Matches or beats YOLO performance

### Multi-Image Scenarios
With dynamic batch, we can process:
- 2 tiles from 1 image (current use case)
- 4 tiles from 2 images in parallel
- 8 tiles from 4 images in parallel
- etc.

This enables efficient multi-image processing for video streams or multiple camera feeds.

## Troubleshooting

### If export fails with "Model not found"
```bash
# Make sure transformers and torch are installed
pip install transformers torch onnx onnxruntime-gpu  # or onnxruntime for CPU
```

### If exported model is too large
- RT-DETR-R50vd: ~100-130 MB (medium accuracy)
- RT-DETR-R101vd: ~180-220 MB (higher accuracy)
- Consider using R50vd for speed, R101vd for accuracy

### If batch inference still fails after export
Check model signature:
```bash
python -c "import onnx; m=onnx.load('../models/rtdetr_r50vd_batch.onnx'); print([i.name + str([d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])"
```

Should show: `['batch', 3, 640, 640]` not `[1, 3, 640, 640]`

## Alternative: Use Ultralytics RT-DETR

If Hugging Face export doesn't work, Ultralytics also provides RT-DETR:

```bash
# Install ultralytics
pip install ultralytics

# Export with dynamic batch
python -c "
from ultralytics import RTDETR
model = RTDETR('rtdetr-l.pt')
model.export(format='onnx', dynamic=True, batch=8, simplify=True)
"
```

This exports `rtdetr-l.onnx` with dynamic batch support.
