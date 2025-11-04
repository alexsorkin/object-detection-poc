//! C FFI (Foreign Function Interface) bindings for Unity integration
//!
//! This module provides C-compatible functions that can be called from Unity C# scripts
//! or other languages that support C interop.

use crate::MilitaryTargetDetector;
use crate::types::{DetectorConfig, ImageData, ImageFormat, TargetClass};

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_float, c_int};
use std::ptr;
use std::slice;
use std::sync::Mutex;

// Global detector instance storage - each detector needs Mutex for mutable access
static mut DETECTORS: Option<Mutex<HashMap<u32, Mutex<MilitaryTargetDetector>>>> = None;
static mut NEXT_DETECTOR_ID: u32 = 1;

/// Initialize the detection library
#[no_mangle]
pub extern "C" fn mtd_init() -> c_int {
    match crate::init() {
        Ok(_) => {
            unsafe {
                DETECTORS = Some(Mutex::new(HashMap::new()));
            }
            0 // Success
        }
        Err(_) => -1, // Error
    }
}

/// Get library version
#[no_mangle]
pub extern "C" fn mtd_version() -> *const c_char {
    let version = CString::new(crate::version()).unwrap();
    version.into_raw()
}

/// Free string returned from C functions
#[no_mangle]
pub extern "C" fn mtd_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        unsafe {
            let _ = CString::from_raw(ptr);
        }
    }
}

/// Create new detector instance
#[no_mangle]
pub extern "C" fn mtd_create_detector(
    model_path: *const c_char,
    input_width: c_int,
    input_height: c_int,
    confidence_threshold: c_float,
    nms_threshold: c_float,
    max_detections: c_int,
    use_gpu: c_int,
) -> c_int {
    if model_path.is_null() {
        return -1;
    }

    let model_path_str = unsafe {
        match CStr::from_ptr(model_path).to_str() {
            Ok(s) => s.to_string(),
            Err(_) => return -1,
        }
    };

    let config = DetectorConfig {
        model_path: model_path_str,
        input_size: (input_width as u32, input_height as u32),
        confidence_threshold,
        nms_threshold,
        max_detections: max_detections as usize,
        use_gpu: use_gpu != 0,
        gpu_device_id: 0,
        num_threads: None,
        optimize_for_speed: true,
    };

    // Create detector with new API (no device parameter)
    match MilitaryTargetDetector::new(config) {
        Ok(detector) => {
            let detector_id = unsafe {
                let id = NEXT_DETECTOR_ID;
                NEXT_DETECTOR_ID += 1;

                if let Some(ref detectors_mutex) = DETECTORS {
                    if let Ok(mut detectors) = detectors_mutex.lock() {
                        detectors.insert(id, Mutex::new(detector));
                        id as c_int
                    } else {
                        -1
                    }
                } else {
                    -1
                }
            };
            detector_id
        }
        Err(_) => -1,
    }
}

/// Destroy detector instance
#[no_mangle]
pub extern "C" fn mtd_destroy_detector(detector_id: c_int) -> c_int {
    unsafe {
        if let Some(ref detectors_mutex) = DETECTORS {
            if let Ok(mut detectors) = detectors_mutex.lock() {
                if detectors.remove(&(detector_id as u32)).is_some() {
                    return 0; // Success
                }
            }
        }
    }
    -1 // Error
}

/// Detection result structure for C interop
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CDetection {
    pub class_id: c_int,
    pub confidence: c_float,
    pub x: c_float,
    pub y: c_float,
    pub width: c_float,
    pub height: c_float,
}

/// Detection result array structure
#[repr(C)]
pub struct CDetectionResult {
    pub detections: *mut CDetection,
    pub count: c_int,
    pub inference_time_ms: c_float,
    pub image_width: c_int,
    pub image_height: c_int,
}

/// Detect targets in image data
#[no_mangle]
pub extern "C" fn mtd_detect_image(
    detector_id: c_int,
    image_data: *const u8,
    data_length: c_int,
    width: c_int,
    height: c_int,
    format: c_int, // 0=RGB, 1=BGR, 2=RGBA, 3=BGRA, 4=Grayscale
) -> *mut CDetectionResult {
    if image_data.is_null() || data_length <= 0 {
        return ptr::null_mut();
    }

    let image_format = match format {
        0 => ImageFormat::RGB,
        1 => ImageFormat::BGR,
        2 => ImageFormat::RGBA,
        3 => ImageFormat::BGRA,
        4 => ImageFormat::Grayscale,
        _ => return ptr::null_mut(),
    };

    // Convert raw data to ImageData
    let data_slice = unsafe { slice::from_raw_parts(image_data, data_length as usize) };
    let image = ImageData::new(
        data_slice.to_vec(),
        width as u32,
        height as u32,
        image_format,
    );

    // Get detector and run inference (detector.detect() now requires &mut self)
    let result = unsafe {
        if let Some(ref detectors_mutex) = DETECTORS {
            if let Ok(detectors) = detectors_mutex.lock() {
                if let Some(detector_mutex) = detectors.get(&(detector_id as u32)) {
                    if let Ok(mut detector) = detector_mutex.lock() {
                        detector.detect(&image)
                    } else {
                        return ptr::null_mut();
                    }
                } else {
                    return ptr::null_mut();
                }
            } else {
                return ptr::null_mut();
            }
        } else {
            return ptr::null_mut();
        }
    };

    match result {
        Ok(detections) => {
            // Convert Vec<Detection> to C-compatible format
            let count = detections.len() as c_int;
            let detections_vec: Vec<CDetection> = detections
                .iter()
                .map(|det| CDetection {
                    class_id: det.class.id() as c_int,
                    confidence: det.confidence,
                    x: det.bbox.x,
                    y: det.bbox.y,
                    width: det.bbox.width,
                    height: det.bbox.height,
                })
                .collect();

            // Allocate result structure
            let c_result = Box::new(CDetectionResult {
                detections: if count > 0 {
                    let detections_ptr = detections_vec.as_ptr() as *mut CDetection;
                    std::mem::forget(detections_vec); // Prevent deallocation
                    detections_ptr
                } else {
                    ptr::null_mut()
                },
                count,
                inference_time_ms: 0.0, // Timing not tracked in simplified API
                image_width: width,
                image_height: height,
            });

            Box::into_raw(c_result)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Detect targets from image file
#[no_mangle]
pub extern "C" fn mtd_detect_file(
    detector_id: c_int,
    image_path: *const c_char,
) -> *mut CDetectionResult {
    if image_path.is_null() {
        return ptr::null_mut();
    }

    let path_str = unsafe {
        match CStr::from_ptr(image_path).to_str() {
            Ok(s) => s,
            Err(_) => return ptr::null_mut(),
        }
    };

    // Load image from file
    let image = match ImageData::from_file(path_str) {
        Ok(img) => img,
        Err(_) => return ptr::null_mut(),
    };

    // Get detector and run inference
    let result = unsafe {
        if let Some(ref detectors_mutex) = DETECTORS {
            if let Ok(detectors) = detectors_mutex.lock() {
                if let Some(detector_mutex) = detectors.get(&(detector_id as u32)) {
                    if let Ok(mut detector) = detector_mutex.lock() {
                        detector.detect(&image)
                    } else {
                        return ptr::null_mut();
                    }
                } else {
                    return ptr::null_mut();
                }
            } else {
                return ptr::null_mut();
            }
        } else {
            return ptr::null_mut();
        }
    };

    match result {
        Ok(detections) => {
            // Convert Vec<Detection> to C-compatible format
            let count = detections.len() as c_int;
            let detections_vec: Vec<CDetection> = detections
                .iter()
                .map(|det| CDetection {
                    class_id: det.class.id() as c_int,
                    confidence: det.confidence,
                    x: det.bbox.x,
                    y: det.bbox.y,
                    width: det.bbox.width,
                    height: det.bbox.height,
                })
                .collect();

            let c_result = Box::new(CDetectionResult {
                detections: if count > 0 {
                    let detections_ptr = detections_vec.as_ptr() as *mut CDetection;
                    std::mem::forget(detections_vec);
                    detections_ptr
                } else {
                    ptr::null_mut()
                },
                count,
                inference_time_ms: 0.0, // Timing not tracked in simplified API
                image_width: image.width as c_int,
                image_height: image.height as c_int,
            });

            Box::into_raw(c_result)
        }
        Err(_) => ptr::null_mut(),
    }
}

/// Free detection result
#[no_mangle]
pub extern "C" fn mtd_free_result(result: *mut CDetectionResult) {
    if result.is_null() {
        return;
    }

    unsafe {
        let boxed_result = Box::from_raw(result);
        if !boxed_result.detections.is_null() && boxed_result.count > 0 {
            let _ = Vec::from_raw_parts(
                boxed_result.detections,
                boxed_result.count as usize,
                boxed_result.count as usize,
            );
        }
    }
}

/// Get class name for class ID
#[no_mangle]
pub extern "C" fn mtd_get_class_name(class_id: c_int) -> *const c_char {
    if let Some(class) = TargetClass::from_id(class_id as u32) {
        let name = CString::new(class.name()).unwrap();
        name.into_raw()
    } else {
        ptr::null()
    }
}

/// Get class color (RGB) for class ID
#[no_mangle]
pub extern "C" fn mtd_get_class_color(class_id: c_int, rgb: *mut u8) {
    if rgb.is_null() {
        return;
    }

    if let Some(class) = TargetClass::from_id(class_id as u32) {
        let color = class.color();
        unsafe {
            *rgb.offset(0) = color[0];
            *rgb.offset(1) = color[1];
            *rgb.offset(2) = color[2];
        }
    }
}

/// Update detector configuration
/// Note: Current ONNX Runtime implementation doesn't support runtime config updates.
/// You need to recreate the detector with new config.
#[no_mangle]
pub extern "C" fn mtd_update_config(
    _detector_id: c_int,
    _confidence_threshold: c_float,
    _nms_threshold: c_float,
    _max_detections: c_int,
) -> c_int {
    // Not supported in current implementation
    // Return error code indicating feature not available
    -2
}

/// Warm up detector (run dummy inference)
/// Note: ONNX Runtime warms up automatically on first run.
#[no_mangle]
pub extern "C" fn mtd_warmup(_detector_id: c_int) -> c_int {
    // ONNX Runtime/CoreML warm up automatically
    // Return success since warmup is handled internally
    0
}

/// Get number of available target classes
#[no_mangle]
pub extern "C" fn mtd_get_class_count() -> c_int {
    TargetClass::all().len() as c_int
}

/// Cleanup library resources
#[no_mangle]
pub extern "C" fn mtd_cleanup() {
    unsafe {
        DETECTORS = None;
        NEXT_DETECTOR_ID = 1;
    }
}

// Error handling functions
#[no_mangle]
pub extern "C" fn mtd_get_last_error() -> *const c_char {
    // This is a simplified error handling approach
    // In practice, you might want to store error messages per thread
    let error_msg = CString::new("Check logs for detailed error information").unwrap();
    error_msg.into_raw()
}
