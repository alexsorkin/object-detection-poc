//! Type definitions for military target detection

use serde::{Deserialize, Serialize};

/// Target classes for military detection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u32)]
pub enum TargetClass {
    ArmedPersonnel = 0,
    RocketLauncher = 1,
    MilitaryVehicle = 2,
    HeavyWeapon = 3,
}

impl TargetClass {
    /// Get class name as string
    pub fn name(&self) -> &'static str {
        match self {
            Self::ArmedPersonnel => "armed_personnel",
            Self::RocketLauncher => "rocket_launcher",
            Self::MilitaryVehicle => "military_vehicle",
            Self::HeavyWeapon => "heavy_weapon",
        }
    }

    /// Get class color for visualization (RGB)
    pub fn color(&self) -> [u8; 3] {
        match self {
            Self::ArmedPersonnel => [255, 0, 0],  // Red
            Self::RocketLauncher => [0, 255, 0],  // Green
            Self::MilitaryVehicle => [0, 0, 255], // Blue
            Self::HeavyWeapon => [255, 255, 0],   // Yellow
        }
    }

    /// Create from class ID
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::ArmedPersonnel),
            1 => Some(Self::RocketLauncher),
            2 => Some(Self::MilitaryVehicle),
            3 => Some(Self::HeavyWeapon),
            _ => None,
        }
    }

    /// Get all target classes
    pub fn all() -> Vec<Self> {
        vec![
            Self::ArmedPersonnel,
            Self::RocketLauncher,
            Self::MilitaryVehicle,
            Self::HeavyWeapon,
        ]
    }
}

/// Bounding box coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X coordinate of top-left corner (normalized 0-1)
    pub x: f32,
    /// Y coordinate of top-left corner (normalized 0-1)
    pub y: f32,
    /// Width of bounding box (normalized 0-1)
    pub width: f32,
    /// Height of bounding box (normalized 0-1)
    pub height: f32,
}

impl BoundingBox {
    /// Create new bounding box
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Get center point coordinates
    pub fn center(&self) -> (f32, f32) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    /// Get area of bounding box
    pub fn area(&self) -> f32 {
        self.width * self.height
    }

    /// Convert to pixel coordinates given image dimensions
    pub fn to_pixels(&self, img_width: u32, img_height: u32) -> PixelBoundingBox {
        PixelBoundingBox {
            x: (self.x * img_width as f32) as u32,
            y: (self.y * img_height as f32) as u32,
            width: (self.width * img_width as f32) as u32,
            height: (self.height * img_height as f32) as u32,
        }
    }

    /// Check if two bounding boxes intersect
    pub fn intersects(&self, other: &BoundingBox) -> bool {
        let x_overlap = self.x < other.x + other.width && self.x + self.width > other.x;
        let y_overlap = self.y < other.y + other.height && self.y + self.height > other.y;
        x_overlap && y_overlap
    }

    /// Calculate intersection over union (IoU) with another bounding box
    pub fn iou(&self, other: &BoundingBox) -> f32 {
        if !self.intersects(other) {
            return 0.0;
        }

        let x_left = self.x.max(other.x);
        let y_top = self.y.max(other.y);
        let x_right = (self.x + self.width).min(other.x + other.width);
        let y_bottom = (self.y + self.height).min(other.y + other.height);

        let intersection_area = (x_right - x_left) * (y_bottom - y_top);
        let union_area = self.area() + other.area() - intersection_area;

        intersection_area / union_area
    }
}

/// Bounding box in pixel coordinates
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PixelBoundingBox {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl PixelBoundingBox {
    /// Create new pixel bounding box
    pub fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Convert to normalized coordinates given image dimensions
    pub fn to_normalized(&self, img_width: u32, img_height: u32) -> BoundingBox {
        BoundingBox {
            x: self.x as f32 / img_width as f32,
            y: self.y as f32 / img_height as f32,
            width: self.width as f32 / img_width as f32,
            height: self.height as f32 / img_height as f32,
        }
    }
}

/// Single detection result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Detection {
    /// Target class
    pub class: TargetClass,
    /// Detection confidence score (0-1)
    pub confidence: f32,
    /// Bounding box coordinates (normalized)
    pub bbox: BoundingBox,
}

impl Detection {
    /// Create new detection
    pub fn new(class: TargetClass, confidence: f32, bbox: BoundingBox) -> Self {
        Self {
            class,
            confidence,
            bbox,
        }
    }
}

/// Detection results for a single frame
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionResult {
    /// List of detected targets
    pub detections: Vec<Detection>,
    /// Inference time in milliseconds
    pub inference_time_ms: f32,
    /// Input image dimensions
    pub image_width: u32,
    pub image_height: u32,
    /// Frame timestamp (optional)
    pub timestamp: Option<u64>,
}

impl DetectionResult {
    /// Create new detection result
    pub fn new(
        detections: Vec<Detection>,
        inference_time_ms: f32,
        image_width: u32,
        image_height: u32,
    ) -> Self {
        Self {
            detections,
            inference_time_ms,
            image_width,
            image_height,
            timestamp: None,
        }
    }

    /// Set timestamp
    pub fn with_timestamp(mut self, timestamp: u64) -> Self {
        self.timestamp = Some(timestamp);
        self
    }

    /// Get number of detections
    pub fn count(&self) -> usize {
        self.detections.len()
    }

    /// Filter detections by minimum confidence
    pub fn filter_by_confidence(mut self, min_confidence: f32) -> Self {
        self.detections
            .retain(|det| det.confidence >= min_confidence);
        self
    }

    /// Filter detections by target class
    pub fn filter_by_class(mut self, target_class: TargetClass) -> Self {
        self.detections.retain(|det| det.class == target_class);
        self
    }

    /// Get detections sorted by confidence (descending)
    pub fn sorted_by_confidence(mut self) -> Self {
        self.detections
            .sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        self
    }
}

/// Configuration for the military target detector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Path to ONNX model file
    pub model_path: String,

    /// Input image size (width, height)
    pub input_size: (u32, u32),

    /// Confidence threshold for detections (0-1)
    pub confidence_threshold: f32,

    /// Non-maximum suppression threshold (0-1)
    pub nms_threshold: f32,

    /// Maximum number of detections per frame
    pub max_detections: usize,

    /// Use GPU acceleration if available
    pub use_gpu: bool,

    /// GPU device ID (for multi-GPU systems)
    pub gpu_device_id: i32,

    /// Number of threads for CPU inference
    pub num_threads: Option<usize>,

    /// Enable optimization for inference speed
    pub optimize_for_speed: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: "models/military_targets.onnx".to_string(),
            input_size: (640, 640),
            confidence_threshold: 0.5,
            nms_threshold: 0.45,
            max_detections: 100,
            use_gpu: true,
            gpu_device_id: 0,
            num_threads: None,
            optimize_for_speed: true,
        }
    }
}

/// Image format for input data
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImageFormat {
    RGB,
    BGR,
    RGBA,
    BGRA,
    Grayscale,
}

/// Model information and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Input tensor name
    pub input_name: String,
    /// Output tensor names
    pub output_names: Vec<String>,
    /// Input tensor shape
    pub input_shape: Vec<usize>,
    /// Output tensor shapes
    pub output_shapes: Vec<Vec<usize>>,
    /// Path to model file
    pub model_path: String,
}

/// Input image data
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Raw pixel data
    pub data: Vec<u8>,
    /// Image width in pixels
    pub width: u32,
    /// Image height in pixels
    pub height: u32,
    /// Pixel format
    pub format: ImageFormat,
}

impl ImageData {
    /// Create new image data
    pub fn new(data: Vec<u8>, width: u32, height: u32, format: ImageFormat) -> Self {
        Self {
            data,
            width,
            height,
            format,
        }
    }

    /// Get number of channels
    pub fn channels(&self) -> u32 {
        match self.format {
            ImageFormat::RGB | ImageFormat::BGR => 3,
            ImageFormat::RGBA | ImageFormat::BGRA => 4,
            ImageFormat::Grayscale => 1,
        }
    }

    /// Validate image data consistency
    pub fn validate(&self) -> bool {
        let expected_size = (self.width * self.height * self.channels()) as usize;
        self.data.len() == expected_size
    }
}
