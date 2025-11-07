//! Type definitions for military target detection

use serde::{Deserialize, Serialize};

/// Target classes for detection (supports multiple model types)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TargetClass {
    // Generic class with ID
    Class(u32),
}

impl TargetClass {
    /// COCO class names (80 classes)
    const COCO_CLASSES: &'static [&'static str] = &[
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ];

    /// Objects365 class names (365 classes)
    const OBJECTS365_CLASSES: &'static [&'static str] = &[
        "person",
        "sneakers",
        "chair",
        "hat",
        "lamp",
        "bottle",
        "cabinet",
        "cup",
        "car",
        "glasses",
        "picture",
        "desk",
        "handbag",
        "street lights",
        "book",
        "plate",
        "helmet",
        "leather shoes",
        "pillow",
        "glove",
        "potted plant",
        "bracelet",
        "flower",
        "tv",
        "storage box",
        "vase",
        "bench",
        "wine glass",
        "boots",
        "bowl",
        "dining table",
        "umbrella",
        "boat",
        "flag",
        "speaker",
        "trash bin",
        "stool",
        "backpack",
        "couch",
        "belt",
        "carpet",
        "basket",
        "towel",
        "slippers",
        "barrel",
        "coffee table",
        "suv",
        "toy",
        "tie",
        "bed",
        "traffic light",
        "pen",
        "microphone",
        "sandals",
        "canned",
        "necklace",
        "mirror",
        "faucet",
        "bicycle",
        "bread",
        "high heels",
        "ring",
        "van",
        "watch",
        "sink",
        "horse",
        "fish",
        "apple",
        "camera",
        "candle",
        "teddy bear",
        "cake",
        "motorcycle",
        "wild bird",
        "laptop",
        "knife",
        "traffic sign",
        "cell phone",
        "paddle",
        "truck",
        "cow",
        "power outlet",
        "clock",
        "drum",
        "fork",
        "bus",
        "hanger",
        "nightstand",
        "pot",
        "sheep",
        "guitar",
        "traffic cone",
        "tea pot",
        "keyboard",
        "tripod",
        "hockey",
        "fan",
        "dog",
        "spoon",
        "blackboard",
        "balloon",
        "air conditioner",
        "cymbal",
        "mouse",
        "telephone",
        "pickup truck",
        "orange",
        "banana",
        "airplane",
        "luggage",
        "skis",
        "soccer",
        "trolley",
        "oven",
        "remote",
        "baseball glove",
        "paper towel",
        "refrigerator",
        "train",
        "tomato",
        "machinery vehicle",
        "tent",
        "shampoo",
        "head phone",
        "lantern",
        "donut",
        "cleaning products",
        "sailboat",
        "tangerine",
        "pizza",
        "kite",
        "computer box",
        "elephant",
        "toiletries",
        "gas stove",
        "broccoli",
        "toilet",
        "stroller",
        "shovel",
        "baseball bat",
        "microwave",
        "skateboard",
        "surfboard",
        "surveillance camera",
        "gun",
        "life saver",
        "cat",
        "lemon",
        "liquid soap",
        "zebra",
        "duck",
        "sports car",
        "giraffe",
        "pumpkin",
        "piano",
        "stop sign",
        "radiator",
        "converter",
        "tissue",
        "carrot",
        "washing machine",
        "vent",
        "cookies",
        "cutting board",
        "tennis racket",
        "candy",
        "skating shoes",
        "scissors",
        "folder",
        "baseball",
        "strawberry",
        "bow tie",
        "pigeon",
        "pepper",
        "coffee machine",
        "bathtub",
        "snowboard",
        "suitcase",
        "grapes",
        "ladder",
        "pear",
        "american football",
        "basketball",
        "potato",
        "paint brush",
        "printer",
        "billiards",
        "fire hydrant",
        "goose",
        "projector",
        "sausage",
        "fire extinguisher",
        "extension cord",
        "facial mask",
        "tennis ball",
        "chopsticks",
        "electronic stove",
        "pie",
        "frisbee",
        "kettle",
        "hamburger",
        "golf club",
        "cucumber",
        "clutch",
        "blender",
        "tong",
        "slide",
        "hot dog",
        "toothbrush",
        "facial cleanser",
        "mango",
        "deer",
        "egg",
        "violin",
        "marker",
        "ship",
        "chicken",
        "onion",
        "ice cream",
        "tape",
        "wheelchair",
        "plum",
        "bar soap",
        "scale",
        "watermelon",
        "cabbage",
        "router",
        "golf ball",
        "pine apple",
        "crane",
        "fire truck",
        "peach",
        "cello",
        "notepaper",
        "tricycle",
        "toaster",
        "helicopter",
        "green beans",
        "brush",
        "carriage",
        "cigar",
        "earphone",
        "penguin",
        "hurdle",
        "swing",
        "radio",
        "cd",
        "parking meter",
        "swan",
        "garlic",
        "french fries",
        "horn",
        "avocado",
        "saxophone",
        "trumpet",
        "sandwich",
        "cue",
        "kiwi fruit",
        "bear",
        "fishing rod",
        "cherry",
        "tablet",
        "green vegetables",
        "nuts",
        "corn",
        "key",
        "screwdriver",
        "globe",
        "broom",
        "pliers",
        "volleyball",
        "hammer",
        "eggplant",
        "trophy",
        "dates",
        "board eraser",
        "rice",
        "tape measure",
        "dumbbell",
        "hamimelon",
        "stapler",
        "camel",
        "lettuce",
        "goldfish",
        "meat balls",
        "medal",
        "toothpaste",
        "antelope",
        "shrimp",
        "rickshaw",
        "trombone",
        "pomegranate",
        "coconut",
        "jellyfish",
        "mushroom",
        "calculator",
        "treadmill",
        "butterfly",
        "egg tart",
        "cheese",
        "pig",
        "pomelo",
        "race car",
        "rice cooker",
        "tuba",
        "crosswalk sign",
        "papaya",
        "hair drier",
        "green onion",
        "chips",
        "dolphin",
        "sushi",
        "urinal",
        "donkey",
        "electric drill",
        "spring rolls",
        "tortoise",
        "parrot",
        "flute",
        "measuring cup",
        "shark",
        "steak",
        "poker card",
        "binoculars",
        "llama",
        "radish",
        "noodles",
        "yak",
        "mop",
        "crab",
        "microscope",
        "barbell",
        "bread",
        "baozi",
        "lion",
        "red cabbage",
        "polar bear",
        "lighter",
        "seal",
        "mangosteen",
        "comb",
        "eraser",
        "pitaya",
        "scallop",
        "pencil case",
        "saw",
        "table tennis paddle",
        "okra",
        "starfish",
        "eagle",
        "monkey",
        "durian",
        "game board",
        "rabbit",
        "french horn",
        "ambulance",
        "asparagus",
        "hoverboard",
        "pasta",
        "target",
        "hotair balloon",
        "chainsaw",
        "lobster",
        "iron",
        "flashlight",
    ];

    /// Get class name as string
    pub fn name(&self) -> String {
        match self {
            Self::Class(id) => {
                let id = *id as usize;

                // Try COCO classes first (most common)
                if id < Self::COCO_CLASSES.len() {
                    return Self::COCO_CLASSES[id].to_string();
                }

                // Try Objects365 classes
                if id < Self::OBJECTS365_CLASSES.len() {
                    return Self::OBJECTS365_CLASSES[id].to_string();
                }

                // Fallback to class_N
                format!("class_{}", id)
            }
        }
    }

    /// Get class color for visualization (RGB) - deterministic color based on class ID
    pub fn color(&self) -> [u8; 3] {
        match self {
            Self::Class(id) => {
                // Generate consistent colors based on class ID
                let hue = (id * 137) % 360; // Golden angle for better color distribution
                let saturation = 0.7;
                let value = 0.9;

                // Simple HSV to RGB conversion
                let c = value * saturation;
                let x = c * (1.0 - ((hue as f32 / 60.0) % 2.0 - 1.0).abs());
                let m = value - c;

                let (r, g, b) = match hue / 60 {
                    0 => (c, x, 0.0),
                    1 => (x, c, 0.0),
                    2 => (0.0, c, x),
                    3 => (0.0, x, c),
                    4 => (x, 0.0, c),
                    _ => (c, 0.0, x),
                };

                [
                    ((r + m) * 255.0) as u8,
                    ((g + m) * 255.0) as u8,
                    ((b + m) * 255.0) as u8,
                ]
            }
        }
    }

    /// Create from class ID (always succeeds)
    pub fn from_id(id: u32) -> Option<Self> {
        Some(Self::Class(id))
    }

    /// Get the numeric class ID
    pub fn id(&self) -> u32 {
        match self {
            Self::Class(id) => *id,
        }
    }

    /// Get all military target classes (for backwards compatibility)
    pub fn all() -> Vec<Self> {
        vec![
            Self::Class(0), // ArmedPersonnel
            Self::Class(1), // RocketLauncher
            Self::Class(2), // MilitaryVehicle
            Self::Class(3), // HeavyWeapon
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

    /// Load image from file path
    pub fn from_file(path: &str) -> crate::Result<Self> {
        use image::GenericImageView;

        let img = image::open(path).map_err(|e| {
            crate::DetectionError::preprocessing(format!("Failed to load image: {}", e))
        })?;

        let (width, height) = img.dimensions();
        let img_rgb = img.to_rgb8();
        let data = img_rgb.into_raw();

        Ok(Self {
            data,
            width,
            height,
            format: ImageFormat::RGB,
        })
    }

    /// Load image from bytes
    pub fn from_bytes(bytes: &[u8]) -> crate::Result<Self> {
        use image::GenericImageView;

        let img = image::load_from_memory(bytes).map_err(|e| {
            crate::DetectionError::preprocessing(format!("Failed to decode image: {}", e))
        })?;

        let (width, height) = img.dimensions();
        let img_rgb = img.to_rgb8();
        let data = img_rgb.into_raw();

        Ok(Self {
            data,
            width,
            height,
            format: ImageFormat::RGB,
        })
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

/// RT-DETR model variants with their corresponding ONNX file paths
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RTDETRModel {
    /// RT-DETR v2 with ResNet-18 backbone
    R18VD,
    /// RT-DETR v2 with ResNet-34 backbone  
    R34VD,
    /// RT-DETR v2 with ResNet-50 backbone
    R50VD,
}

impl RTDETRModel {
    /// Get the model file name for this variant
    pub fn filename(&self) -> &'static str {
        match self {
            Self::R18VD => "rtdetr_v2_r18vd_batch.onnx",
            Self::R34VD => "rtdetr_v2_r34vd_batch.onnx",
            Self::R50VD => "rtdetr_v2_r50vd_batch.onnx",
        }
    }

    /// Get the model name as a string
    pub fn name(&self) -> &'static str {
        match self {
            Self::R18VD => "RT-DETR v2 R18",
            Self::R34VD => "RT-DETR v2 R34",
            Self::R50VD => "RT-DETR v2 R50",
        }
    }

    /// Parse from string (case-insensitive)
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "r18vd" | "r18" => Some(Self::R18VD),
            "r34vd" | "r34" => Some(Self::R34VD),
            "r50vd" | "r50" => Some(Self::R50VD),
            _ => None,
        }
    }

    /// Get all available model variants
    pub fn all() -> Vec<Self> {
        vec![Self::R18VD, Self::R34VD, Self::R50VD]
    }
}

impl Default for RTDETRModel {
    fn default() -> Self {
        Self::R18VD // Smallest/fastest model as default
    }
}

impl std::fmt::Display for RTDETRModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}
