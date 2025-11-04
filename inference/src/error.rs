//! Error types for the military target detection library

use thiserror::Error;

/// Result type alias for the detection library
pub type Result<T> = std::result::Result<T, DetectionError>;

/// Errors that can occur during detection operations
#[derive(Error, Debug)]
pub enum DetectionError {
    #[error("Model loading failed: {0}")]
    ModelLoadError(String),

    #[error("Inference failed: {0}")]
    InferenceError(String),

    #[error("Image preprocessing failed: {0}")]
    PreprocessingError(String),

    #[error("Invalid input dimensions: expected {expected:?}, got {actual:?}")]
    InvalidDimensions {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Unsupported image format: {0}")]
    UnsupportedImageFormat(String),

    #[error("GPU not available or initialization failed")]
    GpuError,

    #[error("Configuration error: {0}")]
    ConfigError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image processing error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("JSON serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),

    #[error("Other error: {0}")]
    Other(String),
}

impl DetectionError {
    pub fn initialization<S: Into<String>>(msg: S) -> Self {
        Self::ModelLoadError(msg.into())
    }

    pub fn model_load<S: Into<String>>(msg: S) -> Self {
        Self::ModelLoadError(msg.into())
    }

    pub fn inference<S: Into<String>>(msg: S) -> Self {
        Self::InferenceError(msg.into())
    }

    pub fn preprocessing<S: Into<String>>(msg: S) -> Self {
        Self::PreprocessingError(msg.into())
    }

    pub fn postprocessing<S: Into<String>>(msg: S) -> Self {
        Self::InferenceError(format!("Postprocessing error: {}", msg.into()))
    }

    pub fn config<S: Into<String>>(msg: S) -> Self {
        Self::ConfigError(msg.into())
    }

    pub fn other<S: Into<String>>(msg: S) -> Self {
        Self::Other(msg.into())
    }
}
