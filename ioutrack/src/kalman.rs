//! Kalman filter implementation for tracking

use anyhow::Result;
use nalgebra::{DMatrix, DVector};

#[derive(Debug, Clone)]
pub struct KalmanFilterParams {
    pub dim_x: usize,    // State dimension
    pub dim_z: usize,    // Observation dimension
    pub x: DVector<f32>, // Initial state
    pub p: DMatrix<f32>, // Initial state covariance
    pub f: DMatrix<f32>, // State transition matrix
    pub h: DMatrix<f32>, // Observation matrix
    pub r: DMatrix<f32>, // Observation noise covariance
    pub q: DMatrix<f32>, // Process noise covariance
}

#[derive(Debug, Clone)]
pub struct KalmanFilter<T> {
    pub dim_x: usize,
    pub dim_z: usize,
    pub x: DVector<T>, // State vector
    pub p: DMatrix<T>, // State covariance matrix
    pub f: DMatrix<T>, // State transition matrix
    pub h: DMatrix<T>, // Observation matrix
    pub r: DMatrix<T>, // Observation noise covariance
    pub q: DMatrix<T>, // Process noise covariance
    pub y: DVector<T>, // Residual
    pub s: DMatrix<T>, // Innovation covariance
    pub k: DMatrix<T>, // Kalman gain
}

impl KalmanFilter<f32> {
    pub fn new(params: KalmanFilterParams) -> Self {
        let dim_x = params.dim_x;
        let dim_z = params.dim_z;

        Self {
            dim_x,
            dim_z,
            x: params.x,
            p: params.p,
            f: params.f,
            h: params.h,
            r: params.r,
            q: params.q,
            y: DVector::zeros(dim_z),
            s: DMatrix::zeros(dim_z, dim_z),
            k: DMatrix::zeros(dim_x, dim_z),
        }
    }

    /// Predict the next state
    pub fn predict(&mut self) {
        // x = F * x
        self.x = &self.f * &self.x;

        // P = F * P * F^T + Q
        self.p = &self.f * &self.p * self.f.transpose() + &self.q;
    }

    /// Update with observation
    pub fn update(&mut self, z: DVector<f32>) -> Result<()> {
        // Residual: y = z - H * x
        self.y = z - &self.h * &self.x;

        // Innovation covariance: S = H * P * H^T + R
        self.s = &self.h * &self.p * self.h.transpose() + &self.r;

        // Kalman gain: K = P * H^T * S^-1
        let s_inv = self
            .s
            .clone()
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to invert innovation covariance matrix"))?;
        self.k = &self.p * self.h.transpose() * s_inv;

        // Update state: x = x + K * y
        self.x = &self.x + &self.k * &self.y;

        // Update covariance: P = (I - K * H) * P
        let i = DMatrix::identity(self.dim_x, self.dim_x);
        self.p = (i - &self.k * &self.h) * &self.p;

        Ok(())
    }

    /// Get current state
    pub fn get_state(&self) -> &DVector<f32> {
        &self.x
    }

    /// Get current covariance
    pub fn get_covariance(&self) -> &DMatrix<f32> {
        &self.p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_kalman_filter_basic() {
        // Simple 1D position tracking
        let params = KalmanFilterParams {
            dim_x: 2,                                                            // [position, velocity]
            dim_z: 1,                                                            // [position]
            x: DVector::from_vec(vec![0.0, 1.0]), // initial position=0, velocity=1
            p: DMatrix::from_diagonal(&DVector::from_vec(vec![1000.0, 1000.0])), // high initial uncertainty
            f: DMatrix::from_row_slice(2, 2, &[1.0, 1.0, 0.0, 1.0]), // x' = x + v, v' = v
            h: DMatrix::from_row_slice(1, 2, &[1.0, 0.0]),           // observe position only
            r: DMatrix::from_element(1, 1, 0.1),                     // measurement noise
            q: DMatrix::from_diagonal(&DVector::from_vec(vec![0.01, 0.01])), // process noise
        };

        let mut kf = KalmanFilter::new(params);

        // Predict
        kf.predict();
        assert_abs_diff_eq!(kf.x[0], 1.0, epsilon = 0.001); // position should be 1.0

        // Update with measurement
        let measurement = DVector::from_vec(vec![0.9]);
        kf.update(measurement).unwrap();

        // State should be somewhere between prediction and measurement
        assert!(kf.x[0] > 0.8 && kf.x[0] < 1.0);
    }
}
