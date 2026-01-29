//! YuNet detector post-processing module
//! Handles anchor-free grid-based bbox decoding and multi-scale NMS
//!
//! YuNet is an anchor-free face detector that predicts directly from grid locations.
//! For each stride (8, 16, 32), it outputs:
//! - cls: [1, H*W, 1] - classification scores
//! - obj: [1, H*W, 1] - objectness scores
//! - bbox: [1, H*W, 4] - bbox deltas (dx, dy, dw, dh)
//! - kps: [1, H*W, 10] - landmark deltas (5 points x 2 coords)
//!
//! Decoding is grid-based without anchors/priors:
//! cx = (grid_x + dx) * stride / input_size
//! cy = (grid_y + dy) * stride / input_size
//! w = dw * stride / input_size
//! h = dh * stride / input_size

use anyhow::Result;
use ndarray::Array2;

const STRIDES: [usize; 3] = [8, 16, 32];

#[derive(Debug, Clone)]
pub struct RawDetection {
    pub bbox: [f32; 4], // x, y, w, h (normalized [0,1])
    pub score: f32,
    pub landmarks: [f32; 10], // 5 points: x1,y1,x2,y2,...,x5,y5 (normalized [0,1])
}

/// Decode YuNet output tensors to detection boxes using anchor-free grid-based decoding
///
/// YuNet outputs 3 tensors per scale (stride 8, 16, 32):
/// - cls: [1, H*W, 1] - classification scores
/// - bbox: [1, H*W, 4] - bbox deltas (dx, dy, dw, dh)
/// - landmarks: [1, H*W, 10] - landmark deltas
///
/// For 640x640 input:
/// - Stride 8:  80x80 grid = 6400 locations
/// - Stride 16: 40x40 grid = 1600 locations
/// - Stride 32: 20x20 grid = 400 locations
pub fn decode_detections(
    cls_scores: Vec<Array2<f32>>,
    bbox_preds: Vec<Array2<f32>>,
    landmark_preds: Vec<Array2<f32>>,
    score_threshold: f32,
    input_size: usize,
) -> Result<Vec<RawDetection>> {
    let mut detections = Vec::new();

    // Process each scale (stride)
    for (scale_idx, &stride) in STRIDES.iter().enumerate() {
        let scores = &cls_scores[scale_idx];
        let bboxes = &bbox_preds[scale_idx];
        let landmarks = &landmark_preds[scale_idx];

        let feature_size = input_size / stride;
        let num_boxes = scores.shape()[0];

        // Sanity check
        if num_boxes != feature_size * feature_size {
            anyhow::bail!(
                "Expected {} boxes for stride {} ({}x{} grid), got {}",
                feature_size * feature_size,
                stride,
                feature_size,
                feature_size,
                num_boxes
            );
        }

        // Iterate through grid locations
        for i in 0..feature_size {
            for j in 0..feature_size {
                let idx = i * feature_size + j;
                let score = scores[[idx, 0]];

                // Filter by score threshold
                if score < score_threshold {
                    continue;
                }

                // Get deltas from network
                let dx = bboxes[[idx, 0]];
                let dy = bboxes[[idx, 1]];
                let dw = bboxes[[idx, 2]];
                let dh = bboxes[[idx, 3]];

                // Anchor-free decoding: directly map from grid to image coordinates
                // Center point
                let cx_px = (j as f32 + dx) * stride as f32;
                let cy_px = (i as f32 + dy) * stride as f32;

                // Width and height (linear, no exp)
                let w_px = dw * stride as f32;
                let h_px = dh * stride as f32;

                // Normalize to [0, 1]
                let cx = cx_px / input_size as f32;
                let cy = cy_px / input_size as f32;
                let w = w_px / input_size as f32;
                let h = h_px / input_size as f32;

                // Convert from center format to corner format (x, y, w, h)
                let x = cx - w / 2.0;
                let y = cy - h / 2.0;

                // Decode landmarks similarly (anchor-free, grid-based)
                let mut lms = [0.0f32; 10];
                for k in 0..5 {
                    let lm_dx = landmarks[[idx, k * 2]];
                    let lm_dy = landmarks[[idx, k * 2 + 1]];

                    // Map from grid to image coordinates
                    let lm_x_px = (j as f32 + lm_dx) * stride as f32;
                    let lm_y_px = (i as f32 + lm_dy) * stride as f32;

                    // Normalize to [0, 1]
                    lms[k * 2] = lm_x_px / input_size as f32;
                    lms[k * 2 + 1] = lm_y_px / input_size as f32;
                }

                detections.push(RawDetection {
                    bbox: [x, y, w, h],
                    score,
                    landmarks: lms,
                });
            }
        }
    }

    Ok(detections)
}

/// Parse YuNet raw outputs into structured tensors
///
/// YuNet outputs 12 tensors total (3 scales × 4 outputs per scale):
/// For each scale (stride 8, 16, 32):
///   - cls: classification score [1, H*W, 1]
///   - obj: objectness score [1, H*W, 1]
///   - bbox: bounding box regression [1, H*W, 4]
///   - kps: landmarks [1, H*W, 10]
///
/// Output order: cls_8, cls_16, cls_32, obj_8, obj_16, obj_32, bbox_8, bbox_16, bbox_32, kps_8, kps_16, kps_32
pub fn parse_yunet_outputs(
    outputs: &[(&[i64], &[f32])],
    input_size: usize,
) -> Result<(Vec<Array2<f32>>, Vec<Array2<f32>>, Vec<Array2<f32>>)> {
    let mut cls_scores = Vec::new();
    let mut bbox_preds = Vec::new();
    let mut landmark_preds = Vec::new();

    // Expected grid sizes for 640x640 input
    let expected_counts = [
        (input_size / 8) * (input_size / 8),   // stride 8:  80x80 = 6400
        (input_size / 16) * (input_size / 16), // stride 16: 40x40 = 1600
        (input_size / 32) * (input_size / 32), // stride 32: 20x20 = 400
    ];

    // Parse classification scores (outputs 0-2)
    let mut cls_scores_vec = Vec::new();
    for (idx, &expected_count) in expected_counts.iter().enumerate() {
        if let Some((shape, data)) = outputs.get(idx) {
            // Verify shape
            if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
                anyhow::bail!(
                    "Unexpected cls shape at index {}: {:?}, expected [1, {}, 1]",
                    idx,
                    shape,
                    expected_count
                );
            }
            let actual_count = shape[1] as usize;
            if actual_count != expected_count {
                anyhow::bail!(
                    "Expected {} locations for cls at index {}, got {}",
                    expected_count,
                    idx,
                    actual_count
                );
            }

            let arr = Array2::from_shape_vec((expected_count, 1), data.to_vec())?;
            cls_scores_vec.push(arr);
        } else {
            anyhow::bail!("Missing cls output at index {}", idx);
        }
    }

    // Parse objectness scores (outputs 3-5)
    let mut obj_scores_vec = Vec::new();
    for (idx, &expected_count) in expected_counts.iter().enumerate() {
        if let Some((shape, data)) = outputs.get(idx + 3) {
            if shape.len() != 3 || shape[0] != 1 || shape[2] != 1 {
                anyhow::bail!(
                    "Unexpected obj shape at index {}: {:?}, expected [1, {}, 1]",
                    idx + 3,
                    shape,
                    expected_count
                );
            }

            let arr = Array2::from_shape_vec((expected_count, 1), data.to_vec())?;
            obj_scores_vec.push(arr);
        } else {
            anyhow::bail!("Missing obj output at index {}", idx + 3);
        }
    }

    // Combine cls and obj scores (element-wise multiplication)
    for (cls_raw, obj) in cls_scores_vec.iter().zip(obj_scores_vec.iter()) {
        let combined = cls_raw * obj;
        cls_scores.push(combined);
    }

    // Parse bounding boxes (outputs 6-8)
    for (idx, &expected_count) in expected_counts.iter().enumerate() {
        if let Some((shape, data)) = outputs.get(idx + 6) {
            if shape.len() != 3 || shape[0] != 1 || shape[2] != 4 {
                anyhow::bail!(
                    "Unexpected bbox shape at index {}: {:?}, expected [1, {}, 4]",
                    idx + 6,
                    shape,
                    expected_count
                );
            }

            let arr = Array2::from_shape_vec((expected_count, 4), data.to_vec())?;
            bbox_preds.push(arr);
        } else {
            anyhow::bail!("Missing bbox output at index {}", idx + 6);
        }
    }

    // Parse landmarks (outputs 9-11)
    for (idx, &expected_count) in expected_counts.iter().enumerate() {
        if let Some((shape, data)) = outputs.get(idx + 9) {
            if shape.len() != 3 || shape[0] != 1 || shape[2] != 10 {
                anyhow::bail!(
                    "Unexpected kps shape at index {}: {:?}, expected [1, {}, 10]",
                    idx + 9,
                    shape,
                    expected_count
                );
            }

            let arr = Array2::from_shape_vec((expected_count, 10), data.to_vec())?;
            landmark_preds.push(arr);
        } else {
            anyhow::bail!("Missing kps output at index {}", idx + 9);
        }
    }

    Ok((cls_scores, bbox_preds, landmark_preds))
}

/// Apply sigmoid activation to scores
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Apply sigmoid to all classification scores
pub fn apply_sigmoid_to_scores(scores: &mut [Array2<f32>]) {
    for score_map in scores {
        score_map.mapv_inplace(sigmoid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_decode_grid_based() {
        // Test anchor-free decoding for a single detection
        let input_size = 640;

        // Create mock outputs for all three strides
        // Stride 8: 80x80 = 6400
        let stride8_size = 80 * 80;
        let scores_8 = Array2::from_shape_vec((stride8_size, 1), vec![0.0; stride8_size]).unwrap();
        let bbox_8 =
            Array2::from_shape_vec((stride8_size, 4), vec![0.0; stride8_size * 4]).unwrap();
        let lm_8 =
            Array2::from_shape_vec((stride8_size, 10), vec![0.0; stride8_size * 10]).unwrap();

        // Stride 16: 40x40 = 1600
        let stride16_size = 40 * 40;
        let scores_16 =
            Array2::from_shape_vec((stride16_size, 1), vec![0.0; stride16_size]).unwrap();
        let bbox_16 =
            Array2::from_shape_vec((stride16_size, 4), vec![0.0; stride16_size * 4]).unwrap();
        let lm_16 =
            Array2::from_shape_vec((stride16_size, 10), vec![0.0; stride16_size * 10]).unwrap();

        // Stride 32: 20x20 = 400 (with one detection)
        let feature_size = 20;
        let mut scores_data = vec![0.0; feature_size * feature_size];
        let mut bbox_data = vec![0.0; feature_size * feature_size * 4];
        let mut lm_data = vec![0.0; feature_size * feature_size * 10];

        // Set up one high-scoring detection at grid position (10, 10)
        let grid_i = 10;
        let grid_j = 10;
        let idx = grid_i * feature_size + grid_j;

        scores_data[idx] = 0.9;

        // Deltas: dx=0.5, dy=0.3, dw=128, dh=128 (pixels in stride units)
        bbox_data[idx * 4 + 0] = 0.5;
        bbox_data[idx * 4 + 1] = 0.3;
        bbox_data[idx * 4 + 2] = 4.0; // 4 * stride = 128 pixels
        bbox_data[idx * 4 + 3] = 4.0;

        // Landmark at center (offset 0, 0)
        lm_data[idx * 10 + 0] = 0.0;
        lm_data[idx * 10 + 1] = 0.0;

        let scores_32 =
            Array2::from_shape_vec((feature_size * feature_size, 1), scores_data).unwrap();
        let bbox_32 = Array2::from_shape_vec((feature_size * feature_size, 4), bbox_data).unwrap();
        let lm_32 = Array2::from_shape_vec((feature_size * feature_size, 10), lm_data).unwrap();

        let scores = vec![scores_8, scores_16, scores_32];
        let bboxes = vec![bbox_8, bbox_16, bbox_32];
        let landmarks = vec![lm_8, lm_16, lm_32];

        let detections = decode_detections(scores, bboxes, landmarks, 0.5, input_size).unwrap();

        assert_eq!(detections.len(), 1);
        let det = &detections[0];

        // Expected center: (grid_j + dx) * stride = (10 + 0.5) * 32 = 336
        //                  (grid_i + dy) * stride = (10 + 0.3) * 32 = 329.6
        // Normalized: cx = 336/640 = 0.525, cy = 329.6/640 = 0.515
        // Expected size: dw * stride = 4 * 32 = 128, w_norm = 128/640 = 0.2
        // x = cx - w/2 = 0.525 - 0.1 = 0.425
        // y = cy - h/2 = 0.515 - 0.1 = 0.415

        assert!((det.bbox[0] - 0.425).abs() < 1e-5); // x
        assert!((det.bbox[1] - 0.415).abs() < 1e-5); // y
        assert!((det.bbox[2] - 0.2).abs() < 1e-5); // w
        assert!((det.bbox[3] - 0.2).abs() < 1e-5); // h
        assert!((det.score - 0.9).abs() < 1e-5);

        // Landmark at center: (10 + 0) * 32 / 640 = 0.5
        assert!((det.landmarks[0] - 0.5).abs() < 1e-5);
        assert!((det.landmarks[1] - 0.5).abs() < 1e-5);
    }
}
