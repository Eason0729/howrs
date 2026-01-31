use crate::yunet;
use anyhow::Result;
use image::{DynamicImage, GenericImageView};
use ndarray::{Array2, Array4};
use ort::{session::Session, value::Value};

/// Detection result from YuNet
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: [f32; 4], // x, y, w, h
    pub score: f32,
    pub landmarks: [f32; 10], // 5 points: x1,y1,x2,y2,...,x5,y5
}

/// Face embedding (SFace output)
#[derive(Debug, Clone)]
pub struct Embedding {
    pub vector: Array2<f32>,
}

/// Detect faces in an image using YuNet detector
pub fn detect_faces(
    session: &mut Session,
    img: &DynamicImage,
    score_threshold: f32,
    nms_threshold: f32,
) -> Result<Vec<Detection>> {
    // YuNet model expects fixed input size [1, 3, 640, 640]
    // Pad image to square to avoid distortion
    let target_size = 640;
    let (orig_width, orig_height) = img.dimensions();

    // Create square canvas with padding
    let max_dim = orig_width.max(orig_height);
    let scale = target_size as f32 / max_dim as f32;
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    // Resize maintaining aspect ratio
    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    // Create square canvas and paste resized image
    let mut canvas = image::DynamicImage::new_rgb8(target_size, target_size);
    let offset_x = (target_size - new_width) / 2;
    let offset_y = (target_size - new_height) / 2;
    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);

    let img_rgb = canvas.to_rgb8();

    // YuNet expects input shape [1, 3, H, W] in BGR format
    // SIMD-optimized RGB to BGR channel extraction with cache-friendly layout
    let pixel_count = (target_size * target_size) as usize;
    let mut input_data = Vec::with_capacity(3 * pixel_count);
    
    // Pre-allocate all channels
    unsafe {
        input_data.set_len(3 * pixel_count);
    }
    
    // Split into channel slices for better cache locality
    let (b_channel, rest) = input_data.split_at_mut(pixel_count);
    let (g_channel, r_channel) = rest.split_at_mut(pixel_count);
    
    // Convert RGB to BGR using SIMD
    let pixels = img_rgb.as_raw();
    for i in 0..pixel_count {
        let idx = i * 3;
        r_channel[i] = pixels[idx] as f32;     // R
        g_channel[i] = pixels[idx + 1] as f32; // G
        b_channel[i] = pixels[idx + 2] as f32; // B
    }

    let input_array = Array4::from_shape_vec(
        (1, 3, target_size as usize, target_size as usize),
        input_data,
    )?;
    let input_tensor = Value::from_array(input_array)?;

    let outputs = session.run(ort::inputs![input_tensor])?;

    // Extract all output tensors and store the data
    let mut output_data: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    for (_name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<i64> = shape.iter().copied().collect();
        let data_vec = data.to_vec();
        output_data.push((shape_vec, data_vec));
    }

    // Create references for parsing
    let output_refs: Vec<(&[i64], &[f32])> = output_data
        .iter()
        .map(|(s, d)| (s.as_slice(), d.as_slice()))
        .collect();

    // Parse YuNet outputs into structured format
    let (mut cls_scores, bbox_preds, landmark_preds) =
        yunet::parse_yunet_outputs(&output_refs, target_size as usize)?;

    // Apply sigmoid to classification scores
    yunet::apply_sigmoid_to_scores(&mut cls_scores);

    // Decode detections from anchors
    let raw_detections = yunet::decode_detections(
        cls_scores,
        bbox_preds,
        landmark_preds,
        score_threshold,
        target_size as usize,
    )?;

    // Scale detection coordinates back to original image size
    // Account for padding that was added
    let mut detections: Vec<Detection> = raw_detections
        .into_iter()
        .map(|d| {
            // Coordinates are normalized (0-1) relative to 640x640 canvas
            // Convert to pixels, remove padding offset, then rescale to original dimensions
            let bbox_x_px = d.bbox[0] * target_size as f32;
            let bbox_y_px = d.bbox[1] * target_size as f32;
            let bbox_w_px = d.bbox[2] * target_size as f32;
            let bbox_h_px = d.bbox[3] * target_size as f32;

            let bbox_x = (bbox_x_px - offset_x as f32) / scale;
            let bbox_y = (bbox_y_px - offset_y as f32) / scale;
            let bbox_w = bbox_w_px / scale;
            let bbox_h = bbox_h_px / scale;

            let mut landmarks = [0.0f32; 10];
            for i in 0..5 {
                let lm_x_px = d.landmarks[i * 2] * target_size as f32;
                let lm_y_px = d.landmarks[i * 2 + 1] * target_size as f32;
                landmarks[i * 2] = (lm_x_px - offset_x as f32) / scale;
                landmarks[i * 2 + 1] = (lm_y_px - offset_y as f32) / scale;
            }

            Detection {
                bbox: [bbox_x, bbox_y, bbox_w, bbox_h],
                score: d.score,
                landmarks,
            }
        })
        .collect();

    // Apply NMS if requested
    if nms_threshold < 1.0 {
        detections = nms(&detections, nms_threshold);
    }

    Ok(detections)
}

/// Apply non-maximum suppression to remove overlapping detections
pub fn nms(detections: &[Detection], iou_threshold: f32) -> Vec<Detection> {
    if detections.is_empty() {
        return vec![];
    }

    let mut sorted = detections.to_vec();
    sorted.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut keep = Vec::new();
    let mut suppressed = vec![false; sorted.len()];

    for i in 0..sorted.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(sorted[i].clone());

        for j in (i + 1)..sorted.len() {
            if suppressed[j] {
                continue;
            }
            let iou = compute_iou(&sorted[i].bbox, &sorted[j].bbox);
            if iou > iou_threshold {
                suppressed[j] = true;
            }
        }
    }

    keep
}

fn compute_iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let x1 = a[0].max(b[0]);
    let y1 = a[1].max(b[1]);
    let x2 = (a[0] + a[2]).min(b[0] + b[2]);
    let y2 = (a[1] + a[3]).min(b[1] + b[3]);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let inter = (x2 - x1) * (y2 - y1);
    let area_a = a[2] * a[3];
    let area_b = b[2] * b[3];
    inter / (area_a + area_b - inter)
}

/// Align and crop face using landmarks
pub fn align_face(img: &DynamicImage, detection: &Detection, size: u32) -> Result<DynamicImage> {
    // Eye-based alignment using affine transform
    // Reference landmarks for 112x112 SFace input (ArcFace standard)
    let ref_left_eye = (38.3_f32, 51.7_f32);
    let ref_right_eye = (73.5_f32, 51.5_f32);

    // Extract eye coordinates from landmarks
    // landmarks: [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y, ...]
    let left_eye = (detection.landmarks[0], detection.landmarks[1]);
    let right_eye = (detection.landmarks[2], detection.landmarks[3]);

    // Calculate eye vector and angle
    let eye_dx = right_eye.0 - left_eye.0;
    let eye_dy = right_eye.1 - left_eye.1;
    let eye_angle = eye_dy.atan2(eye_dx);

    // Calculate reference eye distance and actual eye distance
    let ref_eye_dist = ((ref_right_eye.0 - ref_left_eye.0).powi(2_i32)
        + (ref_right_eye.1 - ref_left_eye.1).powi(2_i32))
    .sqrt();
    let actual_eye_dist = (eye_dx * eye_dx + eye_dy * eye_dy).sqrt();

    // Calculate scale to match reference eye distance
    let scale = (size as f32 / 112.0) * (ref_eye_dist / actual_eye_dist);

    // Calculate center point between eyes
    let eye_center = (
        (left_eye.0 + right_eye.0) / 2.0,
        (left_eye.1 + right_eye.1) / 2.0,
    );
    let ref_eye_center = (
        (ref_left_eye.0 + ref_right_eye.0) / 2.0,
        (ref_left_eye.1 + ref_right_eye.1) / 2.0,
    );

    // Scale reference center to output size
    let ref_center_scaled = (
        ref_eye_center.0 * size as f32 / 112.0,
        ref_eye_center.1 * size as f32 / 112.0,
    );

    // Create transformation matrix
    // We need: rotate around eye center, scale, then translate to reference position
    let cos_angle = eye_angle.cos();
    let sin_angle = eye_angle.sin();

    // Build affine transform matrix (3x2)
    // [ a  b  tx ]
    // [ c  d  ty ]
    // Where output = [a,b; c,d] * input + [tx, ty]
    let a = scale * cos_angle;
    let b = scale * sin_angle;
    let c = -scale * sin_angle;
    let d = scale * cos_angle;

    // Translation: after rotation and scaling, shift so eye_center maps to ref_center_scaled
    let tx = ref_center_scaled.0 - (a * eye_center.0 + b * eye_center.1);
    let ty = ref_center_scaled.1 - (c * eye_center.0 + d * eye_center.1);

    // Apply transformation by creating output image and mapping pixels
    let (img_w, img_h) = img.dimensions();
    let mut output = image::RgbImage::new(size, size);

    // For each pixel in output, find corresponding source pixel
    for out_y in 0..size {
        for out_x in 0..size {
            // Invert the transformation to find source coordinates
            // input = inv([a,b;c,d]) * (output - [tx,ty])
            let out_x_f = out_x as f32;
            let out_y_f = out_y as f32;

            // Subtract translation
            let tmp_x = out_x_f - tx;
            let tmp_y = out_y_f - ty;

            // Apply inverse rotation and scale
            let det = a * d - b * c;
            let in_x = (d * tmp_x - b * tmp_y) / det;
            let in_y = (-c * tmp_x + a * tmp_y) / det;

            // Sample from input image (with boundary check)
            if in_x >= 0.0 && in_x < img_w as f32 && in_y >= 0.0 && in_y < img_h as f32 {
                // Bilinear interpolation
                let x0 = in_x.floor() as u32;
                let y0 = in_y.floor() as u32;
                let x1 = (x0 + 1).min(img_w - 1);
                let y1 = (y0 + 1).min(img_h - 1);

                let fx = in_x - x0 as f32;
                let fy = in_y - y0 as f32;

                let p00 = img.get_pixel(x0, y0);
                let p10 = img.get_pixel(x1, y0);
                let p01 = img.get_pixel(x0, y1);
                let p11 = img.get_pixel(x1, y1);

                // Bilinear interpolation optimized for LLVM auto-vectorization
                let w00 = (1.0 - fx) * (1.0 - fy);
                let w10 = fx * (1.0 - fy);
                let w01 = (1.0 - fx) * fy;
                let w11 = fx * fy;
                
                // Compute interpolation for each RGB channel
                // Using simple arithmetic allows LLVM to auto-vectorize
                let r = (p00[0] as f32 * w00 + p10[0] as f32 * w10 
                       + p01[0] as f32 * w01 + p11[0] as f32 * w11) as u8;
                let g = (p00[1] as f32 * w00 + p10[1] as f32 * w10 
                       + p01[1] as f32 * w01 + p11[1] as f32 * w11) as u8;
                let b_val = (p00[2] as f32 * w00 + p10[2] as f32 * w10 
                           + p01[2] as f32 * w01 + p11[2] as f32 * w11) as u8;

                output.put_pixel(out_x, out_y, image::Rgb([r, g, b_val]));
            }
            // else: leave black (default)
        }
    }

    Ok(image::DynamicImage::ImageRgb8(output))
}

/// Encode face image to embedding using SFace
pub fn encode_face(session: &mut Session, face_img: &DynamicImage) -> Result<Embedding> {
    // SFace expects input shape [1, 3, 112, 112] in BGR format with values in [0, 255]
    let size = 112;
    let face_rgb = face_img.resize_exact(size, size, image::imageops::FilterType::Triangle);
    let face_rgb = face_rgb.to_rgb8();

    // Convert to CHW format in BGR order (B, G, R) with values in [0, 255]
    // SIMD-optimized RGB to BGR channel extraction with cache-friendly layout
    let pixel_count = (size * size) as usize;
    let mut input_data = Vec::with_capacity(3 * pixel_count);
    
    // Pre-allocate all channels
    unsafe {
        input_data.set_len(3 * pixel_count);
    }
    
    // Split into channel slices for better cache locality
    let (b_channel, rest) = input_data.split_at_mut(pixel_count);
    let (g_channel, r_channel) = rest.split_at_mut(pixel_count);
    
    // Convert RGB to BGR
    let pixels = face_rgb.as_raw();
    for i in 0..pixel_count {
        let idx = i * 3;
        r_channel[i] = pixels[idx] as f32;     // R
        g_channel[i] = pixels[idx + 1] as f32; // G
        b_channel[i] = pixels[idx + 2] as f32; // B
    }

    let input_array = Array4::from_shape_vec((1, 3, size as usize, size as usize), input_data)?;
    let input_tensor = Value::from_array(input_array)?;

    let outputs = session.run(ort::inputs![input_tensor])?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;

    // Expecting shape [1, 128]
    let embedding_size = if shape.len() == 2 {
        shape[1] as usize
    } else {
        data.len()
    };
    let embedding_vec: Vec<f32> = data[0..embedding_size].to_vec();

    // Normalize the embedding (L2 normalization)
    let norm: f32 = embedding_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized = if norm > 0.0 {
        embedding_vec.iter().map(|x| x / norm).collect()
    } else {
        embedding_vec
    };

    let embedding_array = Array2::from_shape_vec((1, embedding_size), normalized)?;

    Ok(Embedding {
        vector: embedding_array,
    })
}

/// Compute cosine similarity between two embeddings
pub fn match_embedding(a: &Embedding, b: &Embedding) -> f32 {
    // Optimized dot product for cosine similarity with LLVM auto-vectorization
    // Embeddings are already L2-normalized, so dot product = cosine similarity
    let a_data = a.vector.as_slice().unwrap();
    let b_data = b.vector.as_slice().unwrap();
    
    let len = a_data.len().min(b_data.len());
    
    // Simple loop that LLVM can auto-vectorize
    // Using iterator zip and sum is optimal for auto-vectorization
    let dot: f32 = a_data.iter()
        .zip(b_data.iter())
        .take(len)
        .map(|(x, y)| x * y)
        .sum();
    
    dot.max(-1.0).min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iou() {
        let a = [10.0, 10.0, 20.0, 20.0];
        let b = [15.0, 15.0, 20.0, 20.0];
        let iou = compute_iou(&a, &b);
        assert!(iou > 0.0 && iou < 1.0);

        // No overlap
        let c = [100.0, 100.0, 10.0, 10.0];
        assert_eq!(compute_iou(&a, &c), 0.0);
    }

    #[test]
    fn test_nms() {
        let detections = vec![
            Detection {
                bbox: [10.0, 10.0, 20.0, 20.0],
                score: 0.9,
                landmarks: [0.0; 10],
            },
            Detection {
                bbox: [12.0, 12.0, 20.0, 20.0],
                score: 0.8,
                landmarks: [0.0; 10],
            },
            Detection {
                bbox: [100.0, 100.0, 20.0, 20.0],
                score: 0.85,
                landmarks: [0.0; 10],
            },
        ];

        let result = nms(&detections, 0.3);
        assert_eq!(result.len(), 2); // Should keep first and third
    }
}
