use anyhow::Result;
/// Debug test to examine YuNet's landmark decoding in detail
/// Check if the formula and variance values are correct
use image::GenericImageView;
use ort::session::Session;

#[test]
#[ignore]
fn debug_landmark_decoding() -> Result<()> {
    // Load test image
    let img_path = "test_faces/ir-cam/eason1.png";
    let img = image::open(img_path)?;
    let (orig_width, orig_height) = img.dimensions();
    println!("Original image: {}x{}", orig_width, orig_height);

    // Setup YuNet detector (copy logic from detect_faces)
    let model_path = "models/face_detection_yunet_2023mar.onnx";
    let mut session = Session::builder()?.commit_from_file(model_path)?;

    let target_size = 640;
    let max_dim = orig_width.max(orig_height);
    let scale = target_size as f32 / max_dim as f32;
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    println!(
        "Scale: {}, New dimensions: {}x{}",
        scale, new_width, new_height
    );

    // Resize maintaining aspect ratio
    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    // Create square canvas and paste resized image
    let mut canvas = image::DynamicImage::new_rgb8(target_size, target_size);
    let offset_x = (target_size - new_width) / 2;
    let offset_y = (target_size - new_height) / 2;
    println!("Padding offsets: x={}, y={}", offset_x, offset_y);

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);
    let img_rgb = canvas.to_rgb8();

    // Prepare input
    let mut input_data = Vec::with_capacity((3 * target_size * target_size) as usize);
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[2] as f32);
    }
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[1] as f32);
    }
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[0] as f32);
    }

    let input_array = ndarray::Array4::from_shape_vec(
        (1, 3, target_size as usize, target_size as usize),
        input_data,
    )?;
    let input_tensor = ort::value::Value::from_array(input_array)?;

    // Run inference
    let outputs = session.run(ort::inputs![input_tensor])?;

    // Extract raw output data
    let mut output_data: Vec<(Vec<i64>, Vec<f32>)> = Vec::new();
    for (name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<i64> = shape.iter().copied().collect();
        let data_vec = data.to_vec();
        println!(
            "Output '{}': shape {:?}, first 5 values: {:?}",
            name,
            shape_vec,
            &data_vec[..5.min(data_vec.len())]
        );
        output_data.push((shape_vec, data_vec));
    }

    // Parse outputs using the library function
    let output_refs: Vec<(&[i64], &[f32])> = output_data
        .iter()
        .map(|(s, d)| (s.as_slice(), d.as_slice()))
        .collect();

    let (mut cls_scores, bbox_preds, landmark_preds) =
        howrs_vision::yunet::parse_yunet_outputs(&output_refs, target_size as usize)?;

    howrs_vision::yunet::apply_sigmoid_to_scores(&mut cls_scores);

    // Find the best detection manually to examine raw values
    let mut best_idx = (0, 0, 0.0f32); // (scale_idx, box_idx, score)
    for (scale_idx, scores) in cls_scores.iter().enumerate() {
        for i in 0..scores.shape()[0] {
            let score = scores[[i, 0]];
            if score > best_idx.2 {
                best_idx = (scale_idx, i, score);
            }
        }
    }

    println!("\n=== Best Detection ===");
    println!(
        "Scale index: {}, Box index: {}, Score: {:.4}",
        best_idx.0, best_idx.1, best_idx.2
    );

    // Get raw values for this detection
    let scale_idx = best_idx.0;
    let box_idx = best_idx.1;

    println!("\n=== Raw Network Outputs ===");
    let bbox_raw = &bbox_preds[scale_idx];
    println!(
        "BBox deltas [dx, dy, dw, dh]: [{:.4}, {:.4}, {:.4}, {:.4}]",
        bbox_raw[[box_idx, 0]],
        bbox_raw[[box_idx, 1]],
        bbox_raw[[box_idx, 2]],
        bbox_raw[[box_idx, 3]]
    );

    let lm_raw = &landmark_preds[scale_idx];
    println!("Landmark deltas (raw from network):");
    for i in 0..5 {
        println!(
            "  Point {}: dx={:.4}, dy={:.4}",
            i,
            lm_raw[[box_idx, i * 2]],
            lm_raw[[box_idx, i * 2 + 1]]
        );
    }

    // Generate priors and get the specific prior for this detection
    const STRIDES: [usize; 3] = [8, 16, 32];
    const MIN_SIZES: [[f32; 2]; 3] = [[10.0, 16.0], [32.0, 64.0], [128.0, 256.0]];

    let stride = STRIDES[scale_idx];
    let feature_size = (target_size as usize) / stride;
    let min_size = MIN_SIZES[scale_idx][0];

    // Calculate which cell this box is in
    let cell_y = box_idx / feature_size;
    let cell_x = box_idx % feature_size;

    let cx_prior = (cell_x as f32 + 0.5) * stride as f32 / target_size as f32;
    let cy_prior = (cell_y as f32 + 0.5) * stride as f32 / target_size as f32;
    let s_kx = min_size / target_size as f32;
    let s_ky = min_size / target_size as f32;

    println!("\n=== Prior Anchor ===");
    println!(
        "Cell: [{}, {}] in {}x{} grid (stride {})",
        cell_x, cell_y, feature_size, feature_size, stride
    );
    println!(
        "Prior [cx, cy, w, h]: [{:.4}, {:.4}, {:.4}, {:.4}]",
        cx_prior, cy_prior, s_kx, s_ky
    );
    println!(
        "Prior in pixels: [{:.1}, {:.1}, {:.1}, {:.1}]",
        cx_prior * 640.0,
        cy_prior * 640.0,
        s_kx * 640.0,
        s_ky * 640.0
    );

    // Decode using YuNet formula
    const VARIANCE: [f32; 2] = [0.1, 0.2];
    const LANDMARK_VARIANCE: f32 = 0.25;

    println!(
        "\n=== Decoding with VARIANCE [{}, {}], LANDMARK_VARIANCE {} ===",
        VARIANCE[0], VARIANCE[1], LANDMARK_VARIANCE
    );

    // Decode bbox
    let dx = bbox_raw[[box_idx, 0]];
    let dy = bbox_raw[[box_idx, 1]];
    let dw = bbox_raw[[box_idx, 2]];
    let dh = bbox_raw[[box_idx, 3]];

    let cx_decoded = cx_prior + dx * VARIANCE[0] * s_kx;
    let cy_decoded = cy_prior + dy * VARIANCE[0] * s_ky;
    let w_decoded = s_kx * (dw * VARIANCE[1]).exp();
    let h_decoded = s_ky * (dh * VARIANCE[1]).exp();

    println!(
        "Decoded bbox center [cx, cy]: [{:.4}, {:.4}] -> pixels: [{:.1}, {:.1}]",
        cx_decoded,
        cy_decoded,
        cx_decoded * 640.0,
        cy_decoded * 640.0
    );
    println!(
        "Decoded bbox size [w, h]: [{:.4}, {:.4}] -> pixels: [{:.1}, {:.1}]",
        w_decoded,
        h_decoded,
        w_decoded * 640.0,
        h_decoded * 640.0
    );

    // Decode landmarks - EXAMINE THE FORMULA CAREFULLY
    println!("\n=== Landmark Decoding ===");
    println!("Using formula: lm_x = prior_cx + delta_x * LANDMARK_VARIANCE * prior_w");
    println!("               lm_y = prior_cy + delta_y * LANDMARK_VARIANCE * prior_h");

    for i in 0..5 {
        let dx_lm = lm_raw[[box_idx, i * 2]];
        let dy_lm = lm_raw[[box_idx, i * 2 + 1]];

        let x_decoded = cx_prior + dx_lm * LANDMARK_VARIANCE * s_kx;
        let y_decoded = cy_prior + dy_lm * LANDMARK_VARIANCE * s_ky;

        println!("Point {} (delta: [{:.4}, {:.4}]):", i, dx_lm, dy_lm);
        println!(
            "  Decoded: [{:.4}, {:.4}] (normalized)",
            x_decoded, y_decoded
        );
        println!(
            "  Pixels on 640x640 canvas: [{:.1}, {:.1}]",
            x_decoded * 640.0,
            y_decoded * 640.0
        );
        println!(
            "  After removing padding and scaling to {}x{}:",
            orig_width, orig_height
        );
        let x_final = (x_decoded * 640.0 - offset_x as f32) / scale;
        let y_final = (y_decoded * 640.0 - offset_y as f32) / scale;
        println!("    [{:.1}, {:.1}]", x_final, y_final);
    }

    // Calculate eye distance
    let left_eye_x = cx_prior + lm_raw[[box_idx, 0]] * LANDMARK_VARIANCE * s_kx;
    let left_eye_y = cy_prior + lm_raw[[box_idx, 1]] * LANDMARK_VARIANCE * s_ky;
    let right_eye_x = cx_prior + lm_raw[[box_idx, 2]] * LANDMARK_VARIANCE * s_kx;
    let right_eye_y = cy_prior + lm_raw[[box_idx, 3]] * LANDMARK_VARIANCE * s_ky;

    let eye_dist_norm =
        ((right_eye_x - left_eye_x).powi(2) + (right_eye_y - left_eye_y).powi(2)).sqrt();
    let eye_dist_px = eye_dist_norm * 640.0;
    let eye_dist_final = eye_dist_px / scale;

    println!("\n=== Eye Distance Analysis ===");
    println!("Left eye: [{:.4}, {:.4}]", left_eye_x, left_eye_y);
    println!("Right eye: [{:.4}, {:.4}]", right_eye_x, right_eye_y);
    println!("Distance (normalized): {:.4}", eye_dist_norm);
    println!("Distance (640x640 pixels): {:.1} px", eye_dist_px);
    println!("Distance (original image): {:.1} px", eye_dist_final);
    println!(
        "Face bbox width (original): {:.1} px",
        w_decoded * 640.0 / scale
    );
    println!(
        "Eye distance / bbox width: {:.2}%",
        (eye_dist_px / (w_decoded * 640.0)) * 100.0
    );

    Ok(())
}
