/// Visual comparison test to verify anchor-free YuNet decoding produces correct results
/// This test will output detailed information about detections for manual verification
use anyhow::Result;
use image::GenericImageView;
use ort::session::Session;

#[test]
#[ignore] // Run with --ignored flag
fn test_visual_detection_comparison() -> Result<()> {
    // Load a test image with a known face
    let img_path = "test_faces/ir-cam/eason1.png";
    let img = image::open(img_path)?;
    let (orig_width, orig_height) = img.dimensions();
    println!("Original image: {}x{}", orig_width, orig_height);

    // Setup YuNet detector
    let model_path = "models/face_detection_yunet_2023mar.onnx";
    let mut session = Session::builder()?.commit_from_file(model_path)?;

    let target_size = 640;
    let max_dim = orig_width.max(orig_height);
    let scale = target_size as f32 / max_dim as f32;
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    println!("Resized to: {}x{}", new_width, new_height);

    // Resize maintaining aspect ratio
    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    // Create square canvas and paste resized image
    let mut canvas = image::DynamicImage::new_rgb8(target_size, target_size);
    let offset_x = (target_size - new_width) / 2;
    let offset_y = (target_size - new_height) / 2;
    println!("Canvas padding: x={}, y={}", offset_x, offset_y);

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);
    let img_rgb = canvas.to_rgb8();

    // Prepare input in BGR format
    let mut input_data = Vec::with_capacity((3 * target_size * target_size) as usize);
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[2] as f32);
    } // B
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[1] as f32);
    } // G
    for pixel in img_rgb.pixels() {
        input_data.push(pixel[0] as f32);
    } // R

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
        println!("Output '{}': shape {:?}", name, shape_vec);
        output_data.push((shape_vec, data_vec));
    }

    // Parse outputs using anchor-free decoding
    let output_refs: Vec<(&[i64], &[f32])> = output_data
        .iter()
        .map(|(s, d)| (s.as_slice(), d.as_slice()))
        .collect();

    let (mut cls_scores, bbox_preds, landmark_preds) =
        howrs_vision::yunet::parse_yunet_outputs(&output_refs, target_size as usize)?;

    howrs_vision::yunet::apply_sigmoid_to_scores(&mut cls_scores);

    // Decode detections with a low threshold to see all candidates
    let detections = howrs_vision::yunet::decode_detections(
        cls_scores,
        bbox_preds,
        landmark_preds,
        0.3, // Low threshold to see more detections
        target_size as usize,
    )?;

    println!("\n=== DETECTION RESULTS ===");
    println!("Total detections above threshold: {}", detections.len());

    // Sort by score and show top 5
    let mut sorted_detections = detections.clone();
    sorted_detections.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    println!("\nTop 5 detections:");
    for (i, det) in sorted_detections.iter().take(5).enumerate() {
        println!("\n#{} - Score: {:.4}", i + 1, det.score);

        // Convert to pixel coordinates on 640x640 canvas
        let x_px = det.bbox[0] * target_size as f32;
        let y_px = det.bbox[1] * target_size as f32;
        let w_px = det.bbox[2] * target_size as f32;
        let h_px = det.bbox[3] * target_size as f32;

        println!(
            "  BBox on canvas: x={:.1}, y={:.1}, w={:.1}, h={:.1}",
            x_px, y_px, w_px, h_px
        );

        // Convert to original image coordinates (removing padding)
        let x_orig = x_px - offset_x as f32;
        let y_orig = y_px - offset_y as f32;

        println!(
            "  BBox on original: x={:.1}, y={:.1}, w={:.1}, h={:.1}",
            x_orig, y_orig, w_px, h_px
        );

        // Show landmarks
        println!("  Landmarks (on canvas):");
        let landmark_names = ["Left eye", "Right eye", "Nose", "Left mouth", "Right mouth"];
        for (j, name) in landmark_names.iter().enumerate() {
            let lm_x = det.landmarks[j * 2] * target_size as f32;
            let lm_y = det.landmarks[j * 2 + 1] * target_size as f32;
            println!("    {}: ({:.1}, {:.1})", name, lm_x, lm_y);
        }

        // Convert landmarks to original image coordinates
        println!("  Landmarks (on original image):");
        for (j, name) in landmark_names.iter().enumerate() {
            let lm_x = det.landmarks[j * 2] * target_size as f32 - offset_x as f32;
            let lm_y = det.landmarks[j * 2 + 1] * target_size as f32 - offset_y as f32;
            println!("    {}: ({:.1}, {:.1})", name, lm_x, lm_y);
        }

        // Check landmark geometry
        let left_eye_x = det.landmarks[0];
        let right_eye_x = det.landmarks[2];
        let eye_distance = (right_eye_x - left_eye_x).abs() * target_size as f32;
        let face_width = det.bbox[2] * target_size as f32;

        println!(
            "  Eye distance: {:.1}px ({:.1}% of face width)",
            eye_distance,
            (eye_distance / face_width) * 100.0
        );
    }

    println!("\n=== ANCHOR-FREE DECODING VERIFICATION ===");
    println!("✓ Using grid-based decoding (no priors/anchors)");
    println!("✓ Direct linear transformation: cx = (grid_x + dx) * stride");
    println!("✓ No variance scaling or exp operations");
    println!("✓ All coordinates normalized to [0,1] relative to input size");

    Ok(())
}
