/// Integration test to verify anchor-free YuNet decoding produces correct results
use anyhow::Result;
use image::GenericImageView;
use ort::session::Session;

#[test]
fn test_anchor_free_yunet_decoding() -> Result<()> {
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

    // Resize maintaining aspect ratio
    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    // Create square canvas and paste resized image
    let mut canvas = image::DynamicImage::new_rgb8(target_size, target_size);
    let offset_x = (target_size - new_width) / 2;
    let offset_y = (target_size - new_height) / 2;
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
    for (_name, output) in outputs.iter() {
        let (shape, data) = output.try_extract_tensor::<f32>()?;
        let shape_vec: Vec<i64> = shape.iter().copied().collect();
        let data_vec = data.to_vec();
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

    // Decode detections
    let detections = howrs_vision::yunet::decode_detections(
        cls_scores,
        bbox_preds,
        landmark_preds,
        0.5, // score threshold
        target_size as usize,
    )?;

    println!("Found {} detections", detections.len());

    // We should find at least one face
    assert!(
        !detections.is_empty(),
        "Should detect at least one face in the test image"
    );

    // Check the best detection
    let best_det = detections
        .iter()
        .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap())
        .unwrap();

    println!("Best detection score: {:.4}", best_det.score);
    println!(
        "BBox (normalized): x={:.4}, y={:.4}, w={:.4}, h={:.4}",
        best_det.bbox[0], best_det.bbox[1], best_det.bbox[2], best_det.bbox[3]
    );

    // Score should be reasonably high for a clear face image
    assert!(
        best_det.score > 0.5,
        "Detection score should be > 0.5, got {}",
        best_det.score
    );

    // BBox should be reasonable (within image bounds after normalization)
    assert!(
        best_det.bbox[0] >= 0.0 && best_det.bbox[0] <= 1.0,
        "BBox x should be in [0,1]"
    );
    assert!(
        best_det.bbox[1] >= 0.0 && best_det.bbox[1] <= 1.0,
        "BBox y should be in [0,1]"
    );
    assert!(
        best_det.bbox[2] > 0.0 && best_det.bbox[2] <= 1.0,
        "BBox width should be in (0,1]"
    );
    assert!(
        best_det.bbox[3] > 0.0 && best_det.bbox[3] <= 1.0,
        "BBox height should be in (0,1]"
    );

    // Check landmarks are within reasonable bounds
    for i in 0..5 {
        let lm_x = best_det.landmarks[i * 2];
        let lm_y = best_det.landmarks[i * 2 + 1];
        println!("Landmark {}: x={:.4}, y={:.4}", i, lm_x, lm_y);

        assert!(
            lm_x >= 0.0 && lm_x <= 1.0,
            "Landmark {} x should be in [0,1], got {}",
            i,
            lm_x
        );
        assert!(
            lm_y >= 0.0 && lm_y <= 1.0,
            "Landmark {} y should be in [0,1], got {}",
            i,
            lm_y
        );
    }

    // Verify landmarks are roughly within the bounding box (with some tolerance)
    let bbox_x1 = best_det.bbox[0];
    let bbox_y1 = best_det.bbox[1];
    let bbox_x2 = bbox_x1 + best_det.bbox[2];
    let bbox_y2 = bbox_y1 + best_det.bbox[3];

    for i in 0..5 {
        let lm_x = best_det.landmarks[i * 2];
        let lm_y = best_det.landmarks[i * 2 + 1];

        // Landmarks can be slightly outside bbox, but not too far
        let margin = 0.1; // 10% margin
        assert!(
            lm_x >= bbox_x1 - margin && lm_x <= bbox_x2 + margin,
            "Landmark {} x should be near bbox",
            i
        );
        assert!(
            lm_y >= bbox_y1 - margin && lm_y <= bbox_y2 + margin,
            "Landmark {} y should be near bbox",
            i
        );
    }

    println!("✓ Anchor-free decoding test passed!");
    Ok(())
}
