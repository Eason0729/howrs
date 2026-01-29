use anyhow::Result;
use howrs_vision::{face, model};
use std::path::Path;

/// Test that centered faces are properly detected with bbox containing image center
#[test]
fn test_centered_faces_detection() -> Result<()> {
    // Load detector
    let mut detector = model::detector_session()?;

    // Images that should have centered faces per README
    let centered_images = [
        "test_faces/ir-cam/eason1.png",
        "test_faces/ir-cam/eason2.png",
        "test_faces/ir-cam/eason3.png",
        "test_faces/ir-cam/eason4.png",
        "test_faces/ir-cam/ling1.png",
        "test_faces/ir-cam/ling2.png",
    ];

    for img_path in &centered_images {
        println!("\n=== Testing {} ===", img_path);
        let img = image::open(img_path)?;
        let (width, height) = (img.width(), img.height());
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;
        println!(
            "Image size: {}x{}, center: ({}, {})",
            width, height, center_x, center_y
        );

        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;
        assert!(!detections.is_empty(), "No face detected in {}", img_path);

        let det = &detections[0];
        let bbox_left = det.bbox[0];
        let bbox_top = det.bbox[1];
        let bbox_right = det.bbox[0] + det.bbox[2];
        let bbox_bottom = det.bbox[1] + det.bbox[3];

        println!(
            "  Bbox: [{}, {}, {}, {}] (left, top, right, bottom)",
            bbox_left, bbox_top, bbox_right, bbox_bottom
        );
        println!("  Bbox size: {}x{}", det.bbox[2], det.bbox[3]);

        // Check if center point is inside bbox
        let contains_center = center_x >= bbox_left
            && center_x <= bbox_right
            && center_y >= bbox_top
            && center_y <= bbox_bottom;

        println!("  Contains center: {}", contains_center);

        // Calculate how far bbox center is from image center
        let bbox_center_x = bbox_left + det.bbox[2] / 2.0;
        let bbox_center_y = bbox_top + det.bbox[3] / 2.0;
        let offset_x = (bbox_center_x - center_x).abs();
        let offset_y = (bbox_center_y - center_y).abs();
        println!(
            "  Bbox center offset from image center: ({:.1}, {:.1})",
            offset_x, offset_y
        );

        // Print landmarks
        println!("  Landmarks:");
        for i in 0..5 {
            let x = det.landmarks[i * 2];
            let y = det.landmarks[i * 2 + 1];
            let label = match i {
                0 => "left eye",
                1 => "right eye",
                2 => "nose",
                3 => "left mouth",
                4 => "right mouth",
                _ => "unknown",
            };
            println!("    {}: ({:.1}, {:.1})", label, x, y);
        }

        assert!(contains_center, "{}: Face bbox should contain image center. Bbox: [{:.1}, {:.1}, {:.1}, {:.1}], Center: ({:.1}, {:.1})", img_path, bbox_left, bbox_top, bbox_right, bbox_bottom, center_x, center_y);
    }

    Ok(())
}

/// Test cropped face quality by saving crops for inspection
#[test]
fn test_save_cropped_faces() -> Result<()> {
    // Load detector
    let mut detector = model::detector_session()?;

    let test_images = [
        "test_faces/ir-cam/eason1.png",
        "test_faces/ir-cam/eason2.png",
        "test_faces/ir-cam/ling1.png",
        "test_faces/ir-cam/ling2.png",
    ];

    std::fs::create_dir_all("target/debug_crops")?;

    for img_path in &test_images {
        println!("\nProcessing {}", img_path);
        let img = image::open(img_path)?;
        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            println!("  No face detected!");
            continue;
        }

        let det = &detections[0];
        println!(
            "  Detected face at: [{:.1}, {:.1}, {:.1}, {:.1}]",
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
        );

        // Crop and save
        let cropped = face::align_face(&img, det, 112)?;
        let filename = Path::new(img_path).file_stem().unwrap().to_str().unwrap();
        let out_path = format!("target/debug_crops/{}_crop.png", filename);
        cropped.save(&out_path)?;
        println!("  Saved crop to: {}", out_path);

        // Check if crop is not mostly black
        let rgb_img = cropped.to_rgb8();
        let total_brightness: u64 = rgb_img
            .pixels()
            .map(|p| p[0] as u64 + p[1] as u64 + p[2] as u64)
            .sum();
        let num_pixels = (112 * 112) as u64;
        let avg_brightness = total_brightness / (num_pixels * 3);

        println!(
            "  Average brightness: {} (should be > 20 for visible face)",
            avg_brightness
        );

        assert!(
            avg_brightness > 20,
            "{}: Cropped face is too dark (avg brightness: {}), likely wrong region cropped",
            img_path,
            avg_brightness
        );
    }

    println!("\n✓ Cropped faces saved to target/debug_crops/");
    Ok(())
}
