use anyhow::Result;
use howrs_vision::{face, model};
use image::{DynamicImage, GenericImageView, Rgb, RgbImage};
use std::path::Path;

#[test]
#[ignore = "visualization"]
fn visualize_alignment_process() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;

    let img_path = "test_faces/ir-cam/eason3.png";
    if !Path::new(img_path).exists() {
        println!("Test image not found: {}", img_path);
        return Ok(());
    }

    let img = image::open(img_path)?;
    let (w, h) = img.dimensions();
    println!("Original image: {}x{}", w, h);

    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;
    if detections.is_empty() {
        println!("No faces detected!");
        return Ok(());
    }

    let det = &detections[0];
    println!("\nDetection:");
    println!(
        "  BBox: [{:.1}, {:.1}, {:.1}, {:.1}]",
        det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
    );

    // Draw detection on original image
    let mut vis_img = img.to_rgb8();

    // Draw landmarks
    for i in 0..5 {
        let lm_x = det.landmarks[i * 2] as u32;
        let lm_y = det.landmarks[i * 2 + 1] as u32;
        // Draw a small cross at landmark position
        let color = Rgb([255, 0, 0]);
        for dx in 0..5i32 {
            for dy in 0..5i32 {
                let px = (lm_x as i32 + dx - 2).max(0) as u32;
                let py = (lm_y as i32 + dy - 2).max(0) as u32;
                if px < vis_img.width() && py < vis_img.height() {
                    if dx == 2 || dy == 2 {
                        vis_img.put_pixel(px, py, color);
                    }
                }
            }
        }
        println!("  Landmark {}: ({}, {})", i, lm_x, lm_y);
    }

    let output_name = "vis_original_with_detection.png";
    DynamicImage::ImageRgb8(vis_img).save(output_name)?;
    println!("\nSaved visualization: {}", output_name);

    // Now get aligned face and visualize reference landmarks on it
    let aligned = face::align_face(&img, det, 112)?;
    let mut aligned_rgb = aligned.to_rgb8();

    // Draw reference landmark positions
    let ref_landmarks = [
        (38.2946, 51.6963), // left eye
        (73.5318, 51.5014), // right eye
        (56.0252, 71.7366), // nose
        (41.5493, 92.3655), // left mouth
        (70.7299, 92.2041), // right mouth
    ];

    for &(x, y) in &ref_landmarks {
        let color = Rgb([255, 255, 0]); // yellow
                                        // Draw a cross
        for d in 0..7i32 {
            let px = (x as i32 + d - 3).max(0) as u32;
            let py = y as u32;
            if px < 112 {
                aligned_rgb.put_pixel(px, py, color);
            }
            let px = x as u32;
            let py = (y as i32 + d - 3).max(0) as u32;
            if py < 112 {
                aligned_rgb.put_pixel(px, py, color);
            }
        }
    }

    let output_aligned = "vis_aligned_with_reference.png";
    DynamicImage::ImageRgb8(aligned_rgb).save(output_aligned)?;
    println!("Saved aligned visualization: {}", output_aligned);
    println!("\nReference landmarks (where features SHOULD be):");
    println!("  Eyes at y≈51 (46% from top)");
    println!("  Mouth at y≈92 (82% from top)");

    Ok(())
}
