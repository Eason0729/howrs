//! Test the ACTUAL align_face function and visualize landmarks
use anyhow::Result;
use howrs_vision::{face, model};

#[test]
#[ignore]
fn test_actual_alignment() -> Result<()> {
    let mut detector = model::detector_session()?;

    let img_path = "test_faces/ir-cam/eason1.png";
    let img = image::open(img_path)?;

    println!("Testing: {}", img_path);
    println!("Original size: {}x{}\n", img.width(), img.height());

    let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

    for (i, det) in detections.iter().enumerate() {
        println!("Detection {}:", i);
        println!("  Detected landmarks:");
        println!(
            "    LeftEye: ({:.1}, {:.1})",
            det.landmarks[0], det.landmarks[1]
        );
        println!(
            "    RightEye: ({:.1}, {:.1})",
            det.landmarks[2], det.landmarks[3]
        );

        // Calculate what align_face will do
        let left_eye_x = det.landmarks[0];
        let left_eye_y = det.landmarks[1];
        let right_eye_x = det.landmarks[2];
        let right_eye_y = det.landmarks[3];

        let eye_center_x = (left_eye_x + right_eye_x) / 2.0;
        let eye_center_y = (left_eye_y + right_eye_y) / 2.0;
        let eye_dist =
            ((right_eye_x - left_eye_x).powi(2) + (right_eye_y - left_eye_y).powi(2)).sqrt();

        println!("\n  Eye analysis:");
        println!("    Eye center: ({:.1}, {:.1})", eye_center_x, eye_center_y);
        println!("    Eye distance: {:.1}px", eye_dist);

        let target_eye_dist_ratio = 35.2 / 112.0;
        let target_eye_y_ratio = 51.7 / 112.0;
        let crop_size = eye_dist / target_eye_dist_ratio;

        println!("    Calculated crop size: {:.1}px", crop_size);

        let crop_left = eye_center_x - crop_size / 2.0;
        let crop_top = eye_center_y - crop_size * target_eye_y_ratio;
        let crop_right = crop_left + crop_size;
        let crop_bottom = crop_top + crop_size;

        println!(
            "    Initial crop: ({:.1}, {:.1}) to ({:.1}, {:.1})",
            crop_left, crop_top, crop_right, crop_bottom
        );
        println!(
            "    Crop size: {:.1}x{:.1}",
            crop_right - crop_left,
            crop_bottom - crop_top
        );

        if crop_bottom > 360.0 {
            println!("    ⚠️  Crop extends beyond image bottom (360px)!");
            println!("    Overshoot: {:.1}px", crop_bottom - 360.0);
        }

        // Actually call align_face
        let aligned = face::align_face(&img, det, 112)?;
        aligned.save(format!("actual_aligned_{}.png", i))?;
        println!("\n  Saved aligned face to: actual_aligned_{}.png", i);
        println!("  Please check if eyes are at ~46% from top (y≈51 in 112px image)");
    }

    Ok(())
}
