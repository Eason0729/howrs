use anyhow::Result;
use howrs_vision::{face, model};
use image::GenericImageView;
use std::path::Path;

#[test]
fn test_all_ling_detections() -> Result<()> {
    env_logger::try_init().ok();
    let mut detector = model::detector_session()?;

    println!("\n=== Testing all ling face detections ===\n");

    for i in 1..=4 {
        let img_path = format!("test_faces/ir-cam/ling{}.png", i);
        if !Path::new(&img_path).exists() {
            println!("ling{}.png: NOT FOUND", i);
            continue;
        }

        let img = image::open(&img_path)?;
        let (w, h) = img.dimensions();
        let center_x = w as f32 / 2.0;
        let center_y = h as f32 / 2.0;

        println!(
            "ling{}.png ({}x{}, center: {:.0}, {:.0}):",
            i, w, h, center_x, center_y
        );

        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            println!("  ⚠ NO FACES DETECTED!");
            continue;
        }

        for (j, det) in detections.iter().enumerate() {
            let [x, y, w_box, h_box] = det.bbox;
            let face_center_x = x + w_box / 2.0;
            let face_center_y = y + h_box / 2.0;

            let dist_from_center =
                ((face_center_x - center_x).powi(2) + (face_center_y - center_y).powi(2)).sqrt();

            println!("  Detection {}:", j);
            println!("    Score: {:.4}", det.score);
            println!("    BBox: [{:.1}, {:.1}, {:.1}, {:.1}]", x, y, w_box, h_box);
            println!(
                "    Face center: ({:.1}, {:.1})",
                face_center_x, face_center_y
            );
            println!(
                "    Distance from image center: {:.1} pixels",
                dist_from_center
            );

            // Check if face contains image center
            let contains_center = x <= center_x
                && center_x <= (x + w_box)
                && y <= center_y
                && center_y <= (y + h_box);
            if contains_center {
                println!("    ✓ Face bbox contains image center");
            } else {
                println!("    ⚠ Face bbox does NOT contain image center");
            }

            // Check landmark positions
            println!("    Landmarks:");
            for k in 0..5 {
                let lx = det.landmarks[k * 2];
                let ly = det.landmarks[k * 2 + 1];
                println!("      Point {}: ({:.1}, {:.1})", k + 1, lx, ly);
            }
        }
        println!();
    }

    Ok(())
}
