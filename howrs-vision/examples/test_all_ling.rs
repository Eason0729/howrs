use anyhow::Result;
use howrs_vision::{face, model};

fn main() -> Result<()> {
    let mut detector = model::detector_session()?;

    // Test all ling images
    for i in 1..=4 {
        let img_path = format!("test_faces/ir-cam/ling{}.png", i);
        println!("\n============================================================");
        println!("Testing: {}", img_path);

        let img = image::open(&img_path)?;
        let (w, h) = (img.width(), img.height());
        println!("Image size: {}x{}", w, h);

        let detections = face::detect_faces(&mut detector, &img, 0.6, 0.3)?;

        if detections.is_empty() {
            println!("❌ No face detected!");
            continue;
        }

        let det = &detections[0];
        println!("✓ Detected face:");
        println!(
            "  Bbox: [{:.0}, {:.0}, {:.0}, {:.0}]",
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3]
        );
        println!("  Landmarks:");
        println!(
            "    Eyes: ({:.0}, {:.0}) and ({:.0}, {:.0})",
            det.landmarks[0], det.landmarks[1], det.landmarks[2], det.landmarks[3]
        );
        println!(
            "    Nose: ({:.0}, {:.0})",
            det.landmarks[4], det.landmarks[5]
        );

        // Check if face is centered
        let center_x = w as f32 / 2.0;
        let center_y = h as f32 / 2.0;
        let bbox_contains_center = det.bbox[0] <= center_x
            && center_x <= det.bbox[0] + det.bbox[2]
            && det.bbox[1] <= center_y
            && center_y <= det.bbox[1] + det.bbox[3];
        println!(
            "  Face centered: {}",
            if bbox_contains_center { "✓" } else { "❌" }
        );

        // Align and save
        let aligned = face::align_face(&img, det, 112)?;
        let out_path = format!("debug_ling{}_crop.png", i);
        aligned.save(&out_path)?;

        // Check crop quality
        let rgb = aligned.to_rgb8();
        let eye_left_pixel = rgb.get_pixel(38, 51);
        let eye_right_pixel = rgb.get_pixel(73, 51);
        let nose_pixel = rgb.get_pixel(56, 71);

        println!("  Crop pixels at landmark positions:");
        println!("    Left eye (38,51): {:?}", eye_left_pixel);
        println!("    Right eye (73,51): {:?}", eye_right_pixel);
        println!("    Nose (56,71): {:?}", nose_pixel);
        println!("  Saved to: {}", out_path);
    }

    Ok(())
}
