//! Visualize what YuNet actually sees as input
use anyhow::Result;
use image::GenericImageView;

#[test]
#[ignore]
fn visualize_yunet_input() -> Result<()> {
    let img_path = "test_faces/ir-cam/eason1.png";
    let img = image::open(img_path)?;
    let (orig_w, orig_h) = img.dimensions();
    println!("Original image: {}x{}", orig_w, orig_h);

    // Replicate the padding logic from detect_faces
    let target_size = 640u32;
    let max_dim = orig_w.max(orig_h);
    let scale = target_size as f32 / max_dim as f32;
    let new_width = (orig_w as f32 * scale) as u32;
    let new_height = (orig_h as f32 * scale) as u32;

    println!("Scale: {}", scale);
    println!("Resized to: {}x{}", new_width, new_height);

    let resized = img.resize_exact(new_width, new_height, image::imageops::FilterType::Triangle);

    let mut canvas = image::DynamicImage::new_rgb8(target_size, target_size);
    let offset_x = (target_size - new_width) / 2;
    let offset_y = (target_size - new_height) / 2;

    println!("Canvas: {}x{}", target_size, target_size);
    println!("Offset: ({}, {})", offset_x, offset_y);
    println!(
        "Image placed at: ({}, {}) to ({}, {})",
        offset_x,
        offset_y,
        offset_x + new_width,
        offset_y + new_height
    );

    image::imageops::overlay(&mut canvas, &resized, offset_x as i64, offset_y as i64);

    canvas.save("yunet_input_canvas.png")?;
    println!("\nSaved YuNet input canvas to: yunet_input_canvas.png");
    println!("The face should be in the MIDDLE (vertically) of this 640x640 image");
    println!("If it's in the lower portion, then YuNet will detect eyes in lower portion!");

    Ok(())
}
