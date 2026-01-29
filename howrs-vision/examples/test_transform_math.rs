// Minimal test of similarity transform
// Test if our transform math is correct by using simple known values

fn main() {
    // Simple test: src eyes at (10, 20) and (30, 20), dst eyes should be at (38, 52) and (74, 52)
    let src = vec![[10.0, 20.0], [30.0, 20.0]];
    let dst = vec![[38.0, 52.0], [74.0, 52.0]];

    // Compute means
    let src_mean = [(10.0 + 30.0) / 2.0, (20.0 + 20.0) / 2.0]; // (20, 20)
    let dst_mean = [(38.0 + 74.0) / 2.0, (52.0 + 52.0) / 2.0]; // (56, 52)

    println!("src_mean: ({:.1}, {:.1})", src_mean[0], src_mean[1]);
    println!("dst_mean: ({:.1}, {:.1})", dst_mean[0], dst_mean[1]);

    // Center
    let src_centered = vec![
        [src[0][0] - src_mean[0], src[0][1] - src_mean[1]], // (-10, 0)
        [src[1][0] - src_mean[0], src[1][1] - src_mean[1]], // (10, 0)
    ];
    let dst_centered = vec![
        [dst[0][0] - dst_mean[0], dst[0][1] - dst_mean[1]], // (-18, 0)
        [dst[1][0] - dst_mean[0], dst[1][1] - dst_mean[1]], // (18, 0)
    ];

    // Scale: sqrt(sum of squared norms)
    let src_norm = ((-10f32).powi(2) + 10f32.powi(2)).sqrt(); // sqrt(200) = 14.14
    let dst_norm = ((-18f32).powi(2) + 18f32.powi(2)).sqrt(); // sqrt(648) = 25.46
    let scale = dst_norm / src_norm; // 1.8

    println!("scale: {:.4}", scale);

    // Rotation: both points are horizontal (y=const), so theta should be 0
    // num = sum(sx * dy - sy * dx) = (-10)*0 - 0*(-18) + 10*0 - 0*18 = 0
    // den = sum(sx * dx + sy * dy) = (-10)*(-18) + 0*0 + 10*18 + 0*0 = 180 + 180 = 360
    let mut num = 0.0f32;
    let mut den = 0.0f32;
    for i in 0..2 {
        num += src_centered[i][0] * dst_centered[i][1] - src_centered[i][1] * dst_centered[i][0];
        den += src_centered[i][0] * dst_centered[i][0] + src_centered[i][1] * dst_centered[i][1];
    }
    let theta = num.atan2(den);
    println!("theta: {:.4} rad ({:.1}°)", theta, theta.to_degrees());

    let a = scale * theta.cos(); // 1.8 * 1 = 1.8
    let b = scale * theta.sin(); // 1.8 * 0 = 0

    // Translation: this is the key!
    // Method 1 (current code): tx = dst_mean[0] - (a * src_mean[0] + b * src_mean[1])
    let tx1 = dst_mean[0] - (a * src_mean[0] + b * src_mean[1]);
    let ty1 = dst_mean[1] - (-b * src_mean[0] + a * src_mean[1]);

    println!("\nMethod 1 (current):");
    println!(
        "  tx = {:.1} - ({:.1} * {:.1} + {:.1} * {:.1}) = {:.1}",
        dst_mean[0], a, src_mean[0], b, src_mean[1], tx1
    );
    println!(
        "  ty = {:.1} - ({:.1} * {:.1} + {:.1} * {:.1}) = {:.1}",
        dst_mean[1], -b, src_mean[0], a, src_mean[1], ty1
    );

    // Test forward transform
    println!("\nForward transform test:");
    for i in 0..2 {
        let x_out = a * src[i][0] + b * src[i][1] + tx1;
        let y_out = -b * src[i][0] + a * src[i][1] + ty1;
        println!(
            "  src({:.0}, {:.0}) -> ({:.1}, {:.1}), expected ({:.0}, {:.0})",
            src[i][0], src[i][1], x_out, y_out, dst[i][0], dst[i][1]
        );
    }
}
