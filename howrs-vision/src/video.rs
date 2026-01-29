use anyhow::{Context, Result};
use image::{ImageBuffer, Rgb};
use v4l::buffer::Type;
use v4l::io::mmap::Stream;
use v4l::io::traits::CaptureStream;
use v4l::video::Capture;
use v4l::{Device, Format, FourCC};

pub struct Camera {
    stream: Stream<'static>,
    width: u32,
    height: u32,
    fourcc: FourCC,
}

impl Camera {
    pub fn open(device: &str) -> Result<Self> {
        let dev = Device::with_path(device).context("open camera")?;
        let mut fmt = dev.format().context("get format")?;
        // Prefer RGB, fallback to YUYV, else accept existing format
        let desired = Format::new(fmt.width, fmt.height, FourCC::new(b"RGB3"));
        fmt = dev.set_format(&desired).unwrap_or(fmt);
        if fmt.fourcc != FourCC::new(b"RGB3") {
            let yuyv = Format::new(fmt.width, fmt.height, FourCC::new(b"YUYV"));
            fmt = dev.set_format(&yuyv).unwrap_or(fmt);
        }
        let fourcc = fmt.fourcc;
        let width = fmt.width;
        let height = fmt.height;
        let stream = Stream::with_buffers(&dev, Type::VideoCapture, 4).context("stream")?;
        Ok(Self {
            stream,
            width,
            height,
            fourcc,
        })
    }

    pub fn frame(&mut self) -> Result<ImageBuffer<Rgb<u8>, Vec<u8>>> {
        let (data, meta) = self.stream.next().context("capture frame")?;
        log::debug!(
            "captured frame: width={} height={} fourcc={:?} seq={:?} len={}",
            self.width,
            self.height,
            self.fourcc,
            meta.sequence,
            data.len()
        );
        let buf = match self.fourcc {
            f if f == FourCC::new(b"RGB3") => data.to_vec(),
            f if f == FourCC::new(b"YUYV") => yuyv_to_rgb(self.width, self.height, &data)?,
            f if f == FourCC::new(b"GREY") => grey_to_rgb(self.width, self.height, &data)?,
            other => {
                log::warn!(
                    "unexpected pixel format {:?}, passing through raw len={}",
                    other,
                    data.len()
                );
                data.to_vec()
            }
        };
        let expected = (self.width * self.height * 3) as usize;
        if buf.len() < expected {
            log::error!(
                "buffer too small: got {}, expected {} (fourcc {:?})",
                buf.len(),
                expected,
                self.fourcc
            );
            return Err(anyhow::anyhow!("buffer too small"));
        } else if buf.len() > expected {
            log::warn!(
                "buffer larger than expected ({} > {}), truncating",
                buf.len(),
                expected
            );
        }
        Ok(ImageBuffer::from_raw(self.width, self.height, buf)
            .ok_or_else(|| anyhow::anyhow!("failed to build image buffer"))?)
    }
}

fn yuyv_to_rgb(width: u32, height: u32, data: &[u8]) -> Result<Vec<u8>> {
    let expected = (width * height * 2) as usize;
    if data.len() < expected {
        return Err(anyhow::anyhow!("short YUYV buffer"));
    }
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    let mut chunks = data.chunks_exact(4);
    while let Some(chunk) = chunks.next() {
        let y0 = chunk[0] as f32;
        let u = chunk[1] as f32 - 128.0;
        let y1 = chunk[2] as f32;
        let v = chunk[3] as f32 - 128.0;
        for &y in &[y0, y1] {
            let r = y + 1.402 * v;
            let g = y - 0.344136 * u - 0.714136 * v;
            let b = y + 1.772 * u;
            out.push(clamp(r));
            out.push(clamp(g));
            out.push(clamp(b));
        }
    }
    Ok(out)
}

fn clamp(v: f32) -> u8 {
    v.max(0.0).min(255.0) as u8
}

fn grey_to_rgb(width: u32, height: u32, data: &[u8]) -> Result<Vec<u8>> {
    let expected = (width * height) as usize;
    if data.len() < expected {
        return Err(anyhow::anyhow!("short GREY buffer"));
    }
    let mut out = Vec::with_capacity((width * height * 3) as usize);
    for &y in data.iter().take(expected) {
        out.push(y);
        out.push(y);
        out.push(y);
    }
    Ok(out)
}
