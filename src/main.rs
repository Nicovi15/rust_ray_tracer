use image::*;

const WIDTH : u32 = 256;
const HEIGHT : u32 = 256;
const DEBUG : bool = false;

fn main() {

    // Construct a new RGB ImageBuffer with the specified width and height.
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);
    
    // Iterate over all pixels in the image.
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        // Do something with pixel.
        let r = ((x as f32) / (WIDTH as f32)) as f32;
        let g = ((y as f32) / (HEIGHT as f32)) as f32;
        let b = 0.25 as f32;

        let r = (255.0_f32  * r) as u8;
        let g = (255.0_f32  * g) as u8;
        let b = (255.0_f32  * b) as u8;

        *pixel = image::Rgb([r, g, b]);

        if DEBUG { println!("{},{}", x, y) };
    }
    
    // write it out to a file
    image.save("output.png").unwrap();
}
