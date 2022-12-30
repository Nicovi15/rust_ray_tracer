use image::*;
use std::ops::Add;

const WIDTH : u32 = 256;
const HEIGHT : u32 = 256;
const DEBUG : bool = false;

#[derive(Debug)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}

/*
impl Default for  Vec3 {
    fn default() -> Self {
        Self {
            x : 0.0,
            y : 0.0,
            z : 0.0
        }
    }
}
*/

impl Vec3{

    fn new(x: f64,  y: f64, z:f64) -> Vec3{
        Vec3{x: x, y: y, z: z}
    }

    fn zero() -> Vec3 {
        Vec3{x:0.0, y:0.0, z:0.0}
    }

}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, v: Vec3) -> Vec3 {
        Vec3 { x: self.x + v.x, y: self.y + v.y, z: self.z + v.z }
    }
}

fn main() {

    // Construct a new RGB ImageBuffer with the specified width and height.
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);
    let v0 = Vec3::zero();
    let v1 = Vec3::new(5.5, -12.455, 2.0);
    let v2 = Vec3::new(10.1, 8.0, -5.66);
    println!("{:?}", v0);
    println!("{:?}", v1);
    println!("{:?}", v2);
    println!("{:?}", v1 + v2);
    
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
