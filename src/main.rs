use image::*;
use std::ops::*;

const WIDTH : u32 = 256;
const HEIGHT : u32 = 256;
const DEBUG : bool = false;

#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec3 {
    x: f64,
    y: f64,
    z: f64
}


impl Vec3{

    fn new(x: f64,  y: f64, z:f64) -> Vec3{
        Vec3{x: x, y: y, z: z}
    }

    fn zero() -> Vec3 {
        Vec3{x:0.0, y:0.0, z:0.0}
    }

    fn squared_length(&self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    fn length(&self) -> f64 {
        self.squared_length().sqrt()
    }

    fn dot(&self, v : &Vec3) -> Vec3 {
        Vec3 { x: self.x * v.x, y: self.y * v.y, z: self.z * v.z }
    }

    fn cross(&self, v : &Vec3) -> Vec3 {
        Vec3 { x: self.y * v.z - self.z * v.y, 
               y: self.z * v.x - self.x * v.z, 
               z: self.x * v.y - self.y * v.x }
    }

    fn normalize(&mut self){
        let l = self.length();
        *self = Self {
            x: self.x / l,
            y: self.y / l,
            z: self.z / l
        };
    }

    fn unit_vector(&self) -> Vec3 {
        let l = self.length();
        Vec3::new(self.x / l, self.y / l, self.z / l)
    }

}

impl Add for Vec3 {
    type Output = Vec3;

    fn add(self, v: Vec3) -> Vec3 {
        Vec3 { x: self.x + v.x, y: self.y + v.y, z: self.z + v.z }
    }
}

impl AddAssign for Vec3{
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z
        };
    }
}

impl Neg for Vec3 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec3 { x: -self.x, y: -self.y, z: -self.z}
    }
}

impl Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, v: Vec3) -> Vec3 {
        Vec3 { x: self.x - v.x, y: self.y - v.y, z: self.z - v.z }
    }
}

impl SubAssign for Vec3{
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z
        };
    }
}

impl Mul<Vec3> for Vec3 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        Vec3 { x: self.x * v.x, y: self.y * v.y, z: self.z * v.z }
    }
}

impl Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        Vec3 { x: self * v.x, y: self * v.y, z: self * v.z }
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;

    fn mul(self, f: f64) -> Vec3 {
        Vec3 { x: self.x * f, y: self.y * f, z: self.z * f }
    }
}

impl MulAssign<Vec3> for Vec3{
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x * other.x,
            y: self.y * other.y,
            z: self.z * other.z
        };
    }
}

impl MulAssign<f64> for Vec3{
    fn mul_assign(&mut self, f: f64) {
        *self = Self {
            x: self.x * f,
            y: self.y * f,
            z: self.z * f
        };
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;

    fn div(self, f: f64) -> Vec3 {
        Vec3 { x: self.x / f, y: self.y / f, z: self.z / f }
    }
}

impl DivAssign<f64> for Vec3{
    fn div_assign(&mut self, f: f64) {
        *self = Self {
            x: self.x / f,
            y: self.y / f,
            z: self.z / f
        };
    }
}


fn main() {

    // Construct a new RGB ImageBuffer with the specified width and height.
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);
    let v0 = Vec3::zero();
    let mut  v1 =  2.0 * Vec3::new(5.5, -12.455, 2.0);
    let v2 = Vec3::new(10.1, 8.0, -5.66);
    v1 *= 2.0;
    v1 /= 4.0;
    println!("{:?}", v0);
    println!("{:?}", v1);
    println!("{:?}", v2);
    println!("{:?}", v1 - v2);
    println!("{:?}", v1.unit_vector());
    v1.normalize();
    println!("{:?}", v1);
    println!("{:?}", v1.squared_length());
    println!("{:?}", v1.length());
    
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
