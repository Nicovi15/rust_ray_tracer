use image::*;
use std::ops::*;
use std::rc::Rc;

const ASPECT_RATIO : f64 = 16.0 / 9.0;
const WIDTH : u32 = 400;
const HEIGHT : u32 = (WIDTH as f64 / ASPECT_RATIO) as u32;
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

    fn dot(&self, v : &Vec3) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z 
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

#[derive(Debug, Copy, Clone, PartialEq)]
struct Ray {
    origin: Vec3,
    direction: Vec3
}

impl Ray{

    fn new(origin: Vec3,  direction: Vec3) -> Ray{
        Ray{origin : origin, direction : direction}
    }

    fn at(&self, t: f64) -> Vec3{
        self.origin + t * self.direction
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f64,
    front_face : bool
}

impl HitRecord{
    fn new(point : Vec3, normal : Vec3, t : f64, front_face : bool) -> HitRecord{
        HitRecord { point: point, normal: normal, t: t , front_face : front_face}
    }
}

trait Hittable{
    fn hit(&self, ray : &Ray, t_min : f64, t_max : f64) -> Option<HitRecord>;
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct Sphere {
    center: Vec3,
    radius: f64
}

impl Sphere{
    fn new(center : Vec3, radius : f64) -> Sphere{
        Sphere { center: center, radius: radius }
    }
}

impl Hittable for Sphere{
    fn hit(&self, ray : &Ray, t_min : f64, t_max : f64) -> Option<HitRecord>{
        let oc = ray.origin - self.center;
        let a = ray.direction.squared_length();
        let half_b = oc.dot(&ray.direction);
        let c = oc.squared_length() -  self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
           return None;
        }
        let sqrtd = discriminant.sqrt();
        let mut root = (-half_b - sqrtd ) / a;
        if root < t_min || t_max < root {
            root = (-half_b + sqrtd) / a;
            if root < t_min || t_max < root {
                return None;
            }
        }
        let point = ray.at(root); 
        let outward_normal = (point - self.center) / self.radius;
        let front_face = ray.direction.dot(&outward_normal) < 0.0;
        let normal =  if front_face {outward_normal}  else {-outward_normal};
        let res = HitRecord::new(point, normal, root, front_face);

        Some(res)
    }
}

fn hit(list : &Vec<Rc<dyn Hittable>>, ray : &Ray, t_min : f64, t_max : f64) -> Option<HitRecord>{
    let  mut hit_record : Option<HitRecord> = None;
    let mut closest_so_far = t_max;

    for o in list{
        let result = o.hit(ray, t_min, closest_so_far);
        match result{
            Some(h) => {
                hit_record = Some(h);
                closest_so_far = h.t;
            }
            None => {}
        }
    }
    hit_record
}


fn ray_color(r: &Ray, list : &Vec<Rc<dyn Hittable>>) -> Vec3{
    let hit = hit(list, r, 0.0, 100000.0);

    match hit{
        Some(h) => {
            return 0.5 * (h.normal + Vec3::new(1.0, 1.0, 1.0));
        }
        None => {}
    }

    let unit_dir = r.direction.unit_vector();
    let t = 0.5 * (unit_dir.y + 1.0);
    (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}

fn main() {

    // Vec3 test
    /*
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
    */

    // Construct a new RGB ImageBuffer with the specified width and height.
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);

    // Wolrd
    let mut world : Vec<Rc<dyn Hittable>> = Vec::new();
    world.push(Rc::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0)));
    world.push(Rc::new(Sphere::new(Vec3::new(0.0, 0.0, -1.0), 0.5)));
    
    // Camera
    let viewport_height : f64 = 2.0;
    let viewport_width : f64 = ASPECT_RATIO * viewport_height;
    let focal_length : f64 = 1.0;

    let origin = Vec3::new(0.0, 0.0, 0.0);
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - Vec3::new(0.0, 0.0, focal_length);
    
    // Iterate over all pixels in the image.
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let y = (HEIGHT - 1) - y; 
        
        // Do something with pixel.
        let u = (x as f64) / (WIDTH as f64 - 1.0);
        let v = (y as f64) / (HEIGHT as f64 - 1.0);
        let r = Ray::new(origin, lower_left_corner + u*horizontal + v*vertical - origin);
        let pixel_color = ray_color(&r, &world);

        let r = (255.9  * pixel_color.x) as u8;
        let g = (255.9  * pixel_color.y) as u8;
        let b = (255.9  * pixel_color.z) as u8;

        *pixel = image::Rgb([r, g, b]);

        if DEBUG { println!("{},{}", x, y) };
    }
    
    // write it out to a file
    image.save("output.png").unwrap();
}
