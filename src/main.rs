use image::*;
use std::ops::*;
use rand::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

const ASPECT_RATIO : f64 = 1.0;//16.0 / 9.0;
const WIDTH : u32 = 900;
const HEIGHT : u32 = (WIDTH as f64 / ASPECT_RATIO) as u32;
const DEBUG : bool = false;
const SAMPLES_PER_PIXEL : u32 = 500;
const MAX_DEPTH : i32 = 50;
const PI : f64 = std::f64::consts::PI;
const INFINITY : f64 = f64::INFINITY;

fn deg2rad(degrees : f64) -> f64{
    degrees * PI / 180.0
}

fn clamp(x : f64, min: f64, max: f64) -> f64{
    if x > max { max }
    else if x < min { min }
    else { x }
}

fn rand() ->f64{
    thread_rng().gen()
}

fn rand_in_range(min : f64, max : f64) -> f64{
    min + (max-min) * rand()
}

fn color_rgb(pixel_color : &Vec3, samples_per_pixel : u32) -> [u8; 3]{

    let scale = 1.0 / samples_per_pixel as f64;
    let r = (256.0 * clamp((pixel_color.x * scale).sqrt(), 0.0, 0.999)) as u8;
    let g = (256.0 * clamp((pixel_color.y * scale).sqrt(), 0.0, 0.999)) as u8;
    let b = (256.0 * clamp((pixel_color.z * scale).sqrt(), 0.0, 0.999)) as u8;
    [r, g, b]
}

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

    fn rand() -> Vec3{
        Vec3::new(rand(), rand(), rand())
    }

    fn rand_in_range(min : f64, max : f64) -> Vec3{
        Vec3 { x: rand_in_range(min, max), y: rand_in_range(min, max), z: rand_in_range(min, max) }
    }

    fn random_in_unit_sphere() -> Vec3{
        loop{
            let res = Vec3::rand_in_range(-1.0, 1.0);
            if res.squared_length() >= 1.0 { continue;}
            return res;
        }
    }

    fn random_unit_vector() -> Vec3{
        Vec3::random_in_unit_sphere().unit_vector()
    }

    fn random_in_hemisphere(normal : &Vec3) -> Vec3{
        let in_unit_sphere = Vec3::random_in_unit_sphere();
        if in_unit_sphere.dot(normal) > 0.0 { in_unit_sphere }
        else { -in_unit_sphere }
    }

    fn near_zero(&self) -> bool{
        let eps = 1e-8;
        self.x.abs() < eps && self.y.abs() < eps && self.z.abs() < eps
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

#[derive(Clone)]
struct HitRecord {
    point: Vec3,
    normal: Vec3,
    t: f64,
    front_face : bool,
    mat : Arc<dyn Material + Sync + Send>
}

impl HitRecord{
    fn new(point : Vec3, normal : Vec3, t : f64, front_face : bool, mat : Arc<dyn Material + Sync + Send>) -> HitRecord{
        HitRecord { point: point, normal: normal, t: t , front_face : front_face, mat : mat}
    }
}

trait Hittable{
    fn hit(&self, ray : &Ray, t_min : f64, t_max : f64) -> Option<HitRecord>;
}

trait Material{
    fn scatter(&self, r_in : &Ray, rec : &HitRecord, attenuation : &mut Vec3, scattered : &mut Ray) -> bool;
}


#[derive(Debug, Copy, Clone, PartialEq)]
struct Lambertian{
    albedo : Vec3
}

impl Lambertian {
    fn new(color : Vec3) -> Lambertian{
        Lambertian { albedo: color }
    }
}

struct Metal{
    albedo : Vec3,
    fuzz : f64
}

struct Dielectric{
    ir : f64
}

impl Dielectric{
    fn new(ir : f64) -> Dielectric{
        Dielectric { ir: ir }
    }

    fn reflectance(cosine : f64, ref_idx: f64) -> f64{
        let r0 = (1.0-ref_idx) / (1.0 + ref_idx);
        let r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 -cosine).powf(5.0)
    }
}

impl Metal{
    fn new(color : Vec3, fuzz : f64) -> Metal{
        Metal { albedo: color, fuzz: if fuzz < 1.0 {fuzz} else {1.0} }
    }
}

fn reflect(v : &Vec3, n : &Vec3) -> Vec3{
    *v - 2.0 * v.dot(n) * *n
}

fn refract(uv: &Vec3, n: &Vec3, etai_over_etat: f64) -> Vec3{
    let cos_theta = f64::min(-uv.dot(n), 1.0);
    let r_out_perp = etai_over_etat * (*uv + cos_theta * *n);
    let r_out_parallel = -(1.0 - r_out_perp.squared_length()).abs().sqrt() * *n;
    r_out_perp + r_out_parallel
}

impl Material for Dielectric{
    fn scatter(&self, r_in : &Ray, rec : &HitRecord, attenuation : &mut Vec3, scattered : &mut Ray) -> bool{
        
        *attenuation = Vec3::new(1.0, 1.0, 1.0);
        let refraction_ratio = if rec.front_face { 1.0 / self.ir} else {self.ir};
        let unit_direction = r_in.direction.unit_vector();

        let cos_theta = f64::min(-unit_direction.dot(&rec.normal), 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let direction = if cannot_refract || Dielectric::reflectance(cos_theta, refraction_ratio) > rand(){ 
            reflect(&unit_direction, &rec.normal)} 
            else {refract(&unit_direction, &rec.normal, refraction_ratio)};

        //let refracted = refract(&unit_direction, &rec.normal, refraction_ratio);
        *scattered = Ray::new(rec.point, direction);
        true
    }
}

impl Material for Lambertian{
    fn scatter(&self, r_in : &Ray, rec : &HitRecord, attenuation : &mut Vec3, scattered : &mut Ray) -> bool{
        //let mut scatter_direction = rec.normal + Vec3::random_in_unit_sphere();
        let mut scatter_direction = rec.normal + Vec3::random_unit_vector();

        if scatter_direction.near_zero() { scatter_direction = rec.normal; }

        *scattered = Ray::new(rec.point, scatter_direction);
        *attenuation = self.albedo;
        true
    }
}

impl Material for Metal{
    fn scatter(&self, r_in : &Ray, rec : &HitRecord, attenuation : &mut Vec3, scattered : &mut Ray) -> bool{
        
        let reflected = reflect(&r_in.direction.unit_vector(), &rec.normal);
        //*scattered = Ray::new(rec.point, reflected);
        *scattered = Ray::new(rec.point, reflected + self.fuzz * Vec3::random_in_unit_sphere());
        *attenuation = self.albedo;
        scattered.direction.dot(&rec.normal) > 0.0
    }
}

struct Camera{
    aspect_ratio : f64,
    viewport_height : f64,
    viewport_width : f64,
    focal_length : f64,
    origin : Vec3,
    lower_left_corner : Vec3,
    horizontal : Vec3,
    vertical : Vec3
}

impl Camera {
    fn default() -> Camera{
        Camera { aspect_ratio: ASPECT_RATIO, 
            viewport_height: 2.0, 
            viewport_width: ASPECT_RATIO * 2.0, 
            focal_length: 1.0, 
            origin: Vec3::zero(), 
            lower_left_corner: Vec3::zero() - Vec3::new(ASPECT_RATIO * 2.0, 0.0, 0.0)/2.0 - Vec3::new(0.0, 2.0, 0.0)/2.0 - Vec3::new(0.0, 0.0, 1.0), 
            horizontal: Vec3::new(ASPECT_RATIO * 2.0, 0.0, 0.0), 
            vertical: Vec3::new(0.0, 2.0, 0.0) }
    }

    fn new(aspect_ratio : f64, viewport_height : f64, focal_length : f64, origin : Vec3) -> Camera{
        let viewport_width = aspect_ratio * viewport_height;
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Camera { aspect_ratio: aspect_ratio, viewport_height: viewport_height, 
            viewport_width: viewport_width, focal_length: focal_length, 
            origin: origin, lower_left_corner: lower_left_corner, 
            horizontal: horizontal, vertical: vertical }
    }

    fn get_ray(&self, u: f64, v: f64) -> Ray{
        Ray::new(self.origin, self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin)
    }
}

#[derive(Clone)]
struct Sphere {
    center: Vec3,
    radius: f64,
    mat : Arc<dyn Material + Sync + Send>
}

impl Sphere{
    fn new(center : Vec3, radius : f64, mat : Arc<dyn Material + Sync + Send>) -> Sphere{
        Sphere { center: center, radius: radius, mat: mat }
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
        let res = HitRecord::new(point, normal, root, front_face, self.mat.clone());

        Some(res)
    }
}

fn hit(list : &Vec<Arc<dyn Hittable + Sync + Send>>, ray : &Ray, t_min : f64, t_max : f64) -> Option<HitRecord>{
    let  mut hit_record : Option<HitRecord> = None;
    let mut closest_so_far = t_max;

    for o in list{
        let result = o.hit(ray, t_min, closest_so_far);
        match result{
            Some(h) => {
                hit_record = Some(h.clone());
                closest_so_far = h.t;
            }
            None => {}
        }
    }
    hit_record
}

fn ray_color(r: &Ray, list : &Vec<Arc<dyn Hittable + Sync + Send>>, depth : i32) -> Vec3{

    if depth <= 0 {return Vec3::zero();}

    let hit = hit(list, r, 0.001, 100000.0);

    match hit{
        Some(h) => {
            let mut scattered = Ray::new(Vec3::zero(), Vec3::zero());
            let mut attenuation = Vec3::zero();
            if h.mat.scatter(r, &h, &mut attenuation, &mut scattered)
                { return attenuation * ray_color(&scattered, list, depth-1);}
            else { return Vec3::zero(); };
        }
        None => {}
    }

    let unit_dir = r.direction.unit_vector();
    let t = 0.5 * (unit_dir.y + 1.0);
    (1.0 - t) * Vec3::new(1.0, 1.0, 1.0) + t * Vec3::new(0.5, 0.7, 1.0)
}

fn main() {
    // Construct a new RGB ImageBuffer with the specified width and height.
    let mut image: RgbImage = ImageBuffer::new(WIDTH, HEIGHT);

    // Materials
    let mat0 : Arc<dyn Material + Sync + Send> = Arc::new(Lambertian::new(Vec3::new(0.8, 0.8, 0.0)));
    let mat1 : Arc<dyn Material + Sync + Send> = Arc::new(Lambertian::new(Vec3::new(0.8, 0.0, 0.8)));
    let mat2 : Arc<dyn Material + Sync + Send> = Arc::new(Lambertian::new(Vec3::new(0.0, 0.8, 0.8)));
    let mat3 : Arc<dyn Material + Sync + Send> = Arc::new(Metal::new(Vec3::new(0.8, 0.8, 0.8), 0.2));
    let mat4 : Arc<dyn Material + Sync + Send> = Arc::new(Dielectric::new(1.5));
    let mat5 : Arc<dyn Material + Sync + Send> = Arc::new(Lambertian::new(Vec3::new(0.8, 0.0, 0.0)));

    // World
    let mut world : Vec<Arc<dyn Hittable + Sync + Send>> = Vec::new();
    world.push(Arc::new(Sphere::new(Vec3::new(0.0, -100.5, -1.0), 100.0, mat0.clone())));
    world.push(Arc::new(Sphere::new(Vec3::new(-0.7, 0.0, -2.75), 0.5, mat4.clone())));
    world.push(Arc::new(Sphere::new(Vec3::new(-2.0, 1.5, -4.25), 0.5, mat5.clone())));
    world.push(Arc::new(Sphere::new(Vec3::new(-1.0, 0.0, -1.0), 0.5, mat2.clone())));
    world.push(Arc::new(Sphere::new(Vec3::new(2.15, 0.5, -3.0), 0.5, mat1.clone())));
    world.push(Arc::new(Sphere::new(Vec3::new(0.3, 0.0, -1.75), 0.5, mat3.clone())));
    
    // Camera
    let camera = Camera::default();
    
    // Iterate over all pixels in the image.
    image.enumerate_pixels_mut().par_bridge().for_each(|(x, y, pixel)|{
        let y = (HEIGHT - 1) - y; 
        
        // Do something with pixel.
        let mut pixel_color = Vec3::zero();
        for _i in 0..SAMPLES_PER_PIXEL {
            let u = (x as f64 + rand()) / (WIDTH as f64 - 1.0);
            let v = (y as f64 + rand()) / (HEIGHT as f64 - 1.0);

            pixel_color += ray_color(&camera.get_ray(u, v), &world, MAX_DEPTH);
        }

        *pixel = image::Rgb(color_rgb(&pixel_color, SAMPLES_PER_PIXEL));

        if DEBUG { println!("{},{}", x, y) };
    });
    
    // write it out to a file
    image.save("output.png").unwrap();
}