use crate::renderers::mesh_renderer::*;
use crate::scene::*;
use crate::kinput::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use glutin::event::VirtualKeyCode;

// good way to get slopeyness of a normal? probably if y was 1, max slope is 0.5
// y is 0.1, max slope is probably 0.05. So maybe its just remap Y from 0.95-1.0 to 0.0-1.0
// wow its very 4 way directional and also I wouldve done this for rustvox
// vertical projection is surely normalize and take the y direction?
// maybe its just because the terrain is super directional

// yes using magnitude of gradient
// also the one where it uses that to control amount of higher octave noise? and weve got changing the distribution  of the values for heights. yeah could use a fancy distribution, rare high heights
// how to steepen
// what about erosion simulation

// lol well i could just have wonky noise: chance to be 0..1 and then 0..3 or something. well its a bit like fractal noise if you made the high bits spread out too. look at some mountains

// gradients for cliffs and shit: could flood fill the grass

fn expnoise(seed: u32, x: f64, y: f64) -> f64 {
    // i think -ln of something 0..1 is exp dist
    -noise2d(x, y, seed).ln()
}

fn fracexpnoise(seed: u32, x: f64, y: f64) -> f64 {
    1.000 * expnoise(seed, x, y) +
    0.500 * expnoise(seed*1238715, x*2.0, y*2.0) +
    0.250 * expnoise(seed*9148167, x*4.0, y*4.0) +
    0.125 * expnoise(seed*2442347, x*8.0, y*8.0) /
    1.875
}

fn fracnoise(seed: u32, x: f64, y: f64) -> f64 {
    1.000 * noise2d(x, y, seed) +
    0.500 * noise2d(x*2.0, y*2.0, seed*1238715) +
    0.250 * noise2d(x*4.0, y*4.0, seed*9148167) +
    0.125 * noise2d(x*8.0, y*8.0, seed*2442347) /
    1.875
}

fn ridged2(seed: u32, x: f64, y: f64) -> f64 {
    3.0 * (1.0 - (noise2d(x, y, seed) - 0.5).abs()) +
    0.500 * noise2d(x*2.0, y*2.0, seed*1238715) +
    0.250 * noise2d(x*4.0, y*4.0, seed*9148167) +
    0.125 * noise2d(x*8.0, y*8.0, seed*2442347) /
    3.875
}

fn ridged(seed: u32, x: f64, y: f64) -> f64 {
    1.0 - (fracnoise(seed, x, y) - 0.5).abs()
}

fn hh(seed: u32, x: f64, y: f64) -> f64 {
    // fracnoise(seed, x, y) * 0.1
    ridged2(seed, x, y) * 0.1
}

fn hn(seed: u32, x: f64, y: f64) -> (f64, Vec3) {
    let d = 0.01;
    let h = hh(seed, x, y);
    let hgx = hh(seed, x + d, y);
    let hgy = hh(seed, x, y + d);
    let vx = Vec3::new(d, hgx - h, 0.0);
    let vz = Vec3::new(0.0, hgy - h, d);

    let norm = vz.cross(vx);

    (h, norm)
}

pub struct Terrain {
    stale: bool,
    w: usize,
    h: usize,
    height: Vec<f32>,
    colour: Vec<Vec4>,
    normal: Vec<Vec3>,
    seed: u32,
}

impl Terrain {
    pub fn gen(&mut self) {
        self.height.resize(self.w * self.h, 0.0);
        self.normal.resize(self.w * self.h, Vec3::new(0.0, 0.0, 0.0));
        self.colour.resize(self.w * self.h, Vec4::new(0.0, 0.0, 0.0, 1.0));
        for i in 0..self.w {
            for j in 0..self.h {
                let nx = 4.0 * i as f64 / self.w as f64;
                let ny = 4.0 * j as f64 / self.h as f64;

                let (h, norm) = hn(self.seed, nx, ny);
                let norm = norm.normalize();

                let slopeyness = remap(norm.y, 0.95, 1.0, 0.0, 1.0);

                let c1 = Vec4::new(1.0, 0.0, 0.0, 1.0);
                let c2 = Vec4::new(0.0, 1.0, 0.0, 1.0);


                self.colour[j*self.w + i] = c1.lerp(c2, slopeyness);

                // self.colour[j*self.w + i] = Vec4::new((norm.x + 1.0) / 2.0, (norm.y + 1.0)/2.0, (norm.z + 1.0) / 2.0, 1.0);
                self.normal[j*self.w + i] = norm;
                self.height[j*self.w + i] = h as f32;
            }
        }
        self.stale = true;
    }
}

impl Default for Terrain {
    fn default() -> Self {
        let w = 400;
        let h = 400;

        let mut t = Terrain {
            stale: true,
            w,
            h,
            seed: 0,
            colour: vec![],
            height: vec![],
            normal: vec![],
        };
        t.gen();
        t
    }
}

impl Demo for Terrain {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
        if inputs.key_rising(VirtualKeyCode::R) {
            self.seed += 1;
            self.gen();
        }

        if self.stale {
            let mut mb = MeshBuilder::default();
            for i in 0..self.w {
                for j in 0..self.h {
                    let x = i as f32 / self.w as f32;
                    let y = j as f32 / self.h as f32;
                    let z = self.height[j * self.w + i];
                    let u = i as f32 / self.w as f32;
                    let v = j as f32 / self.h as f32;

                    let pos = Vec3::new(x as f64, z as f64, y as f64);
                    let uv = Vec2::new(u as f64, v as f64);
                    let colour = Vec4::new(1.0, 1.0, 1.0, 1.0);

                    mb.push_element(pos, uv, self.normal[j*self.w + i], colour);
                }
            }
            for i in 0..self.w - 1 {
                for j in 0..self.h - 1 {
                    let i = j * self.w + i;
                    mb.push_tri(i as u32, (i + 1) as u32, (i + self.w) as u32);
                    mb.push_tri((i+1) as u32, (i + self.w + 1) as u32, (i + self.w) as u32);
                }
            }
            outputs.set_mesh = Some(mb);
            let mut tb = TextureBuffer::new(self.w, self.h);
            for i in 0..self.w {
                for j in 0..self.h {
                    tb.set(i as i32, j as i32, self.colour[j*self.w + i]);
                }
            }
            outputs.set_mesh_texture = Some(tb);
            self.stale = false;
        }

        let mt = translation(-0.5, 0.0, -0.5);
        let mr = roty(inputs.t as f32 / 5.0);
        let mm = mat4_mul(mr, mt);

        let cp = Vec3::new(1.0, 0.5, 0.0);
        let ct = Vec3::new(0.0, 0.0, 0.0);
        let cd = ct - cp;
        
        let v = view(cp, ct);
        let p = proj(1.0, inputs.screen_rect.aspect() as f32, 0.001, 100.0);
        let vp = mat4_mul(p, v);

        outputs.draw_mesh = Some((vp, mm, cp, cd));
    }
}

fn rec_noise(max: i32, seed: u32, x: f64, y: f64) -> f64 {
    // and 0 is pretty suss so what if instead of a random unit -1,1 x and y it was just r theta, or r theta varying

    let rns = 0.25;
    let ds = 2.0;

    // let dx = ds * (2.0 * noise2d(rns * x, rns * y, seed  * 1312317) - 1.0);
    // let dy = ds * (2.0 * noise2d(rns * x, rns * y, seed  * 3412477) - 1.0);

    let ra = 4.0;
    
    let r = 0.1 * max as f64 * ds * noise2d(ra * rns * x, ra * rns * y, seed  * 1312317);
    // let r = ds * noise2d(ra * rns * x, ra * rns * y, seed  * 1312317);
    let theta =  2.0 * PI * noise2d(rns * x, rns * y, seed  * 3412477);
    let dx = r * theta.sin();
    let dy = r * theta.cos();

    let mut acc = 0.0;
    acc += noise2d(x, y, seed);  // was lookin like good terrain
    // acc += x.sin() + y.cos();   // shiet welcome to the spooky zone
    // acc += x.sin();
    if max > 0 {
        // acc += rec_noise(max - 1, seed, x + dx, y + dy);
        acc = rec_noise(max - 1, seed, x + dx, y + dy);
    }
    acc
}

// maybe we can remove crumples if we constrain theta to one quadrant
// cause in the middle its really good and appropriate