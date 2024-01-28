use crate::scene::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use crate::kinput::*;
use glutin::event::VirtualKeyCode;
use crate::renderers::mesh_renderer::*;
use std::collections::hash_map::*;

pub struct NoiseExplorer {
    w: usize,
    h: usize,

    seed: u32,

    stale: bool,
    stale_mesh: bool,
    show_mesh: bool,

    height: Vec<f64>,
    colour: Vec<Vec4>,
    normal: Vec<Vec3>,
    max: f64,

    noise_fn: fn(u32, f64, f64) -> f64,
    noise_name: &'static str,
    t_change: f64,
}

impl Default for NoiseExplorer {
    fn default() -> Self {
        let w = 800;
        let h = 800;
        NoiseExplorer {
            w,
            h,
            seed: 69,
            stale: true,
            stale_mesh: true,
            show_mesh: false,
            colour: vec![],
            height: vec![],
            normal: vec![],
            max: 0.0,
            noise_fn: fsxnoise,
            noise_name: "fsxnoise",
            t_change: 0.0,
        }
    }
}

impl NoiseExplorer {
    pub fn run(&mut self) {
        let tstart = std::time::SystemTime::now();
        self.height.resize(self.w * self.h, 0.0);
        self.normal.resize(self.w * self.h, Vec3::new(0.0, 0.0, 0.0));
        self.colour.resize(self.w * self.h, Vec4::new(0.0, 0.0, 0.0, 1.0));
        self.max = 0.0;

        
        for i in 0..self.w {
            for j in 0..self.h {
                let nx = 1.0 * i as f64 / self.w as f64;
                let ny = 1.0 * j as f64 / self.h as f64;
                //let h = noise2d(nx, ny, self.seed + 12341237);
                let (h, n) = self.hn(self.seed, nx, ny);
                let n = n.normalize();
                self.normal[(j*self.w + i) as usize] = n;
                let slopeyness = remap(n.y, 0.0, 1.0, 0.0, 1.0);
                let c1 = Vec4::new(1.0, 0.0, 0.0, 1.0);
                let c2 = Vec4::new(0.0, 1.0, 0.0, 1.0);
                // self.colour[j*self.w + i] = c1.lerp(c2, slopeyness);
                self.colour[j*self.w + i] = c1.lerp(c2, h);
                // let h = (self.noise_fn)(self.seed, nx, ny);

                self.height[(j*self.w + i) as usize] = h;
                if h >= 1.0 {
                    println!("h: {}", h);
                }
                assert!(h >= 0.0 && h <= 1.0);
                if h > self.max {
                    self.max = h;
                }
            }
        }

        self.stale = false;
        println!("run took {:?}", tstart.elapsed().unwrap());
    }

    fn set_noise(&mut self, inputs: &FrameInputState) {
        if inputs.key_press_or_repeat(VirtualKeyCode::Key1) {
            self.noise_fn = fsxnoise;
            self.noise_name = "fsxnoise";
        } else if inputs.key_press_or_repeat(VirtualKeyCode::Key2) {
            self.noise_fn = sxnoise;
            self.noise_name = "sinnoise";
        } else if inputs.key_press_or_repeat(VirtualKeyCode::Key3) {
            self.noise_fn = frac_exps_noise;
            self.noise_name = "expfsxnoise";
        } else if inputs.key_press_or_repeat(VirtualKeyCode::Key4) {
            self.noise_fn = squish_noise;
            self.noise_name = "squish noise";
        } else if inputs.key_press_or_repeat(VirtualKeyCode::Key5) {
            self.noise_fn = rec_noise;
            self.noise_name = "recursive noise";
        } else if inputs.key_press_or_repeat(VirtualKeyCode::R) {

        } else {
            return;
        }
        self.t_change = inputs.t;
        self.seed += 1;
        self.stale = true;
        self.stale_mesh = true;
        println!("stale mesh");
    }

    fn h(&self, seed: u32, x: f64, y: f64) -> f64 {
        let x = x * 8.0;
        let y = y * 8.0;

        (self.noise_fn)(seed, x, y)
    }

    fn hn(&self, seed: u32, x: f64, y: f64) -> (f64, Vec3) {
        let d = 0.01;
        let h = self.h(seed, x, y);
        let hgx = self.h(seed, x, y);
        let hgy = self.h(seed, x, y);
        let vx = Vec3::new(d, hgx - h, 0.0);
        let vz = Vec3::new(0.0, hgy - h, d);
    
        let norm = vz.cross(vx);
    
        (h, norm)
    }
}

impl Demo for NoiseExplorer {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
    //     let mut change = self.edge_chance_slider.frame(inputs, outputs, inputs.screen_rect.height_child(8, 0, 10, 3));
    //     change |= self.edge_chance_fine_slider.frame(inputs, outputs, inputs.screen_rect.height_child(9, 0, 10, 3));
    //     if change {
    //         self.edge_chance = self.edge_chance_slider.curr + self.edge_chance_fine_slider.curr;
    //         self.stale = true;
    //     }

        if inputs.key_rising(VirtualKeyCode::M) {
            self.show_mesh = !self.show_mesh;
            // if self.show_mesh {
            //     self.stale_mesh = true;
            // }
        }

        self.set_noise(inputs);
        if inputs.t - self.t_change < 1.0 {
            let ca = 12.0 / 14.0;
            let ch = 0.02;
            let cw = ch * ca;
            outputs.glyphs.push_center_str(&format!("{} - seed: {}", self.noise_name, self.seed), inputs.screen_rect.w/2.0, inputs.screen_rect.h/2.0, cw, ch, 2.0, Vec4::new(1.0, 1.0, 1.0, 1.0));
        }

        if self.show_mesh {
            if self.stale_mesh {
                self.run();
                let mut mb = MeshBuilder::default();
                for i in 0..self.w {
                    for j in 0..self.h {
                        let x = i as f32 / self.w as f32;
                        let y = j as f32 / self.h as f32;
                        let z = self.height[j * self.w + i];
                        let z = z * 0.15;
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
                self.stale_mesh = false;    
            }
            
            let mt = translation(-0.5, -0.3, -0.5);
            let mr = roty(inputs.t as f32 / 5.0);
            let mm = mat4_mul(mr, mt);

            let cp = Vec3::new(1.0, 0.5, 0.0);
            let ct = Vec3::new(0.0, 0.0, 0.0);
            let cd = ct - cp;
            
            let v = view(cp, ct);
            let p = proj(1.0, inputs.screen_rect.aspect() as f32, 0.001, 100.0);
            let vp = mat4_mul(p, v);

            outputs.draw_mesh = Some((vp, mm, cp, cd));
        } else {
            if self.stale {
                self.run();
                let tw = self.w;
                let th = self.h;
                let mut t = TextureBuffer::new(tw, th);
                for i in 0..tw {
                    for j in 0..th {
                        let h = self.height[i * self.h + j];
                        let colour = Vec4::new(0.0, 0.0, 0.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), h);
    
                        t.set(i as i32, j as i32, colour);
                    }
                }
                outputs.set_texture.push((t, 0));
            }
            outputs.draw_texture.push((inputs.screen_rect, 0));
        }

 
    }
}

fn expsxnoise(seed: u32, x: f64, y: f64) -> f64 {
    let h = 4.0 * expnoise(x, y, seed);
    let h = h.sin();
    let h = (h + 1.0) / 2.0;
    assert!(h >= 0.0);
    assert!(h <= 1.0);
    h
}

fn sxnoise(seed: u32, x: f64, y: f64) -> f64 {
    let h = 2.0 * 2.0 * PI * noise2d(x, y, seed);
    let h = h.sin();
    let h = (h + 1.0) / 2.0;
    assert!(h >= 0.0);
    assert!(h <= 1.0);
    h
}

fn fsxnoise(seed: u32, x: f64, y: f64) -> f64 {
    let x = x / 4.0;
    let y = y / 4.0;
    let h =
    (1.000 * sxnoise(seed, x, y) +
    0.500 * sxnoise(seed*15915717, x*2.0, y*2.0) +
    0.250 * sxnoise(seed*65711517, x*4.0, y*4.0) +
    0.125 * sxnoise(seed*34123587, x*8.0, y*8.0)) /
    1.875;
    assert!(h >= 0.0);
    if h > 1.0 {
        println!("{}", h);
    }
    assert!(h <= 1.0);
    h
}

// what about 1/x noise or 1/x + 0.1 noise
fn fracnoise(seed: u32, x: f64, y: f64) -> f64 {
    (1.000 * noise2d(x, y, seed) +
    0.500 * noise2d(x*2.0, y*2.0, seed*1238715) +
    0.250 * noise2d(x*4.0, y*4.0, seed*9148167) +
    0.125 * noise2d(x*8.0, y*8.0, seed*2442347)) /
    1.875
}

fn frac_squish_noise(seed: u32, x: f64, y: f64) -> f64 {
    let x = x / 4.0;
    let y = y / 4.0;
    (1.000 * squish_noise(seed, x, y) +
    0.500 * squish_noise(seed*1238715, x*2.0, y*2.0) +
    0.250 * squish_noise(seed*9148167, x*4.0, y*4.0) +
    0.125 * squish_noise(seed*2442347, x*8.0, y*8.0)) /
    1.875
}

fn frac_exps_noise(seed: u32, x: f64, y: f64) -> f64 {
    let x = x / 4.0;
    let y = y / 4.0;
    (1.000 * expsxnoise(seed, x, y) +
    0.500 * expsxnoise(seed*1238715, x*2.0, y*2.0) +
    0.250 * expsxnoise(seed*9148167, x*4.0, y*4.0) +
    0.125 * expsxnoise(seed*2442347, x*8.0, y*8.0)) /
    1.875
}


fn squish_noise(seed: u32, x: f64, y: f64) -> f64 {
    let xscale = noise2d(x, y, seed * 105891957) * 0.5 + 1.0;
    let yscale = noise2d(x, y, seed * 185971237) * 0.5 + 1.0;

    // but if we just rotate x and y its gonna vary based on distance to the origin
    //


    let h = noise2d(x * xscale, y * yscale, seed * 1518517);
    let h = (h + 1.0) / 2.0;
    assert!(h >= 0.0);
    assert!(h <= 1.0);
    h
}

// fn percnoise(seed: u32, x: f64, y: f64) -> f64 {
//     percnoise_rec(seed, x, y, HashMap::new())
// }

// fn percnoise_rec(seed: u32, x: f64, y: f64, memo: HashMap<(i32, i32), f64>) -> f64 {
//     let (xfloor, xfrac) = floorfrac(x);
//     let (yfloor, yfrac) = floorfrac(y);

    
//     let x0 = xfloor as i32;
//     let y0 = yfloor as i32;
//     if let val = memo[&(x0, y0)] {
//         return val;
//     }

//     if khash(seed + x0 as u32 * 113510517) % 2 == 0 {
//         let table = [(-1.0, -1.0), (-1.0, 0.0), (-1.0, 1.0), (0.0, -1.0), (0.0, 1.0), (1.0, -1.0), (1.0, 0.0), (1.0, -1.0)];
//         let (dx, dy) = table[(khash(seed + x0 as u32 * 13515107 + y0 as u32 * 14111677) % 8) as usize];
//         let val = percnoise_rec(seed, x + dx, y + dy, memo);
//         memo[&((x + dx).floor() as i32, (y + dy).floor() as i32)] = val;
//         return val;

//         // ah shit doesnt actually stop cycles
//     }

//     let x1 = x0 + 1;
//     let y1 = y0 + 1;

//     let s00 = khash2i(x0, y0, seed);
//     let s10 = khash2i(x1, y0, seed);
//     let s01 = khash2i(x0, y1, seed);
//     let s11 = khash2i(x1, y1, seed);

//     let h00 = krand(s00);
//     let h10 = krand(s10);
//     let h01 = krand(s01);
//     let h11 = krand(s11);

//     let ptop = lerp(h00, h10, smoothstep(xfrac));
//     let pbot = lerp(h01, h11, smoothstep(xfrac));

//     lerp(ptop, pbot, smoothstep(yfrac))
// }


// could also make something that looks like clouds with this pretty easy
// vary num iters would be interesting

// i wonder if this is like domain warping
// anyway its basically follow the path

fn expnoise(x: f64, y: f64, seed: u32) -> f64 {
    // i think -ln of something 0..1 is exp dist
    -noise2d(x, y, seed).ln()
}

fn rec_noise(seed: u32, x: f64, y: f64) -> f64 {
    rec_noise_rec(10, seed, x, y)
}
fn rec_noise_rec(max: i32, seed: u32, x: f64, y: f64) -> f64 {
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
        acc = rec_noise_rec(max - 1, seed, x + dx, y + dy);
    }
    acc
}