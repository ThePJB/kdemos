use crate::{scene::{Demo, FrameOutputs}, kinput::FrameInputState, renderers::mesh_renderer::MeshBuilder};
use crate::kmath::*;
use crate::texture_buffer::*;
use glutin::event::VirtualKeyCode;


// tiles have a fuel value
// fuel hits 0: black
// gotta be buds of new growth
// ignition chance
// fire spreads based on fuel

pub struct Wildfire {
    fuel: Vec<f32>,
    on_fire: Vec<bool>,
    height: Vec<f64>,
    fertility: Vec<f64>,


    w: usize,
    h: usize,

    stale: bool,

    seed: u32,

}

fn f_fn(x: f64, y: f64, seed: u32) -> f64 {
    (1.000 * noise2d(x / 71.0, y / 73.0, seed*5387772) +
    0.500 * noise2d(x / 52.0, y / 35.0, seed*7772441)) / 1.5
}

fn h_fn(x: f64, y: f64, seed: u32) -> f64 {
    (1.000 * noise2d(x / 100.0, y / 100.0, seed*7457) +
    0.500 * noise2d(x / 50.0, y / 50.0, seed*3546346) +
    0.250 * noise2d(x / 25.0, y / 25.0, seed*734521) +
    0.125 * noise2d(x / 12.5, y / 12.5, seed*26277)) /
    1.875
}

fn ridged(x: f64, y: f64, seed: u32) -> f64 {
    2.0 * (h_fn(x, y, seed) - 0.5).abs()
}

fn h2_fn(x: f64, y: f64, seed: u32) -> f64 {
    let warp_scale = 76.0;
    let warp_mag = 50.0;

    // and do you add or mul the dx and dy

    let dx = warp_mag * noise2d(x / warp_scale, y / warp_scale, seed * 1384971237) - warp_mag/2.0;
    let dy = warp_mag * noise2d(x / warp_scale, y / warp_scale, seed * 1324712347) - warp_mag/2.0;

    ridged(x + dx, y + dy, seed * 12312357)
}

impl Default for Wildfire {
    fn default() -> Self {
        let mut w = Wildfire { 
            fuel: vec![],
            on_fire: vec![],
            height: vec![],
            fertility: vec![],
            w: 400,
            h: 400,
            stale: false,
            seed: 0,
        };
        w.gen();
        w
    }
}

impl Wildfire {
    fn gen(&mut self) {
        self.seed = (self.seed + 123124817) * 92351709;
        self.height = vec![0.0; self.w  * self.h];
        self.fertility = vec![0.0; self.w  * self.h];
        self.fuel = vec![0.0; self.w  * self.h];
        self.on_fire = vec![false; self.w  * self.h];
        
        for i in 0..self.w {
            for j in 0..self.h {
                self.height[j*self.w + i] = h2_fn(i as f64, j as f64, self.seed);
            }
        }
        for i in 0..self.w {
            for j in 0..self.h {
                self.fertility[j*self.w + i] = f_fn(i as f64, j as f64, self.seed);
            }
        }
        self.stale = true;
    }
}

impl Demo for Wildfire {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
        if inputs.key_rising(VirtualKeyCode::R) {
            self.gen();
        }

        // grow more fuel
        for i in 0..self.fuel.len() {
            if chance(i as u32 * 123177 + inputs.seed, 0.005 * 2.0 * self.fertility[i]) {
                self.fuel[i] += 2.0;
            }
        }

        // ignite
        for i in 0..self.on_fire.len() {
            if chance(i as u32 * 234234181 + inputs.seed, 0.0000001) {
                self.on_fire[i] = true;
            }
        }

        let mut order: Vec<usize> = (0..self.w*self.h).collect();
        for i in 0..order.len() {
            let swap_idx = khash(inputs.seed + i as u32 * 2398402317) % (order.len() as u32 - i as u32) + i as u32;
            order.swap(i, swap_idx as usize);
        }

        let p_hi = 0.3;
        let p_lo = 0.02;

        for idx in order {
            if self.on_fire[idx] {
                if self.fuel[idx] > 0.0 {
                    self.fuel[idx] = (self.fuel[idx] - 1.0).max(0.0);
                    // spread up
                    if idx > self.w {
                        let p = if self.height[idx - self.w] > self.height[idx] {
                            p_hi
                        } else {
                            p_lo
                        };
                        if chance(idx as u32 * 1237171797 + inputs.seed, p) {
                            self.on_fire[idx - self.w] = true;
                        }
                    }

                    // spread down
                    if idx < self.w*(self.h - 1) {
                        let p = if self.height[idx + self.w] > self.height[idx] {
                            p_hi
                        } else {
                            p_lo
                        };
                        if chance(idx as u32 * 1231241877 + inputs.seed, p) {
                            self.on_fire[idx + self.w] = true;
                        }
                        
                    }

                    // spread left
                    if idx % self.w != 0 {
                        let p = if self.height[idx - 1] > self.height[idx] {
                            p_hi
                        } else {
                            p_lo
                        };
                        if chance(idx as u32 * 534343257 + inputs.seed, p)  {
                            self.on_fire[idx - 1] = true;
                        }
                    }

                    // spread right
                    if (idx + 1) % self.w  != 0 {
                        let p = if self.height[idx + 1] > self.height[idx] {
                            p_hi
                        } else {
                            p_lo
                        };
                        if chance(idx as u32 * 131254717 + inputs.seed, p) {
                            self.on_fire[idx + 1] = true;
                        }
                    }
                } else {
                    self.on_fire[idx] = false;
                }
            }
        }

        let c_ne = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let c_nf = Vec4::new(0.0, 1.0, 0.0, 1.0);
        let c_fe = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let c_ff = Vec4::new(1.0, 1.0, 1.0, 1.0);

        let mut tb = TextureBuffer::new(self.w, self.h);
        for i in 0..self.w {
            for j in 0..self.h {
                let idx = j*self.w + i;
                let c = if self.on_fire[idx] {
                    c_fe.lerp(c_ff, (self.fuel[idx] / 50.0).min(1.0) as f64)
                } else {
                    c_ne.lerp(c_nf, (self.fuel[idx] / 50.0).min(1.0) as f64)
                };
                
                tb.set(i as i32, j as i32, c);
            }
        }

        if self.stale {
            let mut mb = MeshBuilder::default();


            for i in 0..self.w {
                for j in 0..self.h {
                    let x = i as f32 / self.w as f32;
                    let y = j as f32 / self.h as f32;
                    let z = 0.1 * self.height[j * self.w + i];
                    let u = i as f32 / self.w as f32;
                    let v = j as f32 / self.h as f32;

                    let pos = Vec3::new(x as f64, z as f64, y as f64);
                    let uv = Vec2::new(u as f64, v as f64);
                    let normal = Vec3::new(0., 0., 0.);
                    let colour = Vec4::new(1.0, 1.0, 1.0, 1.0);

                    mb.push_element(pos, uv, normal, colour);
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


        outputs.set_mesh_texture = Some(tb);
        outputs.draw_mesh = Some((vp, mm, cp, cd));
    }
}