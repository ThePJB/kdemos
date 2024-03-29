use std::f32::INFINITY;

use crate::scene::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use crate::kinput::*;
use glutin::event::VirtualKeyCode;
use ordered_float::OrderedFloat;

pub struct SeaOfBlood {
    w: usize,
    h: usize,

    seed: u32,

    stale: bool,
}

impl Default for SeaOfBlood {
    fn default() -> Self {
        Self::new(400, 400)
    }
}

impl SeaOfBlood {
    pub fn new(w: usize, h: usize) -> SeaOfBlood {
        SeaOfBlood {
            w,
            h,
            seed: 69,
            stale: true,
        }
    }
}

impl Demo for SeaOfBlood {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
        let max_layer = 10;
        let colours: Vec<Vec4> = (0..max_layer).map(|layer| 0.1 + layer as f64 / max_layer as f64).map(|r| Vec4::new(r, 0.0, 0.0, 1.0)).rev().collect();
        
        let heights: Vec<f32> = (0..max_layer)
            .map(|x| x as f32 / max_layer as f32)// 1 - 0
            .map(|x| (1.0 - x)*x*0.2 + x * 0.2 + 0.2)
            .collect();

        let mut t = TextureBuffer::new(self.w, self.h);
        for layer in (0..10).rev() {
            for i in 0..self.w {
                for j in 0..self.h {
                    let x = i as f32 / self.w as f32;
                    let y = j as f32 / self.h as f32;
                    // let y = y + 0.1 * (10 - layer) as f32;

                    // let phase = 5.0 * noise1d(inputs.t as f32 - 0.05*layer as f32, self.seed * 1248157) + 20.0 * x + 4.0 * inputs.t as f32;
                    let phase = inputs.t as f32 + (2.0*PI as f32/10.0) * layer as f32 *(inputs.t as f32).sin() + 30.0 * x;

                    if y < heights[layer] + 0.0707* phase.sin() {
                        t.set(i as i32, j as i32, colours[layer]);
                    }
                }
            }
        }

        outputs.set_texture.push((t, 0));
        outputs.draw_texture.push((inputs.screen_rect, 0));
    }
}


fn lerp(x1: f32, x2: f32, t: f32) -> f32 {
    x1 * (1.0 - t) + x2 * t
}
fn rand(seed: u32) -> f32 {
    khash(seed) as f32 / 4294967295.0
}
pub fn floorfrac(x: f32) -> (f32, f32) {
    let floor = x.floor();
    if x < 0.0 {
        (floor, (floor - x).abs())
    } else {
        (floor, x - floor)
    }
}
pub fn smoothstep(t: f32) -> f32 {
    t * t * (3. - 2. * t)
}
pub fn noise1d(t: f32, seed: u32) -> f32 {
    let hstart = rand(seed + 489172373 * t.floor() as u32);
    let hend = rand(seed + 489172373 * (t.floor() + 1.0) as u32);
    lerp(hstart, hend, smoothstep(t.fract()))
}
fn noise2d(x: f32, y: f32, seed: u32) -> f32 {
    let (xfloor, xfrac) = floorfrac(x);
    let (yfloor, yfrac) = floorfrac(y);

    let x0 = xfloor as i32;
    let x1 = x0 + 1;
    let y0 = yfloor as i32;
    let y1 = y0 + 1;

    let s00 = khash2i(x0, y0, seed);
    let s10 = khash2i(x1, y0, seed);
    let s01 = khash2i(x0, y1, seed);
    let s11 = khash2i(x1, y1, seed);

    let h00 = rand(s00);
    let h10 = rand(s10);
    let h01 = rand(s01);
    let h11 = rand(s11);

    let ptop = lerp(h00, h10, smoothstep(xfrac));
    let pbot = lerp(h01, h11, smoothstep(xfrac));

    lerp(ptop, pbot, smoothstep(yfrac))
}
fn worley(x: f32, y: f32, seed: u32) -> f32 {
    let (xfloor, xfrac) = floorfrac(x);
    let (yfloor, yfrac) = floorfrac(y);

    let xvalues = [xfloor - 1.0, xfloor - 1.0, xfloor - 1.0, xfloor, xfloor, xfloor, xfloor + 1.0, xfloor + 1.0, xfloor + 1.0];
    let yvalues = [yfloor - 1.0, yfloor, yfloor + 1.0, yfloor - 1.0, yfloor + 1.0, yfloor, yfloor - 1.0, yfloor, yfloor + 1.0];
    let mut px = [0.0; 9];
    let mut py = [0.0; 9];
    for i in 0..9 {
        let si = khash2i(xvalues[i] as i32, yvalues[i] as i32, seed);
        px[i] = xvalues[i] + rand(si);
        py[i] = yvalues[i] + rand(si.wrapping_mul(1234125417));
    }
    let mut mind = INFINITY;
    for i in 0..9 {
        let d2 = (px[i] - x)*(px[i] - x) + (py[i] - y)*(py[i] - y);
        if d2 < mind {
            mind = d2;
        }
    }

    mind

    // consider 3x3 cells around and each one has a point
    // height = minimum distance to a point from this location
}
fn frac_noise(x: f32, y: f32, seed: u32) -> f32 {
    1.000 * noise2d(x, y, seed) +
    0.500 * noise2d(x*2.0, y*2.0, seed.wrapping_mul(1238715)) +
    0.250 * noise2d(x*4.0, y*4.0, seed.wrapping_mul(9148167)) +
    0.125 * noise2d(x*8.0, y*8.0, seed.wrapping_mul(2442347)) /
    1.875
}
fn ridge_noise(x: f32, y: f32, seed: u32) -> f32 {
    (frac_noise(x, y, seed) - 0.5).abs() * 2.0
}
fn walkable(x: f32, y: f32, seed: u32) -> f32 {
    worley(8.0 * x, 8.0 * y, seed)
    // morely(8.0 * x, 8.0 * y, seed)
    // let n = ridge_noise(8.0 * x, 8.0 * y, seed);
    // if n < 0.1 {
    //     1.0
    // } else {
    //     0.0
    // }
}

/*
Worley Variations
 * paths where distance of 2 < threshold
 *  might make a crazy lattice. could constrain it to be the distance between 2 closest
 * nodes where distance of 3 < threshold
 * different distance metric (this is nth)
 * nth closest not just closest
 * worley with dropout
 * frac worley noise


lets not lose sight of what we were doing which was creating a wizrad level
implicit voronoi graph thing basically 
*/

fn morely(x: f32, y: f32, seed: u32) -> f32 {
    let (xfloor, xfrac) = floorfrac(x);
    let (yfloor, yfrac) = floorfrac(y);

    let xvalues = [xfloor - 1.0, xfloor - 1.0, xfloor - 1.0, xfloor, xfloor, xfloor, xfloor + 1.0, xfloor + 1.0, xfloor + 1.0];
    let yvalues = [yfloor - 1.0, yfloor, yfloor + 1.0, yfloor - 1.0, yfloor + 1.0, yfloor, yfloor - 1.0, yfloor, yfloor + 1.0];
    let mut px = [0.0; 9];
    let mut py = [0.0; 9];
    for i in 0..9 {
        let si = khash2i(xvalues[i] as i32, yvalues[i] as i32, seed);
        px[i] = xvalues[i] + rand(si);
        py[i] = yvalues[i] + rand(si.wrapping_mul(1234125417));
    }
    let mut d = [0.0; 9];
    for i in 0..9 {
        d[i] = (px[i] - x)*(px[i] - x) + (py[i] - y)*(py[i] - y);
    }
    let mut min = 0;
    let mut mind = INFINITY;

    for i in 0..9 {
        for i in 0..9 {
            if d[i] < mind {
                mind = d[i];
                min = i;
            }
        }
    }
    let mut acc = 0.0;
    for i in 0..9 {
        for j in 0..9 {
            if i == j {continue;}
            if (min == i || min == j) && (d[i] - d[j]).abs() < 0.05 {
                acc += 0.1;
            }
        }
    }
    acc


    // let mut mind = INFINITY;
    // for i in 0..9 {
    //     let d2 = (px[i] - x)*(px[i] - x) + (py[i] - y)*(py[i] - y);
    //     for j in 0..9 {
    //         // if the distance is ever equal thats good
    //     }
    //     let d2 = (px[i] - x)*(px[i] - x) + (py[i] - y)*(py[i] - y);
    //     if d2 < mind {
    //         mind = d2;
    //     }
    // }

    // mind
}

fn worley3(x: f32, y: f32, t: f32, seed: u32) -> [f32; 3] {
    let (xfloor, xfrac) = floorfrac(x);
    let (yfloor, yfrac) = floorfrac(y);

    let xvalues = [xfloor - 1.0, xfloor - 1.0, xfloor - 1.0, xfloor, xfloor, xfloor, xfloor + 1.0, xfloor + 1.0, xfloor + 1.0];
    let yvalues = [yfloor - 1.0, yfloor, yfloor + 1.0, yfloor - 1.0, yfloor + 1.0, yfloor, yfloor - 1.0, yfloor, yfloor + 1.0];
    let mut px = [0.0; 9];
    let mut py = [0.0; 9];
    for i in 0..9 {
        let si = khash2i(xvalues[i] as i32, yvalues[i] as i32, seed);
        px[i] = xvalues[i] + noise1d(0.3 * t + rand(si * 1414197), si);
        py[i] = yvalues[i] + noise1d(0.3 * t + rand(si * 1324157), si.wrapping_mul(1234125417));

    }
    let mut d = [(0, 0.0); 9];
    for i in 0..9 {
        d[i] = (i, (px[i] - x)*(px[i] - x) + (py[i] - y)*(py[i] - y));
    }

    d.sort_by_key(|x| OrderedFloat(x.1));
    return [d[0].1, d[1].1, d[2].1];
}