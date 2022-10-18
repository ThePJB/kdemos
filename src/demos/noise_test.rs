use crate::scene::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use crate::kinput::*;
use glutin::event::VirtualKeyCode;

pub struct NoiseTest {
    w: usize,
    h: usize,

    seed: u32,

    stale: bool,

    grid: Vec<f64>,
    max: f64,
}

impl Default for NoiseTest {
    fn default() -> Self {
        Self::new(800, 800)
    }
}

impl NoiseTest {
    pub fn new(w: usize, h: usize) -> NoiseTest {
        NoiseTest {
            w,
            h,
            seed: 69,
            stale: true,
            grid: vec![0.0; w*h],
            max: 0.0,
        }
    }

    pub fn run_perc(&mut self) {
        let tstart = std::time::SystemTime::now();
        self.max = 0.0;

        
        for i in 0..self.w {
            for j in 0..self.h {
                let nx = 32.0 * i as f64 / self.w as f64;
                let ny = 32.0 * j as f64 / self.h as f64;
                //let h = noise2d(nx, ny, self.seed + 12341237);
                let h = rec_noise(10, self.seed, nx, ny);

                self.grid[(j*self.w + i) as usize] = h;
                if h > self.max {
                    self.max = h;
                }
            }
        }

        self.stale = false;
        println!("run perc took {:?}", tstart.elapsed().unwrap());
    }
}

impl Demo for NoiseTest {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
    //     let mut change = self.edge_chance_slider.frame(inputs, outputs, inputs.screen_rect.grid_child(8, 0, 10, 3));
    //     change |= self.edge_chance_fine_slider.frame(inputs, outputs, inputs.screen_rect.grid_child(9, 0, 10, 3));
    //     if change {
    //         self.edge_chance = self.edge_chance_slider.curr + self.edge_chance_fine_slider.curr;
    //         self.stale = true;
    //     }

        if inputs.key_rising(VirtualKeyCode::R) {
            self.seed += 1;
            self.stale = true;
        }
        
        if self.stale {
            self.run_perc();
            let tw = self.w;
            let th = self.h;
            let mut t = TextureBuffer::new(tw, th);
            for i in 0..tw {
                for j in 0..th {
                    let h = self.grid[i * self.h + j];
                    let h = h / self.max;
                    let colour = Vec4::new(0.0, 0.0, 0.0, 1.0).lerp(Vec4::new(1.0, 1.0, 1.0, 1.0), h);

                    t.set(i as i32, j as i32, colour);
                }
            }
            outputs.set_texture.push((t, 0));

        }
 
        outputs.draw_texture.push((inputs.screen_rect, 0));
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

// dude imagine how trippy if the underlying noise moved: would have to render to a video, way too slow


// could also make something that looks like clouds with this pretty easy
// vary num iters would be interesting

// i wonder if this is like domain warping
// anyway its basically follow the path