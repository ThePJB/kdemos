use crate::scene::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use crate::kinput::*;
use glutin::event::VirtualKeyCode;

pub struct MDJ {
    w: usize,
    h: usize,
    r: Rect,
    buf: Vec<i32>,
    bufj: Vec<i32>,
    colour_palette: Vec<Vec4>,

    stale_normal: bool,
    stale_julia: bool,

    path_c: Option<Vec2>,

    julia_point: Vec2,
    jr: Rect,
}

const MAX_ITERATIONS: i32 = 160;

impl Default for MDJ {
    fn default() -> Self {
        Self::new(800, 800)
    }
}

impl MDJ {
    pub fn new(w: usize, h: usize) -> MDJ {
        let mut colour_palette = Vec::new();



        colour_palette.push(Vec4::new(0.0, 0.0, 0.0, 1.0));

        let start = Vec4::new(1.0, 0.4, 0.0, 1.0);
        let middle = Vec4::new(0.9, 0.0, 0.7, 1.0);
        let end = Vec4::new(0.9, 0.0, 0.7, 1.0);



        // let start = Vec4::new(1.0, 0.0, 0.4, 1.0);
        let end = Vec4::new(0.9, 0.0, 0.7, 1.0);
        let start = Vec4::new(1.0, 0.4, 0.0, 1.0);
        // let end = Vec4::new(0.9, 0.7, 0.0, 1.0);
        for i in 0..MAX_ITERATIONS/2 {
            colour_palette.push(start.lerp(end, i as f64/(MAX_ITERATIONS/2) as f64));
        };

        
        let start = Vec4::new(0.9, 0.0, 0.7, 1.0);
        // let start = Vec4::new(0.9, 0.7, 0.0, 1.0);


        let end = Vec4::new(0.2, 0.7, 0.5, 1.0);
        for i in 0..MAX_ITERATIONS/2 {
            colour_palette.push(start.lerp(end, i as f64/(MAX_ITERATIONS/2) as f64));
        };

        let mut x = MDJ {
            w,
            h,
            r: Rect::new(-2.0, -1.5, 3.0, 3.0),
            jr: Rect::new(-1.5, -1.5, 3.0, 3.0),
            stale_normal: true,
            stale_julia: true,
            buf: Vec::new(),
            bufj: Vec::new(),
            colour_palette: colour_palette,
            path_c: None,
            julia_point: Vec2::new(0.0, 0.0),
        };
        x
    }

    pub fn compute_normal(&mut self) {
        let tstart = std::time::SystemTime::now();

        self.buf = vec![0; self.w*self.h];

        for i in 0..self.w {
            for j in 0..self.h {
                // convert to float (im) for each pixel coordinate
                let mut it = 0;

                let x0 = self.r.left() + (i as f64 + 0.5) * self.r.w / self.w as f64;
                let y0 = -self.r.bot() + (j as f64 + 0.5) * self.r.h / self.h as f64;
                let y0 = -y0;


                let c = Vec2::new(x0, y0);
                let mut z = Vec2::new(0.0, 0.0);
                while z.x * z.x + z.y * z.y < 4.0 && it < MAX_ITERATIONS {
                    // z = Vec2::new(z.x.abs(), z.y.abs());
                    z = z.complex_mul(z) + c;
                    it += 1;
                }

                self.buf[i * self.h + j] = it;
            }
        }

        println!("compute took {:?}", tstart.elapsed().unwrap());
    }

    pub fn compute_julia(&mut self) {
        let tstart = std::time::SystemTime::now();

        self.bufj = vec![0; self.w*self.h];

        for i in 0..self.w {
            for j in 0..self.h {
                // convert to float (im) for each pixel coordinate
                let mut it = 0;

                let x0 = self.jr.left() + (i as f64 + 0.5) * self.jr.w / self.w as f64;
                let y0 = -self.jr.bot() + (j as f64 + 0.5) * self.jr.h / self.h as f64;
                let y0 = -y0;

                let c = Vec2::new(self.julia_point.x, self.julia_point.y);
                let mut z = Vec2::new(x0, y0);
                while z.x * z.x + z.y * z.y < 4.0 && it < MAX_ITERATIONS {
                    // z = Vec2::new(z.x.abs(), z.y.abs());
                    z = z.complex_mul(z) + c;
                    it += 1;
                }

                self.bufj[i * self.h + j] = it;
            }
        }

        println!("compute julia took {:?}", tstart.elapsed().unwrap());
    }
}

impl Demo for MDJ {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {    
        let r = inputs.screen_rect.fit_aspect_ratio(2.0);
        let (r, jr) = r.split_lr(0.5);

        if inputs.key_falling(VirtualKeyCode::R) {
            *self = MDJ::new(self.w, self.h);
        }
        if inputs.lmb == KeyStatus::JustPressed && inputs.key_held(VirtualKeyCode::LShift) {
            if r.contains(inputs.mouse_pos) {
                let rp = inputs.mouse_pos.transform(r, self.r);
                self.r = Rect::new_centered(rp.x, rp.y, self.r.w*0.5, self.r.h*0.5);
                self.stale_normal = true;
            } else if jr.contains(inputs.mouse_pos) {
                let rp = inputs.mouse_pos.transform(jr, self.jr);
                self.jr = Rect::new_centered(rp.x, rp.y, self.jr.w*0.5, self.jr.h*0.5);
                self.stale_julia = true;
            }
        } else if inputs.lmb == KeyStatus::JustPressed && inputs.key_held(VirtualKeyCode::LControl) {
            if r.contains(inputs.mouse_pos) {
                let rp = inputs.mouse_pos.transform(r, self.r);
                self.r = Rect::new_centered(rp.x, rp.y, self.r.w*2.0, self.r.h*2.0);
                self.stale_normal = true;
            } else if jr.contains(inputs.mouse_pos) {
                let rp = inputs.mouse_pos.transform(jr, self.jr);
                self.jr = Rect::new_centered(rp.x, rp.y, self.jr.w*2.0, self.jr.h*2.0);
                self.stale_julia = true;
            }
        } else if (inputs.lmb == KeyStatus::Pressed && !inputs.key_held(VirtualKeyCode::LShift) && !inputs.key_held(VirtualKeyCode::LControl)) || inputs.lmb == KeyStatus::JustPressed {
            // self.path_c = Some(inputs.mouse_pos.transform(inputs.screen_rect, self.r));
            if r.contains(inputs.mouse_pos) {
                let rp = inputs.mouse_pos.transform(r, self.r);
                self.julia_point = rp;
                println!("julia point set to {:?}", self.julia_point);
                self.stale_julia = true;
            }
        }

        if self.stale_normal {
            self.compute_normal();
            let tw = self.w;
            let th = self.h;
            let mut t = TextureBuffer::new(tw, th);
            for i in 0..tw {
                for j in 1..th {
                    let it = self.buf[i * self.h + j];

                    let colour = self.colour_palette[(MAX_ITERATIONS - it) as usize];

                    t.set(i as i32, j as i32, colour);
                }
            }
            outputs.set_texture.push((t, 0));

            self.stale_normal = false;
        }

        if self.stale_julia {
            self.compute_julia();
            let tw = self.w;
            let th = self.h;
            let mut t = TextureBuffer::new(tw, th);
            for i in 0..tw {
                for j in 1..th {
                    let it = self.bufj[i * self.h + j];

                    let colour = self.colour_palette[(MAX_ITERATIONS - it) as usize];
                    // let colour = Vec4::new(1.0, 0.0, 0.0, 1.0);

                    t.set(i as i32, j as i32, colour);
                }
            }
            outputs.set_texture.push((t, 1));

            self.stale_julia = false;
        }

        if let Some(path_c) = self.path_c {
            let mut zold = Vec2::new(0.0, 0.0);
            for _ in 0..100 {
                let znew = Vec2::new(zold.x.abs(), zold.y.abs());
                let znew = znew.complex_mul(znew) + path_c;
    
                // line start and end - transform to where canvas is. from self.r to screen rect
                let start = zold.transform(self.r, inputs.screen_rect);
                let end = znew.transform(self.r, inputs.screen_rect);
    
    
                outputs.canvas.put_line(start, end, 0.002, 2.0, Vec4::new(1.0, 0.0, 0.0, 1.0));
                zold = znew;
            }
        }
        
        // // axes
        // let xstart = Vec2::new(-2.0, 0.0).transform(self.r, inputs.screen_rect);
        // let xend = Vec2::new(1.0, 0.0).transform(self.r, inputs.screen_rect);
        // outputs.canvas.put_line(xstart, xend, 0.001, 2.0, Vec4::new(0.8, 0.8, 0.8, 1.0));
        // let ystart = Vec2::new(0.0, -1.0).transform(self.r, inputs.screen_rect);
        // let yend = Vec2::new(0.0, 1.0).transform(self.r, inputs.screen_rect);
        // outputs.canvas.put_line(ystart, yend, 0.001, 2.0, Vec4::new(0.8, 0.8, 0.8, 1.0));


 
        outputs.draw_texture.push((r, 0));
        outputs.draw_texture.push((jr, 1));
    }
}