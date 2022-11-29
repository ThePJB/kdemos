use crate::scene::*;
use crate::kmath::*;
use crate::texture_buffer::*;
use crate::kinput::*;
use glutin::event::VirtualKeyCode;

pub static phi: f64 = 0.61803398874989484820458683436563811772030917980576286213544862270526046281890244970720720418939113748475408807538689175212663386222353693179318006076672635443338908659593958290563832266131992829026788067520876689250171169620703222104;

pub struct NewFractal {
    w: usize,
    h: usize,
    r: Rect,
    buf: Vec<i32>,
    colour_palette: Vec<Vec4>,

    stale: bool,
}

const MAX_ITERATIONS: i32 = 400;

impl Default for NewFractal {
    fn default() -> NewFractal {
        NewFractal::new(300, 300)
    }
}

impl NewFractal {
    pub fn new(w: usize, h: usize) -> NewFractal {
        let mut colour_palette = Vec::new();
        let mut period = 16;
        let mut pc = 0;
        let mut i = 0;
        colour_palette.push(Vec4::new(0.0, 0.0, 0.0, 1.0));
        while colour_palette.len() < MAX_ITERATIONS as usize {
            let colour_start = Vec4::new(137.5 * i as f64, 1.0, 1.0, 1.0).hsv_to_rgb();
            let colour_end = Vec4::new(137.5 * (i+1) as f64, 1.0, 1.0, 1.0).hsv_to_rgb();
            colour_palette.push(colour_start.lerp(colour_end, pc as f64 / period as f64));
            pc += 1;
            if pc == period {
                period *= 2;
                pc = 0;
                i += 1;
            }
        }

        let mut x = NewFractal {
            w,
            h,
            r: Rect::new(-2.0, -1.5, 3.0, 3.0),
            stale: true,
            buf: Vec::new(),
            colour_palette: colour_palette,
        };
        x.compute();
        x
    }

    pub fn compute(&mut self) {
        let tstart = std::time::SystemTime::now();

        self.buf = vec![0; self.w*self.h];

        for i in 0..self.w {
            for j in 0..self.h {
                // convert to float (im) for each pixel coordinate
                let mut it = 0;

                let x0 = self.r.left() as f64 + (i as f64 + 0.5) * self.r.w as f64 / self.w as f64;
                let y0 = -self.r.bot() as f64 + (j as f64 + 0.5) * self.r.h as f64 / self.h as f64;


                // let z = Vec2::new(x0, y0);
                // let z2 = z.complex_mul(z);


                let mut z = Vec2::new(0.0, 0.0);
                let c = Vec2::new(x0, y0);

                
                // let rot = z2.offset_r_theta(1.0, 0.01);
                
                while z.x * z.x + z.y * z.y < 4.0 && it < MAX_ITERATIONS {
                    // z = (((z.complex_mul(z) + z) * z) + z + c;

                    // its got the bits that look like interference patterns
                    // i like the living side and the dead side
                    // do one with a slider on how many times this process happens
                    // z = z.complex_mul(z).plus(z).complex_mul(z).plus(z).complex_mul(z).plus(z).complex_mul(z).plus(z).complex_mul(z).plus(z) + c;
                    // z = z.complex_mul(z).plus(z) + c;

                    // yea think more about pascals triangle vs this, (z + 1)^n


                    //oh alternating 1 and 2 was trippy as fuck
                    // lol what if we go around the roots of unity

                    // 1 pole
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.01))) + c;

                    // double pole
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.01)).complex_mul(z.plus(Vec2::new(0.00, 0.01)))) + c;

                    // opposite poles, so sick, noise triangle
                    // oh yeah the circles are the poles, too close divergence time bitch
                    z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.01)).complex_mul(z.plus(Vec2::new(0.00, -0.01)))) + c;


                    // asymmetric poles
                    // oh you get that 45 degree line by having asymmetry
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.01)).complex_mul(z.plus(Vec2::new(0.00, -0.02)))) + c;

                    // oh above was on still too. sick. infinite noise field
                    // it gets creepy because we have the shadow of the mandelbrot set in circles
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.02)).complex_mul(z.plus(Vec2::new(0.00, -0.02)))) + c;

                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, 0.5)).complex_mul(z.plus(Vec2::new(0.00, -0.5)))) + c;

                    // let root2 = 1.0 / 2.0f64.sqrt();
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, root2)).complex_mul(z.plus(Vec2::new(0.00, -root2)))) + c;
                    
                    // conjugate poles on the imaginary axis 2x silver ratio apart
                    // let phi = phi / 2.0;
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.00, phi)).complex_mul(z.plus(Vec2::new(0.00, -phi)))) + c;
                    // seems to be best with num ord = denom ord
                    
                    // // as above but with just z as numerator. thats when you get lines of circles and stuff
                    // z = z.complex_div(z.plus(Vec2::new(0.00, phi)).complex_mul(z.plus(Vec2::new(0.00, -phi)))) + c;
                    
                    // now conjugate but re comp as well
                    // a very wonky boy
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(1.0, phi)).complex_mul(z.plus(Vec2::new(-1.0, -phi)))) + c;
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(-1.0, 1.0)).complex_mul(z.plus(Vec2::new(-1.0, -1.0)))) + c;


                    // opposite poles re axis. it looks like some discretization effects happen
                    // plus its multi stage hyper giga l
                    // super cool
                    // z = z.complex_mul(z).complex_div(z.plus(Vec2::new(0.01, 0.00)).complex_mul(z.plus(Vec2::new(0.01, 0.00)))) + c;

                    // sick
                    // z = (z + Vec2::new(0.0, 0.01)).complex_div(z - Vec2::new(0.0, 0.01)) + c;
                    // z = (z - Vec2::new(0.0, 0.01)).complex_div(z + Vec2::new(0.0, 0.01)) + c;   // both together sick hyperbola thing
                    
                    // weird as fuck circle thing
                    // z = (z + Vec2::new(0.01, 0.00)).complex_div(z - Vec2::new(0.01, 0.00)) + c;

                    // the eye
                    // z = (z - Vec2::new(0.01, 0.00)).complex_div(z + Vec2::new(0.01, 0.00)) + c;

                    // z = (z + c).complex_div(-(z.plus(Vec2::new(0.0, 1.0))));

                    // z = (z.complex_mul(z) + c).complex_div(z.plus(Vec2::new(0.0, 1.0)));
                    // z = (z.complex_mul(z) + c).complex_div(z.plus(Vec2::new(0.0, -0.5)).complex_mul(z.plus(Vec2::new(0.0, 0.5))));
                    // z = (z + c).complex_div(z.plus(Vec2::new(1.0, -1.0)));


                    it += 1;
                }

                self.buf[i * self.h + j] = it;
            }
        }

        println!("compute took {:?}", tstart.elapsed().unwrap());
    }
}

impl Demo for NewFractal {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {    
        if inputs.key_falling(VirtualKeyCode::R) {
            *self = NewFractal::new(self.w, self.h);
        }
        if inputs.lmb == KeyStatus::JustPressed && inputs.key_held(VirtualKeyCode::LShift){
            let rp = inputs.mouse_pos.transform(inputs.screen_rect, self.r);
            self.r = Rect::new_centered(rp.x, rp.y, self.r.w*0.5, self.r.h*0.5);

            self.stale = true;
        } else if inputs.lmb == KeyStatus::JustPressed && inputs.key_held(VirtualKeyCode::LControl){
            let rp = inputs.mouse_pos.transform(inputs.screen_rect, self.r);
            self.r = Rect::new_centered(rp.x, rp.y, self.r.w*2.0, self.r.h*2.0);

            self.stale = true;
        } else if (inputs.lmb == KeyStatus::Pressed && !inputs.key_held(VirtualKeyCode::LShift) && !inputs.key_held(VirtualKeyCode::LControl)) || inputs.lmb == KeyStatus::JustPressed {
            let v = inputs.mouse_pos.transform(inputs.screen_rect, self.r);
        }

        if self.stale {
            self.compute();
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

            self.stale = false;
        }
        
        // axes
        let xstart = Vec2::new(-2.0, 0.0).transform(self.r, inputs.screen_rect);
        let xend = Vec2::new(1.0, 0.0).transform(self.r, inputs.screen_rect);
        outputs.canvas.put_line(xstart, xend, 0.001, 2.0, Vec4::new(0.8, 0.8, 0.8, 1.0));
        let ystart = Vec2::new(0.0, -1.0).transform(self.r, inputs.screen_rect);
        let yend = Vec2::new(0.0, 1.0).transform(self.r, inputs.screen_rect);
        outputs.canvas.put_line(ystart, yend, 0.001, 2.0, Vec4::new(0.8, 0.8, 0.8, 1.0));
 
        outputs.draw_texture.push((inputs.screen_rect, 0));
    }
}