use crate::{scene::{Demo, FrameOutputs}, kinput::FrameInputState};
use crate::kmath::*;
use crate::texture_buffer::*;

pub struct RGBThing {
    tendrils: Vec<Vec<f64>>,
    grid: Vec<Vec4>,

    w: usize,
    h: usize,

    frame: u32,
}

impl RGBThing {
    pub fn new(w: usize, h: usize, seed: u32) -> RGBThing {
        RGBThing {
            tendrils: vec![vec![]; w],
            grid: Vec::new(),
            frame: 0,
            w,
            h,
        }
    }
}

impl Default for RGBThing {
    fn default() -> Self {
        Self::new(200, 200, 2)
    }
}

impl Demo for RGBThing {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
        self.frame += 1;
        let fseed = khash(self.frame * 141417177);

        // first update tendrils
        let grow_chance = 0.1;
        let die_chance = 0.01;
        let angle_step = 5.0;

        for i in 0..self.w {
            let tseed = khash(fseed + 12312417 * i as u32);
            let roll = kuniform(tseed, 0.0, 1.0);
            if roll < die_chance {
                self.tendrils[i] = vec![];
            } else if roll > (1.0 - grow_chance) {
                let last = *self.tendrils[i].get(self.tendrils[i].len() - 1).unwrap_or(&kuniform(tseed * 12341247, 0.0, 360.0));
                self.tendrils[i].push((last + kuniform(tseed * 348960237, -angle_step, angle_step) + 360.0) % 360.0);
            }
        }

        // then do buffer thing
        let old = self.grid.clone();
        for i in 1..self.w {
            for j in 1..self.h {
                self.grid[j*self.w + i] = old[(j-1)*self.w + i-1];
            }
        }
        // then put tendrils in
        // for i in 0..
        

        // render
        let tw = self.w;
        let th = self.h;
        let mut t = TextureBuffer::new(tw, th);
        for i in 0..tw {
            for j in 0..th {
                t.set(i as i32, j as i32, self.grid[j * self.w + i]);
                // let heat = (self.heat[j * self.w + i] - self.lowest_heat) as f64 / self.highest_heat as f64;
                // t.set(i as i32, j as i32, heat * Vec4::new(1.0, 0.0, 0.0, 1.0));
            }
        }
        outputs.set_texture.push((t, 0));
        outputs.draw_texture.push((inputs.screen_rect, 0));

    }
}