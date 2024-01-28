use crate::{scene::{Demo, FrameOutputs}, kinput::FrameInputState};
use crate::kmath::*;
use crate::texture_buffer::*;
use glutin::event::VirtualKeyCode;


// this could so easily be a map for a game. different walkers, walkers have rules when they meet certain conditions eg walk on another walkers tile, you die, or swap to going straight
// what about a UTM with made up rules, like from a given colour you set the colour to something and set the tile to something and move in a certain direction. might start getting weird patterns
// seed = r ^ g ^ b or r << 16 | g << 8 | b and dir is khash(seed), head colour is khash(seed) and tile colour is head colour. id expect it to find cycles
// does this look like percolation? percolation looks like a map for a game
// give the walker a tendency to maintain its velocity, or let it move 8ways

// stay clean: every thousand iters reset to start

// allow it to start self organizing, or is there any way of staling shit, or whats the distribution of tiles? or what if just randomized the direction say
// or chance to keep colour the same?


// and whats the fate, one colour?

// or try new rules minimizing number of colour changes

// i guess an oscillator would be good
// how does the size of the colour space change it, getting loops must be crucial

// the thing is that entropy eventually wins, how do you create order. if the rulews change too

// how many states: explicit number? rather than colours
// who will carry the torch of order: cleanup mode engaged if all squares different
// maybe the behaviour is predictable based on the number of states and their relation
// or if multiple states go back to another one

// programs that randomly self modify, you could give it a goal, it could be to withstand bit flips
// or its goal is to replicate, or to replicate and survive bit flips

// could make one that doesnt go backwards to generate one pattern based onsay only its left and up squares

// what about a de noising agent that considers 3x3 and sets it to one of them based on the 3x3

// come on if this had sound
// 8conn interesting?

// what if some structures dont get rewritten

// every 100 back to the start

// option to stay same colour?

// choose the colour its seen the most or the colour its seen the least?

// or if going off the edge teleports you to the middle

// what about certain colours are immutable?

// origin version

pub struct RGBUTM {
    grid: Vec<[u8; 3]>,
    w: usize,
    h: usize,

    head_colour: [u8; 3],
    head_x: i32,
    head_y: i32,

    steps_per_frame: usize,

    seed: u32,
}

impl Default for RGBUTM {
    fn default() -> Self {
        Self::new(1584/4, 396/4, 129837123)
    }
}

impl RGBUTM {
    pub fn new(w: usize, h: usize, seed: u32) -> RGBUTM {
        let head_colour = [0, 255, 255];

        let head_x = 0;
        let head_y = 0;

        let base_colour = [khash(seed + 1712317) as u8, khash(seed + 1231247) as u8, khash(seed + 123123177) as u8];
        let grid = vec![base_colour; w*h];

        RGBUTM {
            grid,
            w,
            h,
            seed,
            head_colour,
            head_x,
            head_y,
            steps_per_frame: 200,
        }
    }
}

impl Demo for RGBUTM {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {

        if inputs.key_rising(VirtualKeyCode::R) {
            *self = RGBUTM::new(self.w, self.h, self.seed + 1);
        }

        for _ in 0..self.steps_per_frame {
            let tile_value = self.grid[(self.w as i32 * self.head_y + self.head_x) as usize];
            let tile_seed = (tile_value[0] as u32) << 16 | (tile_value[1] as u32) << 8 | tile_value[0] as u32;
            // let tile_seed = (tile_value[0] ^ tile_value[1] ^ tile_value[2]) as u32;
            // let tile_seed = tile_seed & 0x07;
            let tile_seed = tile_seed * (self.seed + 12381177) ;
            self.grid[(self.w as i32 * self.head_y + self.head_x) as usize] = self.head_colour;
            let new_colour = [khash(tile_seed + 123177) as u8, khash(tile_seed + 11975) as u8, khash(tile_seed + 19471357) as u8];
            self.head_colour = new_colour;
            


            // let (dx, dy) = [(1, 0), (0, 1)][(khash(tile_seed + 132847177) % 2) as usize];
            // let (dx, dy) = [(-1, 0), (1, 0), (0, -1), (0, 1)][(khash(tile_seed + 132847177) % 4) as usize];
            let (dx, dy) = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)][(khash(tile_seed + 132847177) % 8) as usize];
            self.head_x = (self.head_x + dx + self.w as i32) % self.w as i32;
            self.head_y = (self.head_y + dy + self.h as i32) % self.h as i32;
        }

        // render
        let tw = self.w;
        let th = self.h;
        let mut t = TextureBuffer::new(tw, th);
        for i in 0..tw {
            for j in 0..th {
                let c = self.grid[j * self.w + i];
                
                t.set(i as i32, j as i32, Vec4::new(c[0] as f64 / 255.0, c[1] as f64 / 255.0, c[2] as f64 / 255.0, 1.0));
            }
        }
        outputs.set_texture.push((t, 0));
        outputs.draw_texture.push((inputs.screen_rect, 0));

    }
}