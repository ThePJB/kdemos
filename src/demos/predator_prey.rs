use crate::{scene::{Demo, FrameOutputs}, kinput::FrameInputState};
use crate::kmath::*;
use crate::texture_buffer::*;

// pp ideas
// fetility
// prey hunger

#[derive(Clone)]
pub enum Tile {
    Predator(f64),
    Prey(f64),
    Food,
    Empty,
}

pub struct PredatorPrey {
    grid: Vec<Tile>,
    w: usize,
    h: usize,

    p_food: f64,
    e_food: f64,
    prey_e_reproduce: f64,
    
    e_prey: f64,
    pred_e_reproduce: f64,

    pred_e_decay: f64,
    prey_e_decay: f64,

    num_pred: i32,
    num_prey: i32,
    num_food: i32,
}

impl Default for PredatorPrey {
    fn default() -> Self {
        Self::new(400, 400)
    }
}

impl PredatorPrey {
    pub fn new(w: usize, h: usize) -> PredatorPrey {
        let seed = 69;
        let initial_pred = 0.001;
        let initial_prey = 0.001;
        let initial_food = 0.01;

        let mut num_pred = 0;
        let mut num_prey = 0;
        let mut num_food = 0;


        let mut grid = vec![Tile::Empty; w*h];
        for i in 0..w {
            for j in 0..h {
                if chance(seed * i as u32 * 123898157 + j as u32 * 134157177, initial_pred) {
                    grid[j*w + i] = Tile::Predator(0.5);
                    num_pred += 1;
                } else if chance(seed * i as u32 * 462872159 + j as u32 * 2095714717, initial_prey) {
                    grid[j*w + i] = Tile::Prey(0.5);
                    num_prey += 1;
                } else if chance(seed * i as u32 * 2059821357 + j as u32 * 569823497, initial_food) {
                    grid[j*w + i] = Tile::Food;
                    num_food += 1;
                }
            }
        }

        PredatorPrey {
            grid,
            w,
            h,
            p_food: 0.005,
            e_food: 0.8,
            prey_e_reproduce: 0.3,
            e_prey: 1.6,
            pred_e_reproduce: 1.5,
            pred_e_decay: 0.2,
            prey_e_decay: 0.1,
            num_food,
            num_pred,
            num_prey,
        }
    }
}

// this is fucked what was wrong with the scrambled order / single buffer again?
// just dont move predators into predators
// maybe just move into a prey if there is one

// trying to get the coolest settings:
// maybe if we adjust reproduce vs starvation
// i want them to have to co habitate. so without predation (locally) prey has to reproduce too much and starve, like algae
// trying to get empty areas

// preds could evolve to be just lethal enough

// hmm not dying but reproducing, interesting

impl Demo for PredatorPrey {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {

        // println!("pred: {} prey: {} food: {}", self.num_pred, self.num_prey, self.num_food);
        // // println!("order: {:?}", order);
        let actual_num_pred = self.grid.iter().filter(|x| match x {Tile::Predator(_) => true, _ => false}).count();
        let actual_num_prey = self.grid.iter().filter(|x| match x {Tile::Prey(_) => true, _ => false}).count();
        let actual_num_food = self.grid.iter().filter(|x| match x {Tile::Food => true, _ => false}).count();
        // println!("apred: {}, aprey: {}, afood: {}", actual_num_pred, actual_num_prey, actual_num_food);
 
        let mut nmo = [0, 0, 0, 0, 0];


        // nothing happens 
        // not drawing any preds or preys
        // no food growing

        // maybe im indexing wrong or w is a problwem or something

        let mut order: Vec<usize> = (0..self.w*self.h).collect();
        for i in 0..order.len() {
            let swap_idx = khash(inputs.seed + i as u32 * 2398402317) % (order.len() as u32 - i as u32) + i as u32;
            order.swap(i, swap_idx as usize);
        }

        // predators move
        for idx in order.iter() {
            let idx = *idx;
            match self.grid[idx] {
                Tile::Predator(pred_energy) => {
                    let pred_energy = pred_energy - self.pred_e_decay * inputs.dt as f64;
                    let mut move_options = Vec::new();
                    if idx > self.w {
                        move_options.push(idx - self.w);
                    }
                    if idx < self.w*self.h - self.w {
                        move_options.push(idx + self.w);
                    }
                    if idx % self.w != 0 {
                        move_options.push(idx - 1);
                    }
                    if (idx + 1) % self.w != 0 {
                        move_options.push(idx + 1);
                    }
                    // move_options.drain_filter(|cidx| match self.grid[*cidx] {
                    //     Tile::Predator(_) => true,
                    //     Tile::Food => true,
                    //     _ => false,
                    // });
                    nmo[move_options.len()] += 1;
        
                    let dest_idx = if move_options.len() == 0 {
                        idx
                    } else {
                        move_options[(khash(inputs.seed * 1983717 + idx as u32 * 12387197) % move_options.len() as u32) as usize]
                    };
                    match self.grid[dest_idx] {
                        Tile::Prey(_) => {
                            let new_energy = pred_energy + self.e_prey;
                            if new_energy > self.pred_e_reproduce {
                                self.grid[dest_idx] = Tile::Predator(new_energy/2.0);
                                self.grid[idx] = Tile::Predator(new_energy/2.0);
                                self.num_pred += 1;
                                self.num_prey -= 1;
                            } else {
                                self.grid[idx] = Tile::Empty;
                                self.grid[dest_idx] = Tile::Predator(new_energy);
                                self.num_prey -= 1;
                            }
                        }
                        _ => {
                            self.grid[idx] = Tile::Empty;
                            if pred_energy > 0.0 {
                                self.grid[dest_idx] = Tile::Predator(pred_energy);
                            } else {
                                self.num_pred -= 1;
                            }
                        },
                    }
                },
                _ => {},
            }
            
        }

        
        // preys move
        for idx in order.iter() {
            let idx = *idx;
            match self.grid[idx] {
                Tile::Prey(prey_energy) => {
                    let prey_energy = prey_energy - self.prey_e_decay * inputs.dt as f64;
                    let mut move_options = Vec::new();
                    if idx > self.w {
                        move_options.push(idx - self.w);
                    }
                    if idx < self.w*self.h - self.w {
                        move_options.push(idx + self.w);
                    }
                    if idx % self.w != 0 {
                        move_options.push(idx - 1);
                    }
                    if (idx + 1) % self.w != 0 {
                        move_options.push(idx + 1);
                    }
                    // move_options.drain_filter(|cidx| match self.grid[*cidx] {
                    //     Tile::Predator(_) => true,
                    //     Tile::Prey(_) => true,
                    //     _ => false,
                    // });
                    nmo[move_options.len()] += 1;
        
                    let dest_idx = if move_options.len() == 0 {
                        idx
                    } else {
                        move_options[(khash(inputs.seed * 1983717 + idx as u32 * 12387197) % move_options.len() as u32) as usize]
                    };
                    
                    match self.grid[dest_idx] {
                        Tile::Food => {
                            let new_energy = prey_energy + self.e_food;
                            if new_energy > self.prey_e_reproduce {
                                self.grid[dest_idx] = Tile::Prey(new_energy/2.0);
                                self.grid[idx] = Tile::Prey(new_energy/2.0);
                                self.num_prey += 1;
                                self.num_food -= 1;
                            } else {
                                self.grid[idx] = Tile::Empty;
                                self.grid[dest_idx] = Tile::Prey(new_energy);
                            }
                        }
                        _ => {
                            self.grid[idx] = Tile::Empty;
                            if prey_energy > 0.0 {
                                self.grid[dest_idx] = Tile::Prey(prey_energy);
                            } else {
                                self.num_prey -= 1;
                            }
                        },
                    }
                },
                _ => {},
            }
        }

        
        // food might appear
        for idx in order.iter() {
            let idx = *idx;
            match self.grid[idx] {
                Tile::Empty => {
                    if chance(inputs.seed * 129837715 + idx as u32 * 91238754, inputs.dt as f64 * self.p_food) {
                        self.grid[idx] = Tile::Food;
                        self.num_food += 1;
                    } else {
                    }
                }
                _ => {},
            }
        }

        // render
        let tw = self.w;
        let th = self.h;
        let mut t = TextureBuffer::new(tw, th);
        for i in 0..tw {
            for j in 0..th {
                t.set(i as i32, j as i32, match self.grid[j * self.w + i] {
                    Tile::Empty => Vec4::new(0.0, 0.0, 0.0, 1.0),
                    Tile::Food => Vec4::new(0.0, 1.0, 0.0, 1.0),
                    Tile::Prey(_) => Vec4::new(1.0, 1.0, 1.0, 1.0),
                    Tile::Predator(_) => Vec4::new(1.0, 0.0, 0.0, 1.0),
                });
            }
        }
        outputs.set_texture.push((t, 0));
        outputs.draw_texture.push((inputs.screen_rect, 0));

        // println!("nmo: {:?}", nmo);
        if inputs.frame % 60 == 0 {
            println!("grass: {} prey: {} pred: {}", self.num_food, self.num_prey, self.num_pred);
            println!("grass: {} prey: {} pred: {}", actual_num_food, actual_num_prey, actual_num_pred);
        }





    }
}