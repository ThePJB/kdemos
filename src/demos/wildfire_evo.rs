use std::f64::INFINITY;

use crate::{scene::{Demo, FrameOutputs}, kinput::FrameInputState, renderers::mesh_renderer::MeshBuilder};
use crate::kmath::*;
use crate::texture_buffer::*;
use glutin::event::VirtualKeyCode;

// stats: better to do quartiles, also variance
// but it does seem to be niching
// could probably get more niching if there was variety in catching fire chance etc
// seasonal fluctuations
// could evolve the mutation rates too
// smaller scale faster simulation
// covariance analysis: i bet its fire res + distance vs growth + scatter rate. weeds vs noble trees

#[repr(usize)]
#[derive(PartialEq, Clone, Copy)]
pub enum TreeGene {
    FireStop = 0,
    // FireStart,
    ScatterChance,
    ScatterDist,
    GrowthChance,
    NumGenes,
}

type Genome = [f64; TreeGene::NumGenes as usize];

fn normalize_genome(g: &mut Genome) {
    let mut acc = 0.0;
    for x in g.iter() {
        acc += *x;
    }
    for x in g {
        *x /= acc;
    }
}

fn new_genome(mut seed: u32) -> Genome {
    let mut g = [0.0; TreeGene::NumGenes as usize];
    for g in g.iter_mut() {
        *g = krand(seed);
        seed = khash(seed);
    }
    normalize_genome(&mut g);
    g
}

fn mutate_genome(g: &mut Genome, mut seed: u32) {
    for g in g.iter_mut() {
        *g += 0.01 * krand(seed);
        seed = khash(seed);
    }
    normalize_genome(g);
}

// so probably its a unit vector. so cheapness is how strongly it gets expressed as a small amount
// this gets remapped into a phenotype for the simulation
// eg growth rate of 0..1 ... chance to grow per frame
// you could consider a linear remapping for each parameter: ax + b
// and the mutation amount. could have a separate vector for that even if you wanted... the free vector. fire start could be free
// vs the trading vector

// trees dying on their own?

// #[derive(Clone, Copy)]
pub struct Tree {
    g: Genome,
    fuel: f32,
    burning: bool,
}

pub struct WildfireEvo {
    height: Vec<f64>,
    trees: Vec<Option<Tree>>,
    niche_view: Option<TreeGene>,
    w: usize,
    h: usize,

    stale: bool,

    seed: u32,
    ffwd: bool,
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

impl Default for WildfireEvo {
    fn default() -> Self {
        let mut w = WildfireEvo { 
            height: vec![],
            trees: vec![],
            w: 400,
            h: 400,
            stale: false,
            seed: 0,
            niche_view: None,
            ffwd: false,
        };
        w.gen();
        w
    }
}

impl WildfireEvo {
    fn gen(&mut self) {
        self.seed = (self.seed + 123124817) * 92351709;
        self.height = vec![0.0; self.w  * self.h];
        self.trees = vec![];

        for i in 0..self.w*self.h {
            let g = new_genome(self.seed + i as u32 * 18341357);
            self.trees.push(Some(Tree {
                g,
                fuel: 1.0,
                burning: false,
            }));
        }
        
        for i in 0..self.w {
            for j in 0..self.h {
                self.height[j*self.w + i] = h2_fn(i as f64, j as f64, self.seed);
            }
        }
        self.stale = true;
    }

    fn grow(&mut self, seed: u32) {
        for i in 0..self.trees.len() {
            if let Some(mut tree) = self.trees[i].as_mut() {
                let p = tree.g[TreeGene::GrowthChance as usize] * 0.01;
                if chance(123127 + seed + i as u32 * 12247717, p) {
                    tree.fuel += 1.0;
                }
            }
        }
    }



    fn ignite(&mut self, seed: u32) {
        for i in 0..self.trees.len() {
            if let Some(mut tree) = self.trees[i].as_mut() {
                if chance(1237597 + i as u32 * 234234181 + seed, 0.0000001) {
                    tree.burning = true;
                }
            }
        }
    }

    fn burn(&mut self, seed: u32) {
        let mut order: Vec<usize> = (0..self.w*self.h).collect();
        for i in 0..order.len() {
            let swap_idx = khash(seed + i as u32 * 2398402317) % (order.len() as u32 - i as u32) + i as u32;
            order.swap(i, swap_idx as usize);
        }

        let p_hi = 0.4;
        let p_lo = 0.03;

        for idx in order {
            if let Some(mut tree) = self.trees[idx].as_mut() {
                if tree.burning {

                    // chance to stop
                    let p = tree.g[TreeGene::FireStop as usize] * 0.2;
                    if chance(12431247 + seed * 2130151957 + idx as u32 * 1231265, p.into()) {
                        tree.burning = false;
                        continue;
                    }

                    tree.fuel -= 1.0;

                    // spread up
                    if idx > self.w {
                        if let Some(mut neigh) = self.trees[idx - self.w].as_mut() {
                            let p = if self.height[idx - self.w] > self.height[idx] {
                                p_hi
                            } else {
                                p_lo
                            };
                            if chance(idx as u32 * 1237171797 + seed, p) {
                                neigh.burning = true;
                            }
                        }
                    }

                    // spread down
                    if idx < self.w*(self.h - 1) {
                        if let Some(mut neigh) = self.trees[idx + self.w].as_mut() {
                            let p = if self.height[idx + self.w] > self.height[idx] {
                                p_hi
                            } else {
                                p_lo
                            };
                            if chance(idx as u32 * 1231241877 + seed, p) {
                                neigh.burning = true;
                            }
                        }
                    }

                    // spread left
                    if idx % self.w != 0 {
                        if let Some(mut neigh) = self.trees[idx - 1].as_mut() {
                            let p = if self.height[idx - 1] > self.height[idx] {
                                p_hi
                            } else {
                                p_lo
                            };
                            if chance(idx as u32 * 534343257 + seed, p)  {
                                neigh.burning = true;
                            }
                        }
                    }

                    // spread right
                    if (idx + 1) % self.w  != 0 {
                        if let Some(mut neigh) = self.trees[idx + 1].as_mut() {
                            let p = if self.height[idx + 1] > self.height[idx] {
                                p_hi
                            } else {
                                p_lo
                            };
                            if chance(idx as u32 * 131254717 + seed, p) {
                                neigh.burning = true;
                            }
                        }
                    }
                }

                // burn out
                if self.trees[idx].is_some() && self.trees[idx].as_ref().unwrap().burning && self.trees[idx].as_ref().unwrap().fuel <= 0.0 {
                    self.trees[idx] = None;
                }
            }
        }
    }

    fn scatter(&mut self, seed: u32) {
        let mut order: Vec<usize> = (0..self.w*self.h).collect();
        for i in 0..order.len() {
            let swap_idx = khash(seed + i as u32 * 1418957177) % (order.len() as u32 - i as u32) + i as u32;
            order.swap(i, swap_idx as usize);
        }

        for idx in order {
            if let Some(tree) = self.trees[idx].as_ref() {
                let x = idx % self.w;
                let y = idx / self.w;

                let sp = tree.g[TreeGene::ScatterChance as usize];
                let sd = tree.g[TreeGene::ScatterDist as usize];

                let s = 341547 + seed * 1234157 + idx as u32 * 1549157;
                let p = sp * 0.01 * tree.fuel as f64;
                if chance(s, p) {
                    let sx = kuniform(s * 15117677, -sd, sd);
                    let sy = kuniform(s * 41519717, -sd, sd);

                    let sx = sx * 20.0 + x as f64;
                    let sy = sy * 20.0 + y as f64;

                    if sx > 0.0 && sx < self.w as f64 && sy > 0.0 && sy < self.h as f64 {
                        let sx = sx as usize;
                        let sy = sy as usize;
                        if self.trees[sy * self.w + sx].is_none() {
                            let mut g = tree.g;
                            mutate_genome(&mut g, s * 19411517);
                            self.trees[sy * self.w + sx] = Some(Tree {
                                g,
                                burning: false,
                                fuel: 1.0,
                            });
                        }
                    }
                }
            }
        }
    }

    fn stats(&self) {
        // print out min max mean median mode etc for the fuel values and genome values
        let mut max_fuel = 0.0;
        let mut alive = 0;
        let mut dead = 0;

        let mut min = [INFINITY; TreeGene::NumGenes as usize];
        let mut max = [0.0f64; TreeGene::NumGenes as usize];
        let mut sum = [0.0f64; TreeGene::NumGenes as usize];

        for t in self.trees.iter() {
            if let Some(t) = t {
                alive += 1;
                if t.fuel > max_fuel {
                    max_fuel = t.fuel;
                }
                for (i, g) in t.g.iter().enumerate() {
                    min[i] = min[i].min(*g);
                    max[i] = max[i].max(*g);
                    sum[i] = sum[i] + *g;
                }
            } else {
                dead += 1;
            }
        }

        for s in sum.iter_mut() {
            *s /= alive as f64;
        }

        println!("max fuel: {:<3} alive {:<6} dead {:<6} stopburn [{:.2} {:.2} {:.2}] pscatter [{:.2} {:.2} {:.2}] sdist [{:.2} {:.2} {:.2}] pgrow [{:.2} {:.2} {:.2}]", max_fuel, alive, dead, 
        min[TreeGene::FireStop as usize],
        sum[TreeGene::FireStop as usize],
        max[TreeGene::FireStop as usize],
        
        min[TreeGene::ScatterChance as usize],
        sum[TreeGene::ScatterChance as usize],
        max[TreeGene::ScatterChance as usize],
        
        min[TreeGene::ScatterDist as usize],
        sum[TreeGene::ScatterDist as usize],
        max[TreeGene::ScatterDist as usize],
        
        min[TreeGene::GrowthChance as usize],
        sum[TreeGene::GrowthChance as usize],
        max[TreeGene::GrowthChance as usize],
        
        );
    }

    fn step(&mut self, seed: u32) {
        self.grow(seed);
        self.ignite(seed);
        self.burn(seed);
        self.scatter(seed);
    }
}

impl Demo for WildfireEvo {
    fn frame(&mut self, inputs: &FrameInputState, outputs: &mut FrameOutputs) {
        if inputs.key_rising(VirtualKeyCode::R) {
            self.gen();
        }

        if inputs.key_rising(VirtualKeyCode::M) {
            for i in 0..1000 {
                self.step(inputs.seed + i * 151716717);
            }
        }

        if inputs.key_rising(VirtualKeyCode::E) {
            self.ffwd ^= true;
        }
        if inputs.key_rising(VirtualKeyCode::Key1) {
            if self.niche_view.is_some() && self.niche_view.unwrap() == TreeGene::FireStop {
                self.niche_view = None;
            } else {
                self.niche_view = Some(TreeGene::FireStop);
            }
        }
        if inputs.key_rising(VirtualKeyCode::Key2) {
            if self.niche_view.is_some() && self.niche_view.unwrap() == TreeGene::ScatterChance {
                self.niche_view = None;
            } else {
                self.niche_view = Some(TreeGene::ScatterChance);
            }
        }
        if inputs.key_rising(VirtualKeyCode::Key3) {
            if self.niche_view.is_some() && self.niche_view.unwrap() == TreeGene::ScatterDist {
                self.niche_view = None;
            } else {
                self.niche_view = Some(TreeGene::ScatterDist);
            }
        }
        if inputs.key_rising(VirtualKeyCode::Key4) {
            if self.niche_view.is_some() && self.niche_view.unwrap() == TreeGene::GrowthChance {
                self.niche_view = None;
            } else {
                self.niche_view = Some(TreeGene::GrowthChance);
            }
        }


        if self.ffwd {
            for i in 0..100 {
                self.step(inputs.seed + i * 151716717);
            }
        } else {
            self.step(inputs.seed);
        }
        self.stats();

        let c_nf = Vec4::new(0.0, 1.0, 0.0, 1.0);

        let c_ne = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let c_fe = Vec4::new(1.0, 0.0, 0.0, 1.0);
        let c_ff = Vec4::new(1.0, 1.0, 1.0, 1.0);

        let mut tb = TextureBuffer::new(self.w, self.h);
        for i in 0..self.w {
            for j in 0..self.h {
                let idx = j*self.w + i;

                if self.niche_view == None {
                    let c = if let Some(t) = self.trees[idx].as_ref() {
                        if t.burning {
                            c_fe.lerp(c_ff, (t.fuel / 50.0).min(1.0) as f64)
                        } else {
                            c_ne.lerp(c_nf, (t.fuel / 50.0).min(1.0) as f64)
                        }
                    } else {
                        c_ne
                    };
    
                    tb.set(i as i32, j as i32, c);
                } else {
                    let colours = [
                        Vec4::new(1.0, 0.0, 0.0, 1.0),
                        Vec4::new(0.0, 1.0, 1.0, 1.0),
                        Vec4::new(0.0, 0.0, 1.0, 1.0),
                        Vec4::new(1.0, 1.0, 0.0, 1.0),
                    ];

                    let c = colours[self.niche_view.unwrap() as usize];
                    let t = self.trees[idx].as_ref().map(|t| t.g[self.niche_view.unwrap() as usize]).unwrap_or(0.0);
                    tb.set(i as i32, j as i32, Vec4::new(0.0, 0.0, 0.0, 1.0).lerp(c, t));
                }
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