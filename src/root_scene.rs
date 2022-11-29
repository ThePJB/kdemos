use glutin::event::VirtualKeyCode;
use crate::scene::*;
use crate::kmath::*;

use crate::demos::bsj::*;
use crate::demos::mdj::*;
use crate::demos::burning_ship::*;
use crate::demos::percoviz::*;
use crate::demos::mandelbrot::*;
use crate::demos::julia::*;
use crate::demos::rgbwalk::*;
use crate::demos::rgbutm::*;
use crate::demos::noise_test::*;
use crate::demos::voronoinoi::*;
use crate::demos::new_fractal::*;
use crate::demos::wildfire::*;
use crate::demos::wildfire_evo::*;
use crate::demos::terrain::*;
use crate::demos::hydrasim::*;

pub struct RootScene {
    curr_scene: Option<Box<dyn Demo>>,
    demo_table: Vec<(&'static str, fn() -> Box<dyn Demo>)>,
    idx: usize,
    show_menu: bool,
}

impl RootScene {
    pub fn new() -> RootScene {
        let mut demo_table: Vec<(&str, fn() -> Box<dyn Demo>)> = Vec::new();
        
        // Fractals
        demo_table.push(("Mandelbrot", init_demo::<Mandelbrot>));
        demo_table.push(("Julia", init_demo::<Julia>));
        demo_table.push(("Mandel-Julia", init_demo::<MDJ>));
        demo_table.push(("Burning Ship", init_demo::<BurningShip>));
        demo_table.push(("Burn-Julia", init_demo::<BSJ>));
        demo_table.push(("New Fractal", init_demo::<NewFractal>));
        
        // Random walk
        demo_table.push(("RGBWalk", init_demo::<RGBWalk>));
        demo_table.push(("RGBUTM", init_demo::<RGBUTM>));

        // Noise
        demo_table.push(("Recnoise", init_demo::<NoiseTest>));

        // Others
        demo_table.push(("Percolation", init_demo::<Percoviz>));
        demo_table.push(("Bowyer Watson", init_demo::<Voronoinoi>));
        demo_table.push(("Wildfire", init_demo::<Wildfire>));
        demo_table.push(("Wildfire Evo", init_demo::<WildfireEvo>));
        demo_table.push(("Terrain", init_demo::<Terrain>));
        demo_table.push(("Hydrasim", init_demo::<Hydrasim>));

        RootScene {
            curr_scene: None,
            demo_table,
            idx: 0,
            show_menu: false,
        }
    }
}

impl Demo for RootScene {
    fn frame(&mut self, inputs: &crate::kinput::FrameInputState, outputs: &mut FrameOutputs) {
        if let Some(curr) = self.curr_scene.as_mut() {
            curr.frame(inputs, outputs);
        } else {
            self.show_menu = true;
        }

        if inputs.key_rising(VirtualKeyCode::Space) {
            self.show_menu = !self.show_menu;
        }

        if inputs.key_rising(VirtualKeyCode::Return) {
            self.curr_scene = Some(self.demo_table[self.idx].1());
            self.show_menu = false;
        }

        if inputs.key_press_or_repeat(VirtualKeyCode::K) {
            if self.idx > 0 {
                self.idx = self.idx - 1;
            }
        }
        if inputs.key_press_or_repeat(VirtualKeyCode::J) {
            if self.idx < self.demo_table.len() - 1 {
                self.idx = self.idx + 1;
            }
        }
        if self.show_menu {
            let ca = 12.0 / 14.0;
            let ch = 0.02;
            let cw = ch * ca;
            let w = cw * 30.0;

            let c = Vec4::grey(0.3);
            let wcom = 1.0 - w;
            outputs.canvas.put_rect(inputs.screen_rect.child(wcom/2.0, 0.0, w, 1.0), 4.0, c);

            for i in 0..self.demo_table.len() {
                let x = inputs.screen_rect.w/2.0;
                let y_initial = 0.005;
                let colour = if i == self.idx {
                    Vec4::new(1.0, 1.0, 0.0, 1.0)
                } else {
                    Vec4::new(1.0, 1.0, 1.0, 1.0)
                };
                outputs.glyphs.push_center_str(self.demo_table[i].0, x, y_initial + i as f64 * ch, cw, ch * 0.8, 5.5, colour);
            }
        }
    }
}