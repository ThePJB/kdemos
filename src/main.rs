#![feature(drain_filter)]

mod application;
mod video;
mod kmath;
mod priority_queue;
mod kimg;
mod kinput;
mod texture_buffer;

mod renderers;
mod scene;
mod widgets;

mod demos;
mod root_scene;

use crate::application::*;

fn main() {
    let event_loop = glutin::event_loop::EventLoop::new();
    let mut application = Application::new(&event_loop);
    
    event_loop.run(move |event, _, _| {
        application.handle_event(event);
    });
}