use std::pin::Pin;

use app::App;
use wgpu_state::WGPUState;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    window::*,
};

mod app;
mod wgpu_state;

#[derive(Debug)]
struct AppState {
    wgpu_state: WGPUState<'static>,
    app: Pin<Box<App>>,
}

#[derive(Debug, Default)]
struct AppWrapper {
    app_state: Option<AppState>,
}

impl ApplicationHandler for AppWrapper {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let app = Pin::new(Box::new(App::new(event_loop).unwrap()));
        let wgpu_state = WGPUState::new(unsafe { std::mem::transmute(app.window()) }).unwrap();

        let state = AppState { wgpu_state, app };

        self.app_state = Some(state);
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        self.app_state = None;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        if let Some(w) = self.app_state.as_mut() {
            if window_id == w.app.window_id() {
                w.app.window_event(event_loop, event, &mut w.wgpu_state);
            }
        }
    }
}

fn main() -> Result<(), anyhow::Error> {
    env_logger::init();
    let event_loop = EventLoop::new()?;
    let mut app = AppWrapper::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}
