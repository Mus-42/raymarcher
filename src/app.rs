use std::time::Instant;

use winit::{
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::ActiveEventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

use crate::wgpu_state::WGPUState;

#[derive(Debug)]
pub struct App {
    is_init: bool,
    window: Window,
    state: InputState,
    last_frame_start: Instant,
}

#[derive(Debug, Default)]
pub struct InputState {
    pub is_foward_pressed: bool,
    pub is_backward_pressed: bool,
    pub is_left_pressed: bool,
    pub is_right_pressed: bool,
    pub is_up_pressed: bool,
    pub is_down_pressed: bool,
    pub mouse_offset: [f32; 2],
    pub mouse_position: [f32; 2],
    pub is_mouse_pressed: bool,
}

impl App {
    pub fn new(event_loop: &ActiveEventLoop) -> Result<Self, anyhow::Error> {
        let attributes = WindowAttributes::default();
        let window = event_loop.create_window(attributes)?;
        let state = InputState::default();

        Ok(Self {
            is_init: false,
            window,
            state,
            last_frame_start: Instant::now(),
        })
    }

    pub fn window_id(&self) -> WindowId {
        self.window.id()
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        event: WindowEvent,
        wgpu_state: &mut WGPUState<'_>,
    ) {
        use WindowEvent as E;
        match event {
            E::KeyboardInput { event, .. } => {
                let is_pressed = event.state == ElementState::Pressed;
                use KeyCode as C;
                use PhysicalKey as K;
                match event.physical_key {
                    K::Code(C::KeyW) => self.state.is_foward_pressed = is_pressed,
                    K::Code(C::KeyA) => self.state.is_left_pressed = is_pressed,
                    K::Code(C::KeyS) => self.state.is_backward_pressed = is_pressed,
                    K::Code(C::KeyD) => self.state.is_right_pressed = is_pressed,
                    K::Code(C::Space) => self.state.is_up_pressed = is_pressed,
                    K::Code(C::ShiftLeft) => self.state.is_down_pressed = is_pressed,
                    _ => {}
                };
            }
            E::MouseInput { state, button, .. } => {
                if button != MouseButton::Left {
                    return;
                }
                let is_pressed = state == ElementState::Pressed;
                self.state.is_mouse_pressed = is_pressed;
            }
            E::CursorMoved { position, .. } => {
                let pos = [position.x as f32, position.y as f32];
                let offset = [
                    pos[0] - self.state.mouse_position[0],
                    pos[1] - self.state.mouse_position[1],
                ];
                self.state.mouse_position = pos;
                self.state.mouse_offset = offset;
            }

            E::CloseRequested => event_loop.exit(),
            E::RedrawRequested => {
                self.window.request_redraw();
                wgpu_state.update(&self.state);
                match wgpu_state.render() {
                    Err(wgpu::SurfaceError::Lost) => wgpu_state.resize(wgpu_state.size),
                    Err(e) => eprintln!("{e:?}"),
                    Ok(()) => {}
                }
                self.state.mouse_offset = [0.; 2];
                println!(
                    "fps: {}",
                    1. / self.last_frame_start.elapsed().as_secs_f64()
                );
                self.last_frame_start = Instant::now();
            }
            E::Resized(new_size) => {
                if self.is_init {
                    wgpu_state.resize(new_size);
                } else {
                    self.is_init = true;
                }
            }
            _ => {}
        }
    }
}
