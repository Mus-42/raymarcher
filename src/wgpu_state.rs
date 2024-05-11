use anyhow::anyhow;
use pollster::FutureExt;
use wgpu::{
    util::{self, DeviceExt},
    Backends, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayoutDescriptor,
    BindGroupLayoutEntry, BindingResource, BindingType, BlendComponent, BlendState, Buffer,
    BufferBindingType, BufferUsages, Color, ColorTargetState, ColorWrites,
    CommandEncoderDescriptor, Device, DeviceDescriptor, Extent3d, Face, FragmentState, FrontFace,
    ImageDataLayout, IndexFormat, Instance, InstanceDescriptor, LoadOp, MultisampleState,
    Operations, PipelineCompilationOptions, PipelineLayoutDescriptor, PolygonMode, PowerPreference,
    PrimitiveState, PrimitiveTopology, Queue, RenderPassColorAttachment, RenderPassDescriptor,
    RenderPipeline, RenderPipelineDescriptor, RequestAdapterOptions, SamplerBindingType,
    ShaderModuleDescriptor, ShaderStages, StoreOp, Surface, SurfaceConfiguration, SurfaceError,
    Texture, TextureDescriptor, TextureDimension, TextureFormat, TextureUsages,
    TextureViewDescriptor, VertexState,
};
use winit::{dpi::PhysicalSize, window::Window};

use crate::app::InputState;

// https://www.shadertoy.com/view/lslXDr
// https://sotrh.github.io/learn-wgpu/beginner/tutorial6-uniforms/#uniform-buffers-and-bind-groups

#[derive(Debug, Clone, Copy, Default)]
struct Camera {
    yaw: f32,
    pitch: f32,
    vel: [f32; 3],
    pos: [f32; 3],
}

impl Camera {
    fn get_transform(&self) -> CameraTransform {
        let (ys, yc) = self.yaw.sin_cos();
        let (ps, pc) = self.pitch.sin_cos();
        let [x, y, z] = self.pos;
        #[rustfmt::skip]
        let transform = [
            [pc * yc, -ps, pc * ys, 0.],
            [yc * ps, pc, ys * ps, 0.],
            [-ys, 0., yc, 0.],
            [x, y, z, 1.],
        ];
        CameraTransform(transform)
    }

    fn update(&mut self, window_state: &InputState) {
        // TODO dt?
        let dt = 1. / 60.;
        self.vel[0] *= 0.97;
        self.vel[1] *= 0.97;
        self.vel[2] *= 0.97;
        if window_state.is_mouse_pressed {
            self.yaw += window_state.mouse_offset[0] / 200.;
            self.pitch += window_state.mouse_offset[1] / 200.;
            self.pitch = self.pitch.clamp(-1.5, 1.5);
        }
        let a_forward = window_state.is_foward_pressed as i8 as f32
            - window_state.is_backward_pressed as i8 as f32;
        let a_upward =
            window_state.is_up_pressed as i8 as f32 - window_state.is_down_pressed as i8 as f32;
        let a_right =
            window_state.is_right_pressed as i8 as f32 - window_state.is_left_pressed as i8 as f32;

        const A: f32 = 10.;

        let (s, c) = self.yaw.sin_cos();
        self.vel[0] += (c * a_forward - s * a_right) * dt * A;
        self.vel[1] += a_upward * dt * A;
        self.vel[2] += (s * a_forward + c * a_right) * dt * A;

        self.pos[0] += self.vel[0] * dt;
        self.pos[1] += self.vel[1] * dt;
        self.pos[2] += self.vel[2] * dt;
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct CameraTransform([[f32; 4]; 4]);

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ScreenSize([f32; 2]);

const TEXMAPS_DATA: &[u8] = include_bytes!("../assets/texmaps.qoi");

#[derive(Debug)]
pub struct WGPUState<'a> {
    surface: Surface<'a>,
    device: Device,
    queue: Queue,
    config: SurfaceConfiguration,
    pub size: PhysicalSize<u32>,
    render_pipeline: RenderPipeline,
    index_buffer: Buffer,
    camera: Camera,
    camera_uniform_buf: Buffer,
    screen_size_uniform_buf: Buffer,
    bindgroup: BindGroup,
    texmaps: Texture,
}

impl<'a> WGPUState<'a> {
    pub fn new(window: &'a Window) -> Result<Self, anyhow::Error> {
        let size = window.inner_size();

        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window)?;

        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .block_on()
            .ok_or_else(|| anyhow!("adapter is None"))?;

        let (device, queue) = adapter
            .request_device(&DeviceDescriptor::default(), None)
            .block_on()?;

        let surface_caps = surface.get_capabilities(&adapter);

        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shader.wgsl").into()),
        });

        let index_buffer = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: &[0, 0, 1, 0, 3, 0, 0, 0, 3, 0, 2, 0],
            usage: BufferUsages::INDEX,
        });

        let (texmaps_desc, texmaps_data) = rapid_qoi::Qoi::decode_alloc(&TEXMAPS_DATA).unwrap();

        use rapid_qoi::Colors as C;

        let (window_size, texmap_format) = match texmaps_desc.colors {
            C::Rgb => (3, TextureFormat::Rgba8Unorm),
            C::Rgba => (4, TextureFormat::Rgba8Unorm),
            C::Srgb => (3, TextureFormat::Rgba8UnormSrgb),
            C::SrgbLinA => (4, TextureFormat::Rgba8UnormSrgb),
        };

        let texmaps_size = Extent3d {
            width: texmaps_desc.width,
            height: texmaps_desc.height,
            depth_or_array_layers: 1,
        };

        let texmaps = device.create_texture(&TextureDescriptor {
            label: Some("texmaps"),
            size: texmaps_size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: texmap_format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texmaps_data_rgba = texmaps_data
            .chunks(window_size)
            .flat_map(|pixel| {
                let mut bytes = [255; 4];
                bytes[..pixel.len()].clone_from_slice(pixel);
                (bytes[1], bytes[2]) = (bytes[2], bytes[1]);
                bytes
            })
            .collect::<Vec<u8>>();

        assert_eq!(
            texmaps_data_rgba.len(),
            texmaps_desc.height as usize * texmaps_desc.width as usize * 4
        );

        queue.write_texture(
            texmaps.as_image_copy(),
            &texmaps_data_rgba,
            ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(4 * texmaps_desc.width),
                rows_per_image: Some(texmaps_desc.height),
            },
            texmaps_size,
        );

        let texmaps_texture_view = texmaps.create_view(&wgpu::TextureViewDescriptor::default());
        let texmaps_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let camera = Camera::default();

        let camera_uniform_data: [u8; std::mem::size_of::<CameraTransform>()] =
            unsafe { std::mem::transmute(camera.get_transform()) };

        let screen_size_uniform_data = [0u8; std::mem::size_of::<[f32; 2]>()];

        let camera_uniform_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("camera uniform buf"),
            contents: &camera_uniform_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let screen_size_uniform_buf = device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some("screen size uniform buf"),
            contents: &screen_size_uniform_data,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let bindgroup_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            entries: &[
                BindGroupLayoutEntry {
                    binding: 0,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 1,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Buffer {
                        ty: BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 2,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                BindGroupLayoutEntry {
                    binding: 3,
                    visibility: ShaderStages::FRAGMENT,
                    ty: BindingType::Sampler(SamplerBindingType::Filtering),
                    count: None,
                },
            ],
            label: Some("camera bind group layout"),
        });

        let bindgroup = device.create_bind_group(&BindGroupDescriptor {
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: camera_uniform_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: screen_size_uniform_buf.as_entire_binding(),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(&texmaps_texture_view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&texmaps_sampler),
                },
            ],
            layout: &bindgroup_layout,
            label: Some("camera  bind group layout"),
        });

        let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&bindgroup_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[],
                compilation_options: PipelineCompilationOptions::default(),
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(ColorTargetState {
                    format: config.format,
                    blend: Some(BlendState {
                        color: BlendComponent::REPLACE,
                        alpha: BlendComponent::REPLACE,
                    }),
                    write_mask: ColorWrites::ALL,
                })],
                compilation_options: PipelineCompilationOptions::default(),
            }),
            primitive: PrimitiveState {
                topology: PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                polygon_mode: PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Ok(Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            index_buffer,
            camera,
            camera_uniform_buf,
            screen_size_uniform_buf,
            bindgroup,
            texmaps,
        })
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = self.size.width;
            self.config.height = self.size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    pub fn update(&mut self, window_state: &InputState) {
        self.camera.update(window_state);

        let camera_uniform_data: [u8; std::mem::size_of::<CameraTransform>()] =
            unsafe { std::mem::transmute(self.camera.get_transform()) };

        self.queue
            .write_buffer(&self.camera_uniform_buf, 0, &camera_uniform_data);

        let screen_size_uniform_data: [u8; std::mem::size_of::<[f32; 2]>()] =
            unsafe { std::mem::transmute([self.size.width as f32, self.size.height as f32]) };

        self.queue
            .write_buffer(&self.screen_size_uniform_buf, 0, &screen_size_uniform_data);
    }

    pub fn render(&mut self) -> Result<(), SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.bindgroup, &[]);
            render_pass.set_index_buffer(self.index_buffer.slice(..), IndexFormat::Uint16);
            render_pass.draw_indexed(0..6, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
