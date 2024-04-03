use std::{
    borrow::Cow,
    collections::HashSet,
    env,
    ffi::{c_char, c_void, CStr, CString},
    fs,
    io::{self, Cursor},
    mem, slice,
    time::Instant,
};

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    platform::run_return::EventLoopExtRunReturn,
    window::{Window, WindowBuilder},
};

use ash::{
    extensions::{
        ext::DebugUtils,
        khr::{Surface, Swapchain},
    },
    util::Align,
    vk::{self, DebugUtilsMessengerEXT, SurfaceFormatKHR},
    Device, Entry, Instance,
};

use ash_window::enumerate_required_extensions;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

use cgmath::{self, InnerSpace};
use rand::{self, Rng};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const PARTICLE_COUNT: u32 = 8192;

const MAX_FRAMES_IN_FLIGHT: i8 = 2;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

const DEVICE_EXTENSIONS: [*const i8; 1] = [Swapchain::name().as_ptr()];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

// Simple offset_of macro akin to C++ offsetof
#[macro_export]
macro_rules! offset_of {
    ($base:path, $field:ident) => {{
        #[allow(unused_unsafe)]
        unsafe {
            let b: $base = mem::zeroed();
            std::ptr::addr_of!(b.$field) as isize - std::ptr::addr_of!(b) as isize
        }
    }};
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data: *mut std::os::raw::c_void,
) -> vk::Bool32 {
    let callback_data = *p_callback_data;
    let message_id_number = callback_data.message_id_number;

    let message_id_name = if callback_data.p_message_id_name.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
    };

    let message = if callback_data.p_message.is_null() {
        Cow::from("")
    } else {
        CStr::from_ptr(callback_data.p_message).to_string_lossy()
    };

    println!(
        "{message_severity:?}:\n{message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

    vk::FALSE
}

struct QueueFamilyIndices {
    graphics_and_compute_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> Self {
        Self {
            graphics_and_compute_family: None,
            present_family: None,
        }
    }

    fn is_complete(&mut self) -> bool {
        self.graphics_and_compute_family.is_some() && self.present_family.is_some()
    }
}

struct SwapChainSupportDetails {
    capabilities: Option<vk::SurfaceCapabilitiesKHR>,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapChainSupportDetails {
    pub fn new() -> Self {
        Self {
            capabilities: None,
            formats: [].to_vec(),
            present_modes: [].to_vec(),
        }
    }
}

#[derive(Clone, Debug, Copy)]
struct UniformBufferObject {
    delta_time: f32,
}

impl UniformBufferObject {
    pub fn new() -> Self {
        Self { delta_time: 1.0 }
    }
}

#[derive(Clone, Debug, Copy)]
struct Particle {
    pos: [f32; 2],
    velocity: [f32; 2],
    color: [f32; 4],
}

impl Particle {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
        ]
    }
}

#[allow(unused)]
struct ComputeShaderApplication {
    window: Option<Window>,
    vk_entry: Option<Entry>,
    instance: Option<Instance>,
    debug_utils: Option<DebugUtils>,
    debug_messenger: Option<DebugUtilsMessengerEXT>,
    surface: Option<vk::SurfaceKHR>,
    surface_loader: Option<Surface>,
    physical_device: Option<vk::PhysicalDevice>,
    device: Option<Device>,
    graphics_queue: Option<vk::Queue>,
    compute_queue: Option<vk::Queue>,
    present_queue: Option<vk::Queue>,

    swapchain_loader: Option<Swapchain>,
    swap_chain: Option<vk::SwapchainKHR>,
    swap_chain_images: Vec<vk::Image>,
    swap_chain_image_format: Option<vk::Format>,
    swap_chain_extent: Option<vk::Extent2D>,
    swap_chain_image_views: Vec<vk::ImageView>,
    swap_chain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: Option<vk::RenderPass>,
    pipeline_layout: Option<vk::PipelineLayout>,
    graphics_pipeline: Option<vk::Pipeline>,

    compute_descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    compute_pipeline_layout: Option<vk::PipelineLayout>,
    compute_pipeline: Option<vk::Pipeline>,

    command_pool: Option<vk::CommandPool>,

    shader_storage_buffers: Vec<vk::Buffer>,
    shader_storage_buffers_memory: Vec<vk::DeviceMemory>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    descriptor_pool: Option<vk::DescriptorPool>,
    compute_descriptor_sets: Vec<vk::DescriptorSet>,

    command_buffers: Vec<vk::CommandBuffer>,
    compute_command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    compute_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    compute_in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    last_frame_time: f32,

    framebuffer_resized: bool,

    last_time: f64,

    start_time: Instant,
}

impl ComputeShaderApplication {
    pub fn new() -> Self {
        Self {
            window: None,
            vk_entry: None,
            instance: None,
            debug_utils: None,
            debug_messenger: None,
            surface: None,
            surface_loader: None,
            physical_device: None,
            device: None,
            graphics_queue: None,
            compute_queue: None,
            present_queue: None,
            swap_chain: None,
            swapchain_loader: None,
            swap_chain_images: [].to_vec(),
            swap_chain_image_format: None,
            swap_chain_extent: None,
            swap_chain_image_views: [].to_vec(),
            swap_chain_framebuffers: [].to_vec(),
            render_pass: None,
            pipeline_layout: None,
            graphics_pipeline: None,
            compute_descriptor_set_layout: None,
            compute_pipeline_layout: None,
            compute_pipeline: None,
            command_pool: None,
            shader_storage_buffers: [].to_vec(),
            shader_storage_buffers_memory: [].to_vec(),
            uniform_buffers: [].to_vec(),
            uniform_buffers_memory: [].to_vec(),
            uniform_buffers_mapped: [].to_vec(),
            descriptor_pool: None,
            compute_descriptor_sets: [].to_vec(),
            command_buffers: [].to_vec(),
            compute_command_buffers: [].to_vec(),
            image_available_semaphores: [].to_vec(),
            render_finished_semaphores: [].to_vec(),
            compute_finished_semaphores: [].to_vec(),
            in_flight_fences: [].to_vec(),
            compute_in_flight_fences: [].to_vec(),
            current_frame: 0,
            last_frame_time: 0.0,
            framebuffer_resized: false,
            last_time: 0.0,
            start_time: Instant::now(),
        }
    }

    pub fn run(&mut self) {
        let event_loop = EventLoop::new();
        self.init_window(&event_loop);
        self.init_vulkan();
        self.main_loop(event_loop);
    }

    fn init_vulkan(&mut self) {
        self.create_instance();
        self.setup_debug_messenger();
        self.create_surface();
        self.pick_physical_device();
        self.create_logical_device();
        self.create_swap_chain();
        self.create_image_views();
        self.create_render_pass();
        self.create_compute_descriptor_set_layout();
        self.create_graphics_pipeline();
        self.create_compute_pipeline();
        self.create_framebuffers();
        self.create_command_pool();
        self.create_shader_storage_buffers();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_compute_descriptor_sets();
        self.create_command_buffers();
        self.create_compute_command_buffers();
        self.create_sync_objects();
    }

    fn init_window(&mut self, event_loop: &EventLoop<()>) {
        let window = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            // .with_resizable(false)
            .build(event_loop)
            .unwrap();
        self.last_frame_time = self.start_time.elapsed().as_secs_f32();
        self.window = Some(window);
    }

    fn main_loop(&mut self, mut event_loop: EventLoop<()>) {
        event_loop.run_return(|event, _, control_flow| {
            control_flow.set_poll();

            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    ..
                } => {
                    println!("Window closed!");
                    control_flow.set_exit();
                }
                Event::WindowEvent {
                    event: WindowEvent::Resized(_new_size),
                    ..
                } => {
                    self.framebuffer_resized = true;
                }
                Event::MainEventsCleared => {
                    self.draw_frame();

                    self.last_frame_time = self.start_time.elapsed().as_secs_f32();
                }
                _ => (),
            }
        });

        // unsafe { self.device.as_ref().unwrap().device_wait_idle().unwrap() };
    }

    fn cleanup_swap_chain(&mut self) {
        unsafe {
            self.swap_chain_framebuffers.iter().for_each(|buffer| {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_framebuffer(*buffer, None);
            });

            self.swap_chain_image_views.iter().for_each(|view| {
                self.device
                    .as_ref()
                    .unwrap()
                    .destroy_image_view(*view, None);
            });

            self.swapchain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(self.swap_chain.unwrap(), None);
        }
    }

    fn recreate_swap_chain(&mut self) {
        let size = self.window.as_ref().unwrap().inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        unsafe { self.device.as_ref().unwrap().device_wait_idle().unwrap() };

        self.cleanup_swap_chain();

        self.create_swap_chain();
        self.create_image_views();
        self.create_framebuffers();
        self.window.as_ref().unwrap().request_redraw();
    }

    fn create_instance(&mut self) {
        self.instance = unsafe {
            let entry = Entry::linked();

            if ENABLE_VALIDATION_LAYERS && !self.check_validation_layer_support(&entry) {
                panic!("validation layers requested, but not available!");
            }

            let app_name = CString::new("Hello Triangle").unwrap();
            let engin_name = CString::new("No Engine").unwrap();

            let app_info = vk::ApplicationInfo::builder()
                .application_name(CStr::from_bytes_with_nul_unchecked(
                    app_name.to_bytes_with_nul(),
                ))
                .application_version(vk::make_api_version(0, 1, 0, 0))
                .engine_name(CStr::from_bytes_with_nul_unchecked(
                    engin_name.as_bytes_with_nul(),
                ))
                .engine_version(vk::make_api_version(0, 1, 0, 0))
                .api_version(vk::API_VERSION_1_0);

            let mut extension_names =
                enumerate_required_extensions(self.window.as_ref().unwrap().raw_display_handle())
                    .unwrap()
                    .to_vec();
            let mut create_info = vk::InstanceCreateInfo::builder().application_info(&app_info);

            let layer_names: Vec<*const c_char> = VALIDATION_LAYERS
                .iter()
                .map(|name| name.as_ptr() as *const c_char)
                .collect();
            if ENABLE_VALIDATION_LAYERS {
                extension_names.push(DebugUtils::name().as_ptr());
                create_info = create_info
                    .enabled_layer_names(&layer_names)
                    .enabled_extension_names(&extension_names);
            }

            let instance_ = entry
                .create_instance(&create_info, None)
                .expect("failed to create instance!");
            self.vk_entry = Some(entry);
            Some(instance_)
        }
    }

    fn populate_debug_messenger_create_info(
        &mut self,
    ) -> vk::DebugUtilsMessengerCreateInfoEXTBuilder {
        let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));
        debug_info
    }

    fn setup_debug_messenger(&mut self) {
        let debug_utils_loader = DebugUtils::new(
            self.vk_entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        );
        let debug_info = self.populate_debug_messenger_create_info();
        let debug_call_back = unsafe {
            debug_utils_loader
                .create_debug_utils_messenger(&debug_info, None)
                .unwrap()
        };
        self.debug_utils = Some(debug_utils_loader);
        self.debug_messenger = Some(debug_call_back);
    }

    fn create_surface(&mut self) {
        let surface = unsafe {
            ash_window::create_surface(
                self.vk_entry.as_ref().unwrap(),
                self.instance.as_ref().unwrap(),
                self.window.as_ref().unwrap().raw_display_handle(),
                self.window.as_ref().unwrap().raw_window_handle(),
                None,
            )
            .expect("failed to create window surface!")
        };
        let surface_loader = Surface::new(
            self.vk_entry.as_ref().unwrap(),
            self.instance.as_ref().unwrap(),
        );

        self.surface = Some(surface);
        self.surface_loader = Some(surface_loader);
    }

    fn pick_physical_device(&mut self) {
        let devices = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_physical_devices()
                .expect("failed to find GPUs with Vulkan support!")
        };

        for device in devices {
            if self.is_device_suitable(device) {
                self.physical_device = Some(device);
                break;
            }
        }

        if !self.physical_device.is_some() {
            panic!("failed to find a suitable GPU!");
        }
    }

    fn create_logical_device(&mut self) {
        let indices = self.find_queue_families(self.physical_device.unwrap());

        let queue_priority = [1.0];

        let unique_queue_families = HashSet::from([
            indices.graphics_and_compute_family.unwrap(),
            indices.present_family.unwrap(),
        ]);
        let queue_create_infos = unique_queue_families
            .iter()
            .map(|&queue_family| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family)
                    .queue_priorities(&queue_priority)
                    .build()
            })
            .collect::<Vec<_>>();

        let device_features = vk::PhysicalDeviceFeatures::default();
        let create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&device_features)
            .enabled_extension_names(&DEVICE_EXTENSIONS);

        // let layer_names: Vec<*const c_char> = VALIDATION_LAYERS
        //     .iter()
        //     .map(|name| name.as_ptr() as *const c_char)
        //     .collect();
        // if ENABLE_VALIDATION_LAYERS {
        //     create_info = create_info.enabled_layer_names(&layer_names);
        // }

        unsafe {
            let device = self
                .instance
                .as_ref()
                .unwrap()
                .create_device(self.physical_device.unwrap(), &create_info, None)
                .expect("failed to create logical device!");
            let graphics_queue =
                device.get_device_queue(indices.graphics_and_compute_family.unwrap(), 0);
            let compute_queue =
                device.get_device_queue(indices.graphics_and_compute_family.unwrap(), 0);
            let present_queue = device.get_device_queue(indices.present_family.unwrap(), 0);
            self.device = Some(device);
            self.graphics_queue = Some(graphics_queue);
            self.compute_queue = Some(compute_queue);
            self.present_queue = Some(present_queue);
        }
    }

    fn create_swap_chain(&mut self) {
        let swap_chain_support = self.query_swap_chain_support(self.physical_device.unwrap());
        let surface_format = self.choose_swap_surface_format(&swap_chain_support.formats);
        let present_mode = self.choose_swap_present_mode(&swap_chain_support.present_modes);
        let extent = self.choose_swap_extent(swap_chain_support.capabilities.as_ref().unwrap());

        let mut image_count = swap_chain_support
            .capabilities
            .as_ref()
            .unwrap()
            .min_image_count
            + 1;
        if swap_chain_support
            .capabilities
            .as_ref()
            .unwrap()
            .max_image_count
            > 0
            && image_count
                > swap_chain_support
                    .capabilities
                    .as_ref()
                    .unwrap()
                    .max_image_count
        {
            image_count = swap_chain_support
                .capabilities
                .as_ref()
                .unwrap()
                .max_image_count;
        }

        let mut create_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(self.surface.unwrap())
            .min_image_count(image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .pre_transform(
                swap_chain_support
                    .capabilities
                    .as_ref()
                    .unwrap()
                    .current_transform,
            )
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .clipped(true);

        let indices = self.find_queue_families(self.physical_device.unwrap());
        let queue_family_indices = [
            indices.graphics_and_compute_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        if indices.graphics_and_compute_family != indices.present_family {
            create_info = create_info
                .image_sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&queue_family_indices);
        } else {
            create_info = create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
        }

        let swapchain_loader = Swapchain::new(
            self.instance.as_ref().unwrap(),
            self.device.as_ref().unwrap(),
        );
        unsafe {
            let swapchain = swapchain_loader
                .create_swapchain(&create_info, None)
                .expect("failed to create swap chain!");

            self.swap_chain_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();

            self.swap_chain = Some(swapchain);
        }

        self.swapchain_loader = Some(swapchain_loader);
        self.swap_chain_image_format = Some(surface_format.format);
        self.swap_chain_extent = Some(extent);
    }

    fn create_image_views(&mut self) {
        self.swap_chain_image_views = self
            .swap_chain_images
            .iter()
            .map(|swap_chain_image| {
                let create_info = vk::ImageViewCreateInfo::builder()
                    .image(*swap_chain_image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(self.swap_chain_image_format.unwrap())
                    .components(vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    })
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    });
                let im_view = unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_image_view(&create_info, None)
                };
                im_view.expect("failed to create image views!")
            })
            .collect::<Vec<_>>();
    }

    fn create_render_pass(&mut self) {
        let color_attachments = [vk::AttachmentDescription {
            format: self.swap_chain_image_format.unwrap(),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        }];

        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let subpass = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .build()];

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&color_attachments)
            .subpasses(&subpass)
            .dependencies(&dependencies);

        self.render_pass = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_render_pass(&render_pass_info, None)
                    .unwrap(),
            )
        };
    }

    fn create_compute_descriptor_set_layout(&mut self) {
        let layout_bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                ..Default::default()
            },
        ];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);

        self.compute_descriptor_set_layout = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_set_layout(&layout_info, None)
                .expect("failed to create compute descriptor set layout!")
        });
    }

    fn create_graphics_pipeline(&mut self) {
        let vert_shader_module = self.create_shader_module("31_shader_compute_vert.spv".to_owned());
        let frag_shader_module = self.create_shader_module("31_shader_compute_frag.spv".to_owned());

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let vert_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(vert_shader_module)
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(shader_entry_name);

        let frag_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .module(frag_shader_module)
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(shader_entry_name);

        let shader_stages = [
            vert_shader_stage_info.build(),
            frag_shader_stage_info.build(),
        ];

        let binding_descriptions = [Particle::get_binding_description()];
        let attribute_descriptions = Particle::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::POINT_LIST)
            .primitive_restart_enable(false);

        let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);

        let rasterizer = vk::PipelineRasterizationStateCreateInfo::builder()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisampling = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            )
            .blend_enable(true)
            .color_blend_op(vk::BlendOp::ADD)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .build()];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::default();

        unsafe {
            self.pipeline_layout = Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_pipeline_layout(&pipeline_layout_info, None)
                    .expect("failed to create pipeline layout!"),
            );

            let pipeline_infos = [vk::GraphicsPipelineCreateInfo::builder()
                .stages(&shader_stages)
                .vertex_input_state(&vertex_input_info)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterizer)
                .multisample_state(&multisampling)
                .color_blend_state(&color_blending)
                .dynamic_state(&dynamic_state)
                .layout(self.pipeline_layout.unwrap())
                .render_pass(self.render_pass.unwrap())
                .subpass(0)
                .build()];

            let graphics_pipelines = self
                .device
                .as_ref()
                .unwrap()
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .expect("failed to create graphics pipeline!");
            self.graphics_pipeline = Some(graphics_pipelines[0]);

            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(vert_shader_module, None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(frag_shader_module, None);
        };
    }

    fn create_compute_pipeline(&mut self) {
        let module = self.create_shader_module("31_shader_compute_comp.spv".to_owned());

        let shader_entry_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"main\0") };
        let compute_shader_stage_info = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(shader_entry_name)
            .build();

        let binding = [self.compute_descriptor_set_layout.unwrap()];
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&binding);

        unsafe {
            let compute_pipeline_layout = self
                .device
                .as_ref()
                .unwrap()
                .create_pipeline_layout(&pipeline_layout_info, None)
                .expect("failed to create compute pipeline layout!");

            let pipeline_infos = [vk::ComputePipelineCreateInfo::builder()
                .layout(compute_pipeline_layout)
                .stage(compute_shader_stage_info)
                .build()];

            let compute_pipelines = self
                .device
                .as_ref()
                .unwrap()
                .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_infos, None)
                .expect("failed to create compute pipeline!");

            self.device
                .as_ref()
                .unwrap()
                .destroy_shader_module(module, None);

            self.compute_pipeline = Some(compute_pipelines[0]);
            self.compute_pipeline_layout = Some(compute_pipeline_layout);
        }
    }

    fn create_framebuffers(&mut self) {
        self.swap_chain_framebuffers = self
            .swap_chain_image_views
            .iter()
            .map(|view| {
                let attachments = [*view];
                let framebuffer_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(self.render_pass.unwrap())
                    .attachments(&attachments)
                    .width(self.swap_chain_extent.as_ref().unwrap().width)
                    .height(self.swap_chain_extent.as_ref().unwrap().height)
                    .layers(1);

                unsafe {
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_framebuffer(&framebuffer_info, None)
                        .expect("failed to create framebuffer!")
                }
            })
            .collect::<Vec<vk::Framebuffer>>();
    }

    fn create_command_pool(&mut self) {
        let queue_family_indices = self.find_queue_families(self.physical_device.unwrap());

        let create_info = vk::CommandPoolCreateInfo::builder()
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
            .queue_family_index(queue_family_indices.graphics_and_compute_family.unwrap());

        self.command_pool = unsafe {
            Some(
                self.device
                    .as_ref()
                    .unwrap()
                    .create_command_pool(&create_info, None)
                    .unwrap(),
            )
        };
    }

    fn create_shader_storage_buffers(&mut self) {
        let mut rng = rand::thread_rng();

        let mut particles: Vec<Particle> = Vec::with_capacity(PARTICLE_COUNT.try_into().unwrap());

        for _i in 0..PARTICLE_COUNT {
            let sq: f32 = rng.gen_range(0.0..=1.0);
            let r = 0.25 * sq.sqrt();
            let th: f32 = rng.gen_range(0.0..=1.0);
            let theta = th * 2.0 * 3.14159265358979323846;
            // let aspect = (HEIGHT / WIDTH) as f32;
            let x = r * theta.cos() * (HEIGHT as f32) / (WIDTH as f32);
            let y = r * theta.sin();

            let v = cgmath::vec2(x, y);
            let vn = v.normalize() * 0.00025;
            particles.push(Particle {
                pos: [x, y],
                velocity: [vn.x, vn.y],
                color: [
                    rng.gen_range(0.0..=1.0),
                    rng.gen_range(0.0..=1.0),
                    rng.gen_range(0.0..=1.0),
                    1.0,
                ],
            });
        }

        let buffer_size = (mem::size_of::<Particle>() * PARTICLE_COUNT as usize) as u64;

        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            buffer_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        unsafe {
            let data_ptr = self
                .device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut data_slice = Align::new(data_ptr, mem::align_of::<f32>() as u64, buffer_size);
            data_slice.copy_from_slice(&particles);
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);

            let mut ssbs: Vec<vk::Buffer> =
                Vec::with_capacity(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());
            let mut ssbms: Vec<vk::DeviceMemory> =
                Vec::with_capacity(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());
            for _i in 0..MAX_FRAMES_IN_FLIGHT {
                let (buffer, buffer_memory) = self.create_buffer(
                    buffer_size,
                    vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                );
                self.copy_buffer(staging_buffer, buffer, buffer_size);

                ssbs.push(buffer);
                ssbms.push(buffer_memory);
            }

            self.shader_storage_buffers = ssbs;
            self.shader_storage_buffers_memory = ssbms;

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);
        }
    }

    fn create_uniform_buffers(&mut self) {
        let buffer_size = (mem::size_of::<UniformBufferObject>()) as u64;

        for _i in 0..MAX_FRAMES_IN_FLIGHT {
            let (buffer, buffer_memory) = self.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            );
            unsafe {
                let ptr = self
                    .device
                    .as_ref()
                    .unwrap()
                    .map_memory(buffer_memory, 0, buffer_size, vk::MemoryMapFlags::empty())
                    .unwrap();
                self.uniform_buffers_mapped.push(ptr);
            }

            self.uniform_buffers.push(buffer);
            self.uniform_buffers_memory.push(buffer_memory);
        }
    }

    fn create_descriptor_pool(&mut self) {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: (MAX_FRAMES_IN_FLIGHT * 2) as u32,
            },
        ];

        let pool_info = vk::DescriptorPoolCreateInfo::builder()
            .pool_sizes(&pool_sizes)
            .max_sets(MAX_FRAMES_IN_FLIGHT as u32)
            .build();

        self.descriptor_pool = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_pool(&pool_info, None)
                .expect("failed to create descriptor pool!")
        });
    }

    fn create_compute_descriptor_sets(&mut self) {
        let layouts = [
            self.compute_descriptor_set_layout.unwrap(),
            self.compute_descriptor_set_layout.unwrap(),
        ];
        let alloc_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(self.descriptor_pool.unwrap())
            .set_layouts(&layouts)
            .build();
        unsafe {
            let descriptor_sets = self
                .device
                .as_ref()
                .unwrap()
                .allocate_descriptor_sets(&alloc_info)
                .expect("failed to allocate descriptor sets!");

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                let index = i as usize;
                let buffer_info = [vk::DescriptorBufferInfo {
                    buffer: self.uniform_buffers[index],
                    offset: 0,
                    range: mem::size_of::<UniformBufferObject>() as u64,
                }];

                let storage_buffer_info_last_frame = [vk::DescriptorBufferInfo {
                    buffer: self.shader_storage_buffers
                        [(index + 1) % MAX_FRAMES_IN_FLIGHT as usize],
                    offset: 0,
                    range: (mem::size_of::<Particle>() * PARTICLE_COUNT as usize) as u64,
                }];

                let storage_buffer_info_current_frame = [vk::DescriptorBufferInfo {
                    buffer: self.shader_storage_buffers[index],
                    offset: 0,
                    range: (mem::size_of::<Particle>() * PARTICLE_COUNT as usize) as u64,
                }];

                let descriptor_writes = [
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[index])
                        .dst_binding(0)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .buffer_info(&buffer_info)
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[index])
                        .dst_binding(1)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&storage_buffer_info_last_frame)
                        .build(),
                    vk::WriteDescriptorSet::builder()
                        .dst_set(descriptor_sets[index])
                        .dst_binding(2)
                        .dst_array_element(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&storage_buffer_info_current_frame)
                        .build(),
                ];

                self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&descriptor_writes, &[]);
            }

            self.compute_descriptor_sets = descriptor_sets;
        }
    }

    fn create_buffer(
        &mut self,
        size: u64,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Buffer, vk::DeviceMemory) {
        let buffer_info = vk::BufferCreateInfo {
            size,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        unsafe {
            let buffer = self
                .device
                .as_ref()
                .unwrap()
                .create_buffer(&buffer_info, None)
                .expect("failed to create buffer!");
            let mem_requirements = self
                .device
                .as_ref()
                .unwrap()
                .get_buffer_memory_requirements(buffer);
            let alloc_info = vk::MemoryAllocateInfo {
                allocation_size: mem_requirements.size,
                memory_type_index: self
                    .find_memory_type(mem_requirements.memory_type_bits, properties),
                ..Default::default()
            };

            let buffer_memory = self
                .device
                .as_ref()
                .unwrap()
                .allocate_memory(&alloc_info, None)
                .expect("failed to allocate buffer memory!");

            self.device
                .as_ref()
                .unwrap()
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .unwrap();

            return (buffer, buffer_memory);
        }
    }

    fn copy_buffer(&mut self, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: u64) {
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: self.command_pool.unwrap(),
            command_buffer_count: 1,
            ..Default::default()
        };

        unsafe {
            let command_buffers = self
                .device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .unwrap();
            let command_buffer = command_buffers[0];
            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap();
            let copy_regions = [vk::BufferCopy {
                size,
                ..Default::default()
            }];

            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &copy_regions,
            );
            self.device
                .as_ref()
                .unwrap()
                .end_command_buffer(command_buffer)
                .unwrap();

            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build()];

            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    self.graphics_queue.unwrap(),
                    &submit_infos,
                    vk::Fence::null(),
                )
                .unwrap();
            self.device
                .as_ref()
                .unwrap()
                .queue_wait_idle(self.graphics_queue.unwrap())
                .unwrap();

            self.device
                .as_ref()
                .unwrap()
                .free_command_buffers(self.command_pool.unwrap(), &command_buffers);
        }
    }

    fn find_memory_type(&mut self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
        let mem_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_memory_properties(self.physical_device.unwrap())
        };

        for i in 0..mem_properties.memory_type_count {
            let memory_type: vk::MemoryType = mem_properties.memory_types[i as usize];
            if (type_filter & (1 << i)) != 0
                && (memory_type.property_flags & properties) == properties
            {
                return i;
            }
        }

        panic!("failed to find suitable memory type!");
    }

    fn create_command_buffers(&mut self) {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool.unwrap())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());

        self.command_buffers = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .expect("failed to allocate command buffers!")
        };
    }

    fn create_compute_command_buffers(&mut self) {
        let alloc_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(self.command_pool.unwrap())
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(MAX_FRAMES_IN_FLIGHT.try_into().unwrap());

        self.compute_command_buffers = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .expect("failed to allocate command buffers!")
        };
    }

    fn record_command_buffer(&mut self, command_buffer: vk::CommandBuffer, image_index: usize) {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin recording command buffer!")
        };

        let area = vk::Rect2D::builder()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(self.swap_chain_extent.unwrap())
            .build();

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        }];
        let render_pass_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.render_pass.unwrap())
            .framebuffer(self.swap_chain_framebuffers[image_index])
            .render_area(area)
            .clear_values(&clear_values);

        unsafe {
            self.device.as_ref().unwrap().cmd_begin_render_pass(
                command_buffer,
                &render_pass_info,
                vk::SubpassContents::INLINE,
            );

            self.device.as_ref().unwrap().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.graphics_pipeline.unwrap(),
            );

            let viewports = [vk::Viewport::builder()
                .x(0.0)
                .y(0.0)
                .width(self.swap_chain_extent.as_ref().unwrap().width as f32)
                .height(self.swap_chain_extent.as_ref().unwrap().height as f32)
                .min_depth(0.0)
                .max_depth(1.0)
                .build()];

            self.device
                .as_ref()
                .unwrap()
                .cmd_set_viewport(command_buffer, 0, &viewports);

            let scissors = [vk::Rect2D::builder()
                .extent(self.swap_chain_extent.unwrap())
                .offset(vk::Offset2D { x: 0, y: 0 })
                .build()];

            self.device
                .as_ref()
                .unwrap()
                .cmd_set_scissor(command_buffer, 0, &scissors);

            let buffers = [self.shader_storage_buffers[self.current_frame]];
            let offsets = [0];
            self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &buffers,
                &offsets,
            );

            self.device
                .as_ref()
                .unwrap()
                .cmd_draw(command_buffer, PARTICLE_COUNT, 1, 0, 0);

            self.device
                .as_ref()
                .unwrap()
                .cmd_end_render_pass(command_buffer);
            self.device
                .as_ref()
                .unwrap()
                .end_command_buffer(command_buffer)
                .expect("failed to record command buffer!");
        }
    }

    fn record_compute_command_buffer(&mut self, command_buffer: vk::CommandBuffer) {
        let begin_info = vk::CommandBufferBeginInfo::default();

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("failed to begin recording compute command buffer!");

            self.device.as_ref().unwrap().cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline.unwrap(),
            );

            let descriptor_sets = [self.compute_descriptor_sets[self.current_frame]];
            self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.compute_pipeline_layout.unwrap(),
                0,
                &descriptor_sets,
                &[],
            );

            self.device
                .as_ref()
                .unwrap()
                .cmd_dispatch(command_buffer, PARTICLE_COUNT / 256, 1, 1);

            self.device
                .as_ref()
                .unwrap()
                .end_command_buffer(command_buffer)
                .expect("failed to record compute command buffer!");
        }
    }

    fn create_sync_objects(&mut self) {
        let semaphore_info = vk::SemaphoreCreateInfo::default();

        let fence_info = vk::FenceCreateInfo::builder()
            .flags(vk::FenceCreateFlags::SIGNALED)
            .build();

        for _i in 0..MAX_FRAMES_IN_FLIGHT {
            unsafe {
                self.image_available_semaphores.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_semaphore(&semaphore_info, None)
                        .expect("failed to create synchronization objects for a frame!"),
                );
                self.render_finished_semaphores.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_semaphore(&semaphore_info, None)
                        .expect("failed to create synchronization objects for a frame!"),
                );
                self.compute_finished_semaphores.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_semaphore(&semaphore_info, None)
                        .expect("failed to create compute synchronization objects for a frame!"),
                );
                self.in_flight_fences.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_fence(&fence_info, None)
                        .expect("failed to create synchronization objects for a frame!"),
                );
                self.compute_in_flight_fences.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_fence(&fence_info, None)
                        .expect("failed to create compute synchronization objects for a frame!"),
                );
            }
        }
    }

    fn update_uniform_buffer(&mut self, current_image: usize) {
        // let time = self.start_time.elapsed().as_secs_f32();

        let ubos = [UniformBufferObject {
            delta_time: self.last_frame_time * 0.2,
        }];

        let size = mem::size_of::<UniformBufferObject>() as u64;
        let mut align = unsafe {
            Align::new(
                self.uniform_buffers_mapped[current_image],
                mem::align_of::<f32>() as u64,
                size,
            )
        };
        align.copy_from_slice(&ubos);
    }

    fn draw_frame(&mut self) {
        unsafe {
            let comp_fences = [self.compute_in_flight_fences[self.current_frame]];
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&comp_fences, true, u64::MAX)
                .unwrap();

            self.update_uniform_buffer(self.current_frame);

            self.device
                .as_ref()
                .unwrap()
                .reset_fences(&comp_fences)
                .unwrap();

            self.device
                .as_ref()
                .unwrap()
                .reset_command_buffer(
                    self.compute_command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
            self.record_compute_command_buffer(self.compute_command_buffers[self.current_frame]);

            let comp_submit_info = [vk::SubmitInfo::builder()
                .command_buffers(&[self.compute_command_buffers[self.current_frame]])
                .signal_semaphores(&[self.compute_finished_semaphores[self.current_frame]])
                .build()];

            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    self.compute_queue.unwrap(),
                    &comp_submit_info,
                    self.compute_in_flight_fences[self.current_frame],
                )
                .unwrap();

            // Graphics submission
            let fences = [self.in_flight_fences[self.current_frame]];
            self.device
                .as_ref()
                .unwrap()
                .wait_for_fences(&fences, true, u64::MAX)
                .unwrap();

            let im_result = self.swapchain_loader.as_ref().unwrap().acquire_next_image(
                self.swap_chain.unwrap(),
                u64::MAX,
                self.image_available_semaphores[self.current_frame],
                vk::Fence::null(),
            );

            if im_result.err() == Some(vk::Result::ERROR_OUT_OF_DATE_KHR) {
                self.recreate_swap_chain();
                return;
            }

            self.device
                .as_ref()
                .unwrap()
                .reset_fences(&fences)
                .expect("failed to acquire swap chain image!");

            let (image_index, _) = im_result.unwrap();

            self.device
                .as_ref()
                .unwrap()
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::RELEASE_RESOURCES,
                )
                .unwrap();
            self.record_command_buffer(
                self.command_buffers[self.current_frame],
                image_index.try_into().unwrap(),
            );

            let command_buffers = [self.command_buffers[self.current_frame]];
            let wait_semaphores = [
                self.compute_finished_semaphores[self.current_frame],
                self.image_available_semaphores[self.current_frame],
            ];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
            let wait_dst_stage_mask = [
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            ];
            let submit_infos = [vk::SubmitInfo::builder()
                .wait_semaphores(&wait_semaphores)
                .wait_dst_stage_mask(&wait_dst_stage_mask)
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build()];

            self.device
                .as_ref()
                .unwrap()
                .queue_submit(
                    self.graphics_queue.unwrap(),
                    &submit_infos,
                    self.in_flight_fences[self.current_frame],
                )
                .expect("failed to submit draw command buffer!");

            let swapchains = [self.swap_chain.unwrap()];
            let image_indices = [image_index];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(&signal_semaphores)
                .swapchains(&swapchains)
                .image_indices(&image_indices);

            let result = self
                .swapchain_loader
                .as_ref()
                .unwrap()
                .queue_present(self.present_queue.unwrap(), &present_info);

            if self.framebuffer_resized
                || result.err() == Some(vk::Result::ERROR_OUT_OF_DATE_KHR)
                || result.err() == Some(vk::Result::SUBOPTIMAL_KHR)
            {
                self.framebuffer_resized = false;
                self.recreate_swap_chain();
            }

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT as usize;
        }
    }

    fn create_shader_module(&mut self, shader_name: String) -> vk::ShaderModule {
        let exe_path = env::current_exe().unwrap();
        // let exe_name = exe_path
        //     .file_name()
        //     .unwrap()
        //     .to_str()
        //     .unwrap()
        //     .split('.')
        //     .collect::<Vec<_>>()[0];

        // let vert_shader_file_name = format!("{}_{}.spv", exe_name, shader_name);
        let vert_shader_file_path = exe_path.parent().unwrap().join(&shader_name);
        if !vert_shader_file_path.exists() {
            panic!("{}", format!("{:?} 不存在", vert_shader_file_path));
        }
        let mut vert_shader_file = Cursor::new(fs::read(vert_shader_file_path).unwrap());
        let vert_shader_code = self.read_spv(&mut vert_shader_file).unwrap();

        let vert_shader_module_create_info =
            vk::ShaderModuleCreateInfo::builder().code(&vert_shader_code);

        let shader_module = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_shader_module(&vert_shader_module_create_info, None)
                .unwrap()
        };
        shader_module
    }

    fn choose_swap_surface_format<'a>(
        &'a mut self,
        available_formats: &'a Vec<vk::SurfaceFormatKHR>,
    ) -> SurfaceFormatKHR {
        for format in available_formats {
            if format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                return *format;
            }
        }
        available_formats[0]
    }

    fn choose_swap_present_mode(
        &mut self,
        available_present_modes: &Vec<vk::PresentModeKHR>,
    ) -> vk::PresentModeKHR {
        for mode in available_present_modes {
            if mode == &vk::PresentModeKHR::MAILBOX {
                return *mode;
            }
        }
        vk::PresentModeKHR::FIFO
    }

    fn choose_swap_extent(&mut self, capabilities: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
        if capabilities.current_extent.width != std::u32::MAX {
            return capabilities.current_extent;
        } else {
            let win_size = self.window.as_ref().unwrap().inner_size();

            let mut actual_extent = vk::Extent2D {
                width: win_size.width,
                height: win_size.height,
            };

            actual_extent.width = actual_extent.width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            );
            actual_extent.height = actual_extent.height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            );

            return actual_extent;
        }
    }

    fn query_swap_chain_support(&mut self, device: vk::PhysicalDevice) -> SwapChainSupportDetails {
        let mut details = SwapChainSupportDetails::new();
        unsafe {
            let surface_loader = self.surface_loader.as_ref().unwrap();
            let capabilities = surface_loader
                .get_physical_device_surface_capabilities(device, self.surface.unwrap())
                .unwrap();

            details.formats = surface_loader
                .get_physical_device_surface_formats(device, self.surface.unwrap())
                .unwrap();

            details.present_modes = surface_loader
                .get_physical_device_surface_present_modes(device, self.surface.unwrap())
                .unwrap();

            details.capabilities = Some(capabilities);
        };
        details
    }

    fn is_device_suitable(&mut self, device: vk::PhysicalDevice) -> bool {
        let mut indices = self.find_queue_families(device);

        let extensions_supported = self.check_device_extension_support(device);

        let mut swap_chain_adequate = false;
        if extensions_supported {
            let swap_chain_support = self.query_swap_chain_support(device);
            swap_chain_adequate = !swap_chain_support.formats.is_empty()
                && !swap_chain_support.present_modes.is_empty();
        }

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(&mut self, device: vk::PhysicalDevice) -> bool {
        let available_extensions = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .enumerate_device_extension_properties(device)
                .unwrap()
        };

        let available_extension_names = available_extensions
            .iter()
            .map(|prop| unsafe { CStr::from_ptr(prop.extension_name.as_ptr()) })
            .collect::<Vec<_>>();
        // println!("available extensions:");
        // available_extensions.iter().for_each(|prop| {
        //     println!("{:?}", unsafe {
        //         CStr::from_ptr(prop.extension_name.as_ptr())
        //     })
        // });
        DEVICE_EXTENSIONS.iter().all(|name| {
            available_extension_names.contains(unsafe { &CStr::from_ptr(name.clone()) })
        })
    }

    fn find_queue_families(&mut self, device: vk::PhysicalDevice) -> QueueFamilyIndices {
        let queue_families = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_queue_family_properties(device)
        };

        let mut indices = QueueFamilyIndices::new();
        let mut i: u32 = 0;
        for queue_family in queue_families {
            if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                indices.graphics_and_compute_family = Some(i);
            }

            let present_support = unsafe {
                self.surface_loader
                    .as_ref()
                    .unwrap()
                    .get_physical_device_surface_support(device, i, self.surface.unwrap())
                    .unwrap()
            };

            if present_support {
                indices.present_family = Some(i);
            }

            if indices.is_complete() {
                break;
            }

            i += 1;
        }

        indices
    }

    fn check_validation_layer_support(&mut self, entry: &Entry) -> bool {
        let available_layers = entry.enumerate_instance_layer_properties().unwrap();

        for layer_name in VALIDATION_LAYERS {
            let mut layer_found = false;

            for layer_properties in &available_layers {
                let name = unsafe {
                    CStr::from_ptr(layer_properties.layer_name.as_ptr())
                        .to_str()
                        .unwrap()
                };
                if name == layer_name.to_string() {
                    layer_found = true;
                    break;
                }
            }

            if !layer_found {
                return false;
            }
        }
        true
    }

    fn read_spv<R: io::Read + io::Seek>(&mut self, x: &mut R) -> io::Result<Vec<u32>> {
        // TODO use stream_len() once it is stabilized and remove the subsequent rewind() call
        let size = x.seek(io::SeekFrom::End(0))?;
        x.rewind()?;
        if size % 4 != 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input length not divisible by 4",
            ));
        }
        if size > usize::max_value() as u64 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "input too long"));
        }
        let words = (size / 4) as usize;
        // https://github.com/MaikKlein/ash/issues/354:
        // Zero-initialize the result to prevent read_exact from possibly
        // reading uninitialized memory.
        let mut result = vec![0u32; words];
        x.read_exact(unsafe {
            slice::from_raw_parts_mut(result.as_mut_ptr().cast::<u8>(), words * 4)
        })?;
        const MAGIC_NUMBER: u32 = 0x0723_0203;
        if !result.is_empty() && result[0] == MAGIC_NUMBER.swap_bytes() {
            for word in &mut result {
                *word = word.swap_bytes();
            }
        }
        if result.is_empty() || result[0] != MAGIC_NUMBER {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "input missing SPIR-V magic number",
            ));
        }
        Ok(result)
    }
}

impl Drop for ComputeShaderApplication {
    fn drop(&mut self) {
        unsafe {
            self.device.as_ref().unwrap().device_wait_idle().unwrap();

            self.cleanup_swap_chain();

            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline(self.graphics_pipeline.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline_layout(self.pipeline_layout.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline(self.compute_pipeline.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_pipeline_layout(self.compute_pipeline_layout.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_render_pass(self.render_pass.unwrap(), None);

            self.uniform_buffers.iter().for_each(|buffer| {
                self.device.as_ref().unwrap().destroy_buffer(*buffer, None);
            });

            self.uniform_buffers_memory.iter().for_each(|memory| {
                self.device.as_ref().unwrap().free_memory(*memory, None);
            });

            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_pool(self.descriptor_pool.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_set_layout(self.compute_descriptor_set_layout.unwrap(), None);

            self.shader_storage_buffers.iter().for_each(|buffer| {
                self.device.as_ref().unwrap().destroy_buffer(*buffer, None);
            });

            self.shader_storage_buffers_memory
                .iter()
                .for_each(|memory| {
                    self.device.as_ref().unwrap().free_memory(*memory, None);
                });

            self.image_available_semaphores
                .iter()
                .for_each(|semaphore| {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_semaphore(*semaphore, None);
                });
            self.render_finished_semaphores
                .iter()
                .for_each(|semaphore| {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_semaphore(*semaphore, None);
                });
            self.compute_finished_semaphores
                .iter()
                .for_each(|semaphore| {
                    self.device
                        .as_ref()
                        .unwrap()
                        .destroy_semaphore(*semaphore, None);
                });
            self.in_flight_fences.iter().for_each(|fence| {
                self.device.as_ref().unwrap().destroy_fence(*fence, None);
            });
            self.compute_in_flight_fences.iter().for_each(|fence| {
                self.device.as_ref().unwrap().destroy_fence(*fence, None);
            });

            self.device
                .as_ref()
                .unwrap()
                .destroy_command_pool(self.command_pool.unwrap(), None);

            self.device.as_ref().unwrap().destroy_device(None);
            self.surface_loader
                .as_ref()
                .unwrap()
                .destroy_surface(self.surface.unwrap(), None);
            self.debug_utils
                .as_ref()
                .unwrap()
                .destroy_debug_utils_messenger(self.debug_messenger.unwrap(), None);
            self.instance.as_ref().unwrap().destroy_instance(None);
        };

        self.swapchain_loader = None;
        self.surface_loader = None;
        self.vk_entry = None;
    }
}

fn main() {
    let mut app = ComputeShaderApplication::new();
    app.run();
}
