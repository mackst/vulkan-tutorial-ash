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

use tobj;

use cgmath::{perspective, Deg, Matrix4, Point3, Vector3};
use image::io::Reader as ImageReader;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const MODEL_NAME: &str = "viking_room.obj";
const TEXTURE_NAME: &str = "viking_room.png";

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
    graphics_family: Option<u32>,
    present_family: Option<u32>,
}

impl QueueFamilyIndices {
    pub fn new() -> Self {
        Self {
            graphics_family: None,
            present_family: None,
        }
    }

    fn is_complete(&mut self) -> bool {
        self.graphics_family.is_some() && self.present_family.is_some()
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
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
    tex_coord: [f32; 2],
}

impl Vertex {
    pub fn get_binding_description() -> vk::VertexInputBindingDescription {
        vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<Self>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }
    }

    pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 3] {
        [
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, pos) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: offset_of!(Self, color) as u32,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 2,
                format: vk::Format::R32G32_SFLOAT,
                offset: offset_of!(Self, tex_coord) as u32,
            },
        ]
    }
}

#[derive(Clone, Debug, Copy)]
#[allow(dead_code)]
struct UniformBufferObject {
    model: Matrix4<f32>,
    view: Matrix4<f32>,
    proj: Matrix4<f32>,
}

#[allow(unused)]
struct HelloTriangleApplication {
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
    present_queue: Option<vk::Queue>,

    swapchain_loader: Option<Swapchain>,
    swap_chain: Option<vk::SwapchainKHR>,
    swap_chain_images: Vec<vk::Image>,
    swap_chain_image_format: Option<vk::Format>,
    swap_chain_extent: Option<vk::Extent2D>,
    swap_chain_image_views: Vec<vk::ImageView>,
    swap_chain_framebuffers: Vec<vk::Framebuffer>,

    render_pass: Option<vk::RenderPass>,
    descriptor_set_layout: Option<vk::DescriptorSetLayout>,
    pipeline_layout: Option<vk::PipelineLayout>,
    graphics_pipeline: Option<vk::Pipeline>,

    command_pool: Option<vk::CommandPool>,

    depth_image: Option<vk::Image>,
    depth_image_memory: Option<vk::DeviceMemory>,
    depth_image_view: Option<vk::ImageView>,

    mip_levels: u32,
    texture_image: Option<vk::Image>,
    texture_image_memory: Option<vk::DeviceMemory>,
    texture_image_view: Option<vk::ImageView>,
    texture_sampler: Option<vk::Sampler>,

    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    vertex_buffer: Option<vk::Buffer>,
    vertex_buffer_memory: Option<vk::DeviceMemory>,
    index_buffer: Option<vk::Buffer>,
    index_buffer_memory: Option<vk::DeviceMemory>,

    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    uniform_buffers_mapped: Vec<*mut c_void>,

    descriptor_pool: Option<vk::DescriptorPool>,
    descriptor_sets: Vec<vk::DescriptorSet>,

    command_buffers: Vec<vk::CommandBuffer>,

    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    current_frame: usize,

    framebuffer_resized: bool,

    start_time: Instant,
}

impl HelloTriangleApplication {
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
            present_queue: None,
            swap_chain: None,
            swapchain_loader: None,
            swap_chain_images: [].to_vec(),
            swap_chain_image_format: None,
            swap_chain_extent: None,
            swap_chain_image_views: [].to_vec(),
            swap_chain_framebuffers: [].to_vec(),
            render_pass: None,
            descriptor_set_layout: None,
            pipeline_layout: None,
            graphics_pipeline: None,
            command_pool: None,
            depth_image: None,
            depth_image_memory: None,
            depth_image_view: None,
            mip_levels: 0,
            texture_image: None,
            texture_image_memory: None,
            texture_image_view: None,
            texture_sampler: None,
            vertices: [].to_vec(),
            indices: [].to_vec(),
            vertex_buffer: None,
            vertex_buffer_memory: None,
            index_buffer: None,
            index_buffer_memory: None,
            uniform_buffers: [].to_vec(),
            uniform_buffers_memory: [].to_vec(),
            uniform_buffers_mapped: [].to_vec(),
            descriptor_pool: None,
            descriptor_sets: [].to_vec(),
            command_buffers: [].to_vec(),
            image_available_semaphores: [].to_vec(),
            render_finished_semaphores: [].to_vec(),
            in_flight_fences: [].to_vec(),
            current_frame: 0,
            framebuffer_resized: false,
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
        self.create_descriptor_set_layout();
        self.create_graphics_pipeline();
        self.create_command_pool();
        self.create_depth_resources();
        self.create_framebuffers();
        self.create_texture_image();
        self.create_texture_image_view();
        self.create_texture_sampler();
        self.load_model();
        self.create_vertex_buffer();
        self.create_index_buffer();
        self.create_uniform_buffers();
        self.create_descriptor_pool();
        self.create_descriptor_sets();
        self.create_command_buffers();
        self.create_sync_objects();
    }

    fn init_window(&mut self, event_loop: &EventLoop<()>) {
        let window = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            // .with_resizable(false)
            .build(event_loop)
            .unwrap();
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
                }
                _ => (),
            }
        });

        // unsafe { self.device.as_ref().unwrap().device_wait_idle().unwrap() };
    }

    fn cleanup_swap_chain(&mut self) {
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_image_view(self.depth_image_view.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_image(self.depth_image.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.depth_image_memory.unwrap(), None);

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
        self.create_depth_resources();
        self.create_framebuffers();
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
            indices.graphics_family.unwrap(),
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

        let device_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .build();
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
            let graphics_queue = device.get_device_queue(indices.graphics_family.unwrap(), 0);
            let present_queue = device.get_device_queue(indices.present_family.unwrap(), 0);
            self.device = Some(device);
            self.graphics_queue = Some(graphics_queue);
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
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        if indices.graphics_family != indices.present_family {
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
        let mut image_views = Vec::new();
        for i in 0..self.swap_chain_images.len() {
            image_views.push(self.create_image_view(
                self.swap_chain_images[i],
                self.swap_chain_image_format.unwrap(),
                vk::ImageAspectFlags::COLOR,
                1,
            ));
        }
        self.swap_chain_image_views = image_views;
    }

    fn create_render_pass(&mut self) {
        let color_attachment = vk::AttachmentDescription {
            format: self.swap_chain_image_format.unwrap(),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::STORE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
            ..Default::default()
        };

        let depth_attachment = vk::AttachmentDescription {
            format: self.find_depth_format(),
            samples: vk::SampleCountFlags::TYPE_1,
            load_op: vk::AttachmentLoadOp::CLEAR,
            store_op: vk::AttachmentStoreOp::DONT_CARE,
            stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
            stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ..Default::default()
        };

        let color_attachment_refs = [vk::AttachmentReference {
            attachment: 0,
            layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        }];

        let depth_attachment_ref = vk::AttachmentReference {
            attachment: 1,
            layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        let subpass = [vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_refs)
            .depth_stencil_attachment(&depth_attachment_ref)
            .build()];

        let dependencies = [vk::SubpassDependency {
            src_subpass: vk::SUBPASS_EXTERNAL,
            dst_subpass: 0,
            src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            // src_access_mask: vk::AccessFlags::empty(),
            dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            ..Default::default()
        }];

        let attachments = [color_attachment, depth_attachment];
        let render_pass_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
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

    fn create_descriptor_set_layout(&mut self) {
        let ubo_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        };

        let sampler_layout_binding = vk::DescriptorSetLayoutBinding {
            binding: 1,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            ..Default::default()
        };

        let bindings = [ubo_layout_binding, sampler_layout_binding];

        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .build();

        self.descriptor_set_layout = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_descriptor_set_layout(&layout_info, None)
                .expect("failed to create descriptor set layout!")
        });
    }

    fn create_graphics_pipeline(&mut self) {
        let vert_shader_module = self.create_shader_module("27_shader_depth_vert.spv".to_owned());
        let frag_shader_module = self.create_shader_module("27_shader_depth_frag.spv".to_owned());

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

        let binding_descriptions = [Vertex::get_binding_description()];
        let attribute_descriptions = Vertex::get_attribute_descriptions();
        let vertex_input_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&binding_descriptions)
            .vertex_attribute_descriptions(&attribute_descriptions);

        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
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

        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        let color_blend_attachment = [vk::PipelineColorBlendAttachmentState {
            color_write_mask: vk::ColorComponentFlags::RGBA,
            blend_enable: 0,
            ..Default::default()
        }];

        let color_blending = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(&color_blend_attachment)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let set_layouts = [self.descriptor_set_layout.unwrap()];
        let pipeline_layout_info =
            vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

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
                .depth_stencil_state(&depth_stencil)
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

    fn create_framebuffers(&mut self) {
        self.swap_chain_framebuffers = self
            .swap_chain_image_views
            .iter()
            .map(|view| {
                let attachments = [*view, self.depth_image_view.unwrap()];
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
            .queue_family_index(queue_family_indices.graphics_family.unwrap());

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

    fn create_depth_resources(&mut self) {
        let depth_format = self.find_depth_format();

        let (image, image_memory) = self.create_image(
            self.swap_chain_extent.as_ref().unwrap().width,
            self.swap_chain_extent.as_ref().unwrap().height,
            1,
            depth_format,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.depth_image_view =
            Some(self.create_image_view(image, depth_format, vk::ImageAspectFlags::DEPTH, 1));

        self.depth_image = Some(image);
        self.depth_image_memory = Some(image_memory);
    }

    fn find_supported_format(
        &mut self,
        candidates: &Vec<vk::Format>,
        tiling: vk::ImageTiling,
        features: vk::FormatFeatureFlags,
    ) -> Result<vk::Format, ()> {
        for format in candidates {
            let props = unsafe {
                self.instance
                    .as_ref()
                    .unwrap()
                    .get_physical_device_format_properties(self.physical_device.unwrap(), *format)
            };

            if tiling == vk::ImageTiling::LINEAR
                && (props.linear_tiling_features & features) == features
            {
                return Ok(*format);
            } else if tiling == vk::ImageTiling::OPTIMAL
                && (props.optimal_tiling_features & features) == features
            {
                return Ok(*format);
            }
        }

        panic!("failed to find supported format!");
    }

    fn find_depth_format(&mut self) -> vk::Format {
        self.find_supported_format(
            &vec![
                vk::Format::D32_SFLOAT,
                vk::Format::D32_SFLOAT_S8_UINT,
                vk::Format::D24_UNORM_S8_UINT,
            ],
            vk::ImageTiling::OPTIMAL,
            vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
        )
        .unwrap()
    }

    fn has_stencil_component(&mut self, format: vk::Format) -> bool {
        format == vk::Format::D32_SFLOAT_S8_UINT || format == vk::Format::D24_UNORM_S8_UINT
    }

    fn create_texture_image(&mut self) {
        let exe_path = env::current_exe().unwrap();
        let texture_path = exe_path.parent().unwrap().join(TEXTURE_NAME);
        let image = ImageReader::open(texture_path)
            .expect("failed to load texture image!")
            .decode()
            .unwrap();
        let width = image.width();
        let height = image.height();
        let image_size = (width * height * 4) as vk::DeviceSize;
        //mipLevels = static_cast<uint32_t>(std::floor(std::log2(std::max(texWidth, texHeight)))) + 1;
        let mip_levels = (f32::floor(f32::log2(u32::max(width, height) as f32))) as u32 + 1;

        let (staging_buffer, staging_buffer_memory) = self.create_buffer(
            image_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );

        let pixels = image.to_rgba8().into_raw();

        unsafe {
            let data_ptr = self
                .device
                .as_ref()
                .unwrap()
                .map_memory(
                    staging_buffer_memory,
                    0,
                    image_size,
                    vk::MemoryMapFlags::empty(),
                )
                .unwrap();
            let mut data_slice = Align::new(data_ptr, mem::align_of::<u8>() as u64, image_size);
            data_slice.copy_from_slice(&pixels);
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);
        }

        let (image, image_memory) = self.create_image(
            width,
            height,
            mip_levels,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageTiling::OPTIMAL,
            vk::ImageUsageFlags::TRANSFER_SRC
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        self.transition_image_layout(
            image,
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            mip_levels,
        );
        self.copy_buffer_to_image(staging_buffer, image, width, height);

        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);
        }

        self.generate_mipmaps(image, vk::Format::R8G8B8A8_SRGB, width, height, mip_levels);

        self.texture_image = Some(image);
        self.texture_image_memory = Some(image_memory);
    }

    fn generate_mipmaps(
        &mut self,
        image: vk::Image,
        image_format: vk::Format,
        width: u32,
        height: u32,
        mip_levels: u32,
    ) {
        let format_properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_format_properties(self.physical_device.unwrap(), image_format)
        };

        if !format_properties
            .optimal_tiling_features
            .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
        {
            panic!("texture image format does not support linear blitting!");
        }

        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier {
            image,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_array_layer: 0,
                layer_count: 1,
                level_count: 1,
                base_mip_level: 0,
            },
            ..Default::default()
        };

        let mut mip_width = width as i32;
        let mut mip_height = height as i32;
        for i in 1..mip_levels {
            barrier.subresource_range.base_mip_level = i - 1;
            barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_READ;

            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            let dst_x = if mip_width > 1 { mip_width / 2 } else { 1 };
            let dst_y = if mip_height > 1 { mip_height / 2 } else { 1 };
            let blit = vk::ImageBlit {
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: mip_width,
                        y: mip_height,
                        z: 1,
                    },
                ],
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i - 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: dst_x,
                        y: dst_y,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: i,
                    base_array_layer: 0,
                    layer_count: 1,
                },
            };

            unsafe {
                self.device.as_ref().unwrap().cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[blit],
                    vk::Filter::LINEAR,
                );
            }

            barrier.old_layout = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
            barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_READ;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            unsafe {
                self.device.as_ref().unwrap().cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }

            if mip_width > 1 {
                mip_width = mip_width / 2;
            }
            if mip_height > 1 {
                mip_height = mip_height / 2;
            }
        }

        barrier.subresource_range.base_mip_level = mip_levels - 1;
        barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
        barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
        barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );
        }

        self.end_single_time_commands(command_buffer);
    }

    fn create_texture_image_view(&mut self) {
        self.texture_image_view = Some(self.create_image_view(
            self.texture_image.unwrap(),
            vk::Format::R8G8B8A8_SRGB,
            vk::ImageAspectFlags::COLOR,
            self.mip_levels,
        ));
    }

    fn create_texture_sampler(&mut self) {
        let properties = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_properties(self.physical_device.unwrap())
        };

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(properties.limits.max_sampler_anisotropy)
            .border_color(vk::BorderColor::INT_OPAQUE_BLACK)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .min_lod(0.0)
            .max_lod(self.mip_levels as f32)
            .mip_lod_bias(0.0)
            .build();

        self.texture_sampler = Some(unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_sampler(&sampler_info, None)
                .expect("failed to create texture sampler!")
        });
    }

    fn create_image_view(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        aspect_flags: vk::ImageAspectFlags,
        mip_levels: u32,
    ) -> vk::ImageView {
        let view_info = vk::ImageViewCreateInfo {
            image,
            view_type: vk::ImageViewType::TYPE_2D,
            format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: aspect_flags,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let image_view = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .create_image_view(&view_info, None)
                .expect("failed to create texture image view!")
        };
        image_view
    }

    fn create_image(
        &mut self,
        width: u32,
        height: u32,
        mip_levels: u32,
        format: vk::Format,
        tiling: vk::ImageTiling,
        usage: vk::ImageUsageFlags,
        properties: vk::MemoryPropertyFlags,
    ) -> (vk::Image, vk::DeviceMemory) {
        let image_info = vk::ImageCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            format,
            tiling,
            initial_layout: vk::ImageLayout::UNDEFINED,
            usage,
            samples: vk::SampleCountFlags::TYPE_1,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        unsafe {
            let image = self
                .device
                .as_ref()
                .unwrap()
                .create_image(&image_info, None)
                .expect("failed to create image!");

            let mem_requirements = self
                .device
                .as_ref()
                .unwrap()
                .get_image_memory_requirements(image);

            let alloc_info = vk::MemoryAllocateInfo {
                allocation_size: mem_requirements.size,
                memory_type_index: self
                    .find_memory_type(mem_requirements.memory_type_bits, properties),
                ..Default::default()
            };

            let image_memory = self
                .device
                .as_ref()
                .unwrap()
                .allocate_memory(&alloc_info, None)
                .expect("failed to allocate image memory!");

            self.device
                .as_ref()
                .unwrap()
                .bind_image_memory(image, image_memory, 0)
                .unwrap();

            return (image, image_memory);
        }
    }

    fn transition_image_layout(
        &mut self,
        image: vk::Image,
        format: vk::Format,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        mip_levels: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let mut barrier = vk::ImageMemoryBarrier {
            old_layout,
            new_layout,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: mip_levels,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };

        let source_stage: vk::PipelineStageFlags;
        let destination_stage: vk::PipelineStageFlags;
        if old_layout == vk::ImageLayout::UNDEFINED
            && new_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::empty();
            barrier.dst_access_mask = vk::AccessFlags::TRANSFER_WRITE;

            source_stage = vk::PipelineStageFlags::TOP_OF_PIPE;
            destination_stage = vk::PipelineStageFlags::TRANSFER;
        } else if old_layout == vk::ImageLayout::TRANSFER_DST_OPTIMAL
            && new_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        {
            barrier.src_access_mask = vk::AccessFlags::TRANSFER_WRITE;
            barrier.dst_access_mask = vk::AccessFlags::SHADER_READ;

            source_stage = vk::PipelineStageFlags::TRANSFER;
            destination_stage = vk::PipelineStageFlags::FRAGMENT_SHADER;
        } else {
            panic!("unsupported layout transition!");
        }

        let barriers = [barrier];
        unsafe {
            self.device.as_ref().unwrap().cmd_pipeline_barrier(
                command_buffer,
                source_stage,
                destination_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &barriers,
            );
        }

        self.end_single_time_commands(command_buffer);
    }

    fn copy_buffer_to_image(
        &mut self,
        buffer: vk::Buffer,
        image: vk::Image,
        width: u32,
        height: u32,
    ) {
        let command_buffer = self.begin_single_time_commands();

        let regions = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: vk::Extent3D {
                width: width,
                height: height,
                depth: 1,
            },
        }];

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer_to_image(
                command_buffer,
                buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            )
        };

        self.end_single_time_commands(command_buffer);
    }

    fn load_model(&mut self) {
        let exe_path = env::current_exe().unwrap();
        let model_path = exe_path.parent().unwrap().join(MODEL_NAME);
        let (models, _materials) = tobj::load_obj(model_path, &tobj::GPU_LOAD_OPTIONS).unwrap();

        let mesh = &models[0].mesh;
        let vertex_count = mesh.positions.len() / 3;
        let mut vertices = Vec::with_capacity(vertex_count);

        for i in 0..vertex_count {
            let x = mesh.positions[i * 3 + 0];
            let y = mesh.positions[i * 3 + 1];
            let z = mesh.positions[i * 3 + 2];
            let u = mesh.texcoords[i * 2 + 0];
            let v = mesh.texcoords[i * 2 + 1];

            let vertex = Vertex {
                pos: [x, y, z],
                color: [1.0, 1.0, 1.0],
                tex_coord: [u, v],
            };

            vertices.push(vertex);
        }

        self.vertices = vertices;
        self.indices = mesh.indices.clone();
    }

    fn create_vertex_buffer(&mut self) {
        let buffer_size = (mem::size_of::<Vertex>() * self.vertices.len()) as u64;

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
            let mut data_slice =
                Align::new(data_ptr, mem::align_of::<Vertex>() as u64, buffer_size);
            data_slice.copy_from_slice(&self.vertices);
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);

            let (vertex_buffer, vertex_buffer_memory) = self.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            self.copy_buffer(staging_buffer, vertex_buffer, buffer_size);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);

            self.vertex_buffer = Some(vertex_buffer);
            self.vertex_buffer_memory = Some(vertex_buffer_memory);
        }
    }

    fn create_index_buffer(&mut self) {
        let buffer_size = (mem::size_of::<u32>() * self.indices.len()) as u64;

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
            let mut data_slice = Align::new(data_ptr, mem::align_of::<u16>() as u64, buffer_size);
            data_slice.copy_from_slice(&self.indices);
            self.device
                .as_ref()
                .unwrap()
                .unmap_memory(staging_buffer_memory);

            let (index_buffer, index_buffer_memory) = self.create_buffer(
                buffer_size,
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            self.copy_buffer(staging_buffer, index_buffer, buffer_size);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(staging_buffer, None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(staging_buffer_memory, None);

            self.index_buffer = Some(index_buffer);
            self.index_buffer_memory = Some(index_buffer_memory);
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
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: MAX_FRAMES_IN_FLIGHT as u32,
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

    fn create_descriptor_sets(&mut self) {
        let layouts = [
            self.descriptor_set_layout.unwrap(),
            self.descriptor_set_layout.unwrap(),
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

                let image_info = [vk::DescriptorImageInfo {
                    image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image_view: self.texture_image_view.unwrap(),
                    sampler: self.texture_sampler.unwrap(),
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
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .image_info(&image_info)
                        .build(),
                ];

                self.device
                    .as_ref()
                    .unwrap()
                    .update_descriptor_sets(&descriptor_writes, &[]);
            }

            self.descriptor_sets = descriptor_sets;
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

    fn begin_single_time_commands(&mut self) -> vk::CommandBuffer {
        let alloc_info = vk::CommandBufferAllocateInfo {
            level: vk::CommandBufferLevel::PRIMARY,
            command_pool: self.command_pool.unwrap(),
            command_buffer_count: 1,
            ..Default::default()
        };

        let command_buffers = unsafe {
            self.device
                .as_ref()
                .unwrap()
                .allocate_command_buffers(&alloc_info)
                .unwrap()
        };

        let command_buffer = command_buffers[0];
        let begin_info = vk::CommandBufferBeginInfo {
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            self.device
                .as_ref()
                .unwrap()
                .begin_command_buffer(command_buffer, &begin_info)
                .unwrap()
        };

        command_buffer
    }

    fn end_single_time_commands(&mut self, command_buffer: vk::CommandBuffer) {
        unsafe {
            let command_buffers = [command_buffer];
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

    fn copy_buffer(&mut self, src_buffer: vk::Buffer, dst_buffer: vk::Buffer, size: u64) {
        let command_buffer = self.begin_single_time_commands();

        let copy_regions = [vk::BufferCopy {
            size,
            ..Default::default()
        }];

        unsafe {
            self.device.as_ref().unwrap().cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &copy_regions,
            )
        };

        self.end_single_time_commands(command_buffer);
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

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];

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

            let buffers = [self.vertex_buffer.unwrap()];
            let offsets = [0];
            self.device.as_ref().unwrap().cmd_bind_vertex_buffers(
                command_buffer,
                0,
                &buffers,
                &offsets,
            );

            self.device.as_ref().unwrap().cmd_bind_index_buffer(
                command_buffer,
                self.index_buffer.unwrap(),
                0,
                vk::IndexType::UINT32,
            );

            self.device.as_ref().unwrap().cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout.unwrap(),
                0,
                &[self.descriptor_sets[self.current_frame]],
                &[],
            );

            self.device.as_ref().unwrap().cmd_draw_indexed(
                command_buffer,
                self.indices.len() as u32,
                1,
                0,
                0,
                0,
            );

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
                self.in_flight_fences.push(
                    self.device
                        .as_ref()
                        .unwrap()
                        .create_fence(&fence_info, None)
                        .expect("failed to create synchronization objects for a frame!"),
                );
            }
        }
    }

    fn update_uniform_buffer(&mut self, current_image: usize) {
        let time = self.start_time.elapsed().as_secs_f32();

        let extent = self.swap_chain_extent.as_ref().unwrap();
        let aspect = extent.width as f32 / extent.height as f32;
        let mut proj = perspective(Deg(45.0), aspect, 0.1, 10.0);
        proj[1][1] = proj[1][1] * -1.0;
        let ubos = [UniformBufferObject {
            model: Matrix4::from_angle_z(Deg(time * 90.0f32.to_radians())),
            view: Matrix4::look_at_rh(
                Point3 {
                    x: 2.0,
                    y: 2.0,
                    z: 2.0,
                },
                Point3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                Vector3 {
                    x: 0.0,
                    y: 0.0,
                    z: 1.0,
                },
            ),
            proj,
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

            let (image_index, _) = im_result.unwrap();

            self.update_uniform_buffer(self.current_frame);

            self.device.as_ref().unwrap().reset_fences(&fences).unwrap();

            self.device
                .as_ref()
                .unwrap()
                .reset_command_buffer(
                    self.command_buffers[self.current_frame],
                    vk::CommandBufferResetFlags::empty(),
                )
                .unwrap();
            self.record_command_buffer(
                self.command_buffers[self.current_frame],
                image_index.try_into().unwrap(),
            );

            let command_buffers = [self.command_buffers[self.current_frame]];
            let wait_semaphores = [self.image_available_semaphores[self.current_frame]];
            let signal_semaphores = [self.render_finished_semaphores[self.current_frame]];
            let wait_dst_stage_mask = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
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

        let supported_features = unsafe {
            self.instance
                .as_ref()
                .unwrap()
                .get_physical_device_features(device)
        };

        indices.is_complete()
            && extensions_supported
            && swap_chain_adequate
            && supported_features.sampler_anisotropy == vk::TRUE
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
                indices.graphics_family = Some(i);
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

impl Drop for HelloTriangleApplication {
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
                .destroy_render_pass(self.render_pass.unwrap(), None);

            self.uniform_buffers.iter().for_each(|buffer| {
                self.device.as_ref().unwrap().destroy_buffer(*buffer, None);
            });

            self.uniform_buffers_memory.iter().for_each(|mem| {
                self.device.as_ref().unwrap().free_memory(*mem, None);
            });

            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_pool(self.descriptor_pool.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_sampler(self.texture_sampler.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .destroy_image_view(self.texture_image_view.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_image(self.texture_image.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.texture_image_memory.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_descriptor_set_layout(self.descriptor_set_layout.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(self.index_buffer.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.index_buffer_memory.unwrap(), None);

            self.device
                .as_ref()
                .unwrap()
                .destroy_buffer(self.vertex_buffer.unwrap(), None);
            self.device
                .as_ref()
                .unwrap()
                .free_memory(self.vertex_buffer_memory.unwrap(), None);

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
            self.in_flight_fences.iter().for_each(|fence| {
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
    let mut app = HelloTriangleApplication::new();
    app.run();
}
