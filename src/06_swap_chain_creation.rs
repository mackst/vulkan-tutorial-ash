use std::{
    borrow::Cow,
    ffi::{c_char, CStr, CString},
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
    vk::SurfaceFormatKHR,
};
use ash::{
    vk::{self, DebugUtilsMessengerEXT},
    Device, Entry, Instance,
};
use ash_window::enumerate_required_extensions;

use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

const DEVICE_EXTENSIONS: [*const i8; 1] = [Swapchain::name().as_ptr()];

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

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

    swap_chain: Option<vk::SwapchainKHR>,
    swapchain_loader: Option<Swapchain>,
    swap_chain_images: Vec<vk::Image>,
    swap_chain_image_format: Option<vk::Format>,
    swap_chain_extent: Option<vk::Extent2D>,
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
    }

    fn init_window(&mut self, event_loop: &EventLoop<()>) {
        let window = WindowBuilder::new()
            .with_title("Vulkan")
            .with_inner_size(LogicalSize::new(WIDTH, HEIGHT))
            .with_resizable(false)
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

                _ => (),
            }
        });
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
        let mut unique_queue_families = vec![
            indices.graphics_family.unwrap(),
            indices.present_family.unwrap(),
        ];
        unique_queue_families.dedup();
        let mut queue_create_infos = Vec::new();
        for queue_family in unique_queue_families {
            let queue_create_info = vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(queue_family)
                .queue_priorities(&queue_priority);
            queue_create_infos.push(queue_create_info.build());
        }

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
}

impl Drop for HelloTriangleApplication {
    fn drop(&mut self) {
        unsafe {
            self.swapchain_loader
                .as_ref()
                .unwrap()
                .destroy_swapchain(self.swap_chain.unwrap(), None);
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

        self.vk_entry = None;
    }
}

fn main() {
    let mut app = HelloTriangleApplication::new();
    app.run();
}
