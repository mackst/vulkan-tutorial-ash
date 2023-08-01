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

use ash::extensions::ext::DebugUtils;
use ash::{
    vk::{self, DebugUtilsMessengerEXT},
    Entry, Instance,
};
use ash_window::enumerate_required_extensions;

use raw_window_handle::HasRawDisplayHandle;

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] = &["VK_LAYER_KHRONOS_validation"];

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

#[allow(unused)]
struct HelloTriangleApplication {
    window: Option<Window>,
    vk_entry: Option<Entry>,
    instance: Option<Instance>,
    debug_utils: Option<DebugUtils>,
    debug_messenger: Option<DebugUtilsMessengerEXT>,
}

impl HelloTriangleApplication {
    pub fn new() -> Self {
        Self {
            window: None,
            vk_entry: None,
            instance: None,
            debug_utils: None,
            debug_messenger: None,
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
