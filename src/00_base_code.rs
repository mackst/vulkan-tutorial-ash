use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    platform::run_return::EventLoopExtRunReturn,
    window::{Window, WindowBuilder},
};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

#[allow(unused)]
struct HelloTriangleApplication {
    window: Option<Window>,
}

impl HelloTriangleApplication {
    pub fn new() -> Self {
        Self { window: None }
    }

    pub fn run(&mut self) {
        let event_loop = EventLoop::new();
        self.init_window(&event_loop);
        self.init_vulkan();
        self.main_loop(event_loop);
    }

    fn init_vulkan(&mut self) {}

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
}

fn main() {
    let mut app = HelloTriangleApplication::new();
    app.run();
}
