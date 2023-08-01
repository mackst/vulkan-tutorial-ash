use std::{env, ffi::OsStr, fs, path::Path, process::Command};

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let manifest_dir_path = Path::new(&manifest_dir);
    let src_path = manifest_dir_path.join("src");
    let shader_files = fs::read_dir(src_path)
        .unwrap()
        .map(Result::unwrap)
        .filter(|res| res.file_type().unwrap().is_file())
        .filter(|res| {
            res.path().extension() == Some(OsStr::new("vert"))
                || res.path().extension() == Some(OsStr::new("frag"))
                || res.path().extension() == Some(OsStr::new("comp"))
        });

    let profile = env::var("PROFILE").unwrap();
    let target_dir = manifest_dir_path.join("target");
    if !target_dir.exists() {
        panic!("路径不存在：{}", target_dir.to_str().unwrap());
    }
    let exe_dir = Path::new(&target_dir).join(profile);
    if !exe_dir.exists() {
        panic!("路径不存在：{}", target_dir.to_str().unwrap());
    }
    shader_files.for_each(|de| {
        let path = de.path();
        let shader_file_name = path.file_name().unwrap();
        let spv_file_name = shader_file_name.to_str().unwrap().replace(".", "_");
        let compile_shader_file = exe_dir.join(format!("{}.spv", spv_file_name));
        // println!("{:?}", path);
        // println!("spv: {:?}", compile_shader_file);
        let result = Command::new("glslc")
            .current_dir(manifest_dir_path)
            .arg(&path)
            .arg("-o")
            .arg(&compile_shader_file)
            .output();

        match result {
            Ok(output) => {
                if output.status.success() {
                    println!("{}", String::from_utf8(output.stdout).unwrap());
                } else {
                    println!("{}", String::from_utf8(output.stderr).unwrap());
                    panic!("错误发生。");
                }
            }
            Err(error) => {
                panic!("无法编译shader：{}", error);
            }
        }
    });

    // 贴图
    let resources_dir = manifest_dir_path.join("resources");
    let texture_jpeg = "texture.jpg";
    let texture_path = resources_dir.join(texture_jpeg);
    if texture_path.exists() {
        let texture_dst_path = exe_dir.join(texture_jpeg);
        fs::copy(&texture_path, texture_dst_path).expect(&format!("无法拷贝: {:?}", &texture_path));
    }

    // obj模型和贴图
    let obj_name = "viking_room.obj";
    let obj_texture_name = "viking_room.png";
    let obj_path = resources_dir.join(obj_name);
    let obj_texture_path = resources_dir.join(obj_texture_name);
    if obj_path.exists() {
        let obj_dst_path = exe_dir.join(obj_name);
        fs::copy(&obj_path, obj_dst_path).expect(&format!("无法拷贝: {:?}", &obj_path));
    }
    if obj_texture_path.exists() {
        let obj_texture_dst_path = exe_dir.join(obj_texture_name);
        fs::copy(&obj_texture_path, obj_texture_dst_path)
            .expect(&format!("无法拷贝: {:?}", &obj_texture_path));
    }
}
