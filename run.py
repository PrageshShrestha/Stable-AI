import argparse
import os
from contextlib import nullcontext

import rembg
import torch
from PIL import Image
from tqdm import tqdm

from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground
from ai_texture_transfer import apply_ai_texture_transfer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image", type=str, nargs="+", help="Path to input image(s) or folder."
    )
    parser.add_argument(
        "--device",
        default=get_device(),
        type=str,
        help=f"Device to use. If no CUDA/MPS-compatible device is found, the baking will fail. Default: '{get_device()}'",
    )
    parser.add_argument(
        "--pretrained-model",
        default="stabilityai/stable-fast-3d",
        type=str,
        help="Path to the pretrained model. Could be either a huggingface model id is or a local path. Default: 'stabilityai/stable-fast-3d'",
    )
    parser.add_argument(
        "--foreground-ratio",
        default=0.85,
        type=float,
        help="Ratio of the foreground size to the image size. Only used when --no-remove-bg is not specified. Default: 0.85",
    )
    parser.add_argument(
        "--output-dir",
        default="output/",
        type=str,
        help="Output directory to save the results. Default: 'output/'",
    )
    parser.add_argument(
        "--texture-resolution",
        default=2048,
        type=int,
        help="Texture atlas resolution. Default: 2048",
    )
    parser.add_argument(
        "--remesh_option",
        choices=["none", "triangle", "quad"],
        default="triangle",
        help="Remeshing option",
    )
    parser.add_argument(
        "--target_vertex_count",
        type=int,
        help="Target vertex count. -1 does not perform a reduction.",
        default=50000,
    )
    parser.add_argument(
        "--batch_size", default=1, type=int, help="Batch size for inference"
    )
    parser.add_argument(
        "--reference-image", 
        type=str, 
        default=None, 
        help="Path to reference image for texture transfer"
    )
    parser.add_argument(
        "--enable-texture-transfer",
        action="store_true",
        help="Enable AI texture transfer using reference image"
    )
    default_prompt="""
Execute a high-fidelity 3D texture projection by mapping the exact color values and material patterns from the provided reference image onto the geometry of the grey mesh. Perform a precise 1:1 spatial alignment where every visual element from the photo is projected onto its corresponding part of the 3D model with absolute accuracy. Generate a complete set of PBR maps including Albedo, Roughness, and Metallic textures based on the surface qualities visible in the reference. Ensure the UV mapping is seamless with no stretching, ghosting, or mirroring artifacts on the sides or rear of the mesh. The final texture must be a clean, de-lighted Albedo map that captures the fine surface details, grain, and color variations of the original object without baking in the environmental shadows or highlights from the reference photo. Maintain high-resolution edge crispness where different colored parts of the object meet to ensure a professional, production-quality finish.

"""
    parser.add_argument(
        "--texture-prompt",
        type=str,
        default=default_prompt,
        help="Text prompt for texture generation"
    )
    args = parser.parse_args()

    # Ensure args.device contains cuda
    devices = ["cuda", "mps", "cpu"]
    if not any(args.device in device for device in devices):
        raise ValueError("Invalid device. Use cuda, mps or cpu")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    device = args.device
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        device = "cpu"

    print("Device used: ", device)

    model = SF3D.from_pretrained(
        args.pretrained_model,
        config_name="config.yaml",
        weight_name="model.safetensors",
    )
    model.to(device)
    model.eval()

    rembg_session = rembg.new_session()
    images = []
    idx = 0
    for image_path in args.image:

        def handle_image(image_path, idx):
            image = remove_background(
                Image.open(image_path).convert("RGBA"), rembg_session
            )
            image = resize_foreground(image, args.foreground_ratio)
            os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
            image.save(os.path.join(output_dir, str(idx), "input.png"))
            images.append(image)

        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, f)
                for f in os.listdir(image_path)
                if f.endswith((".png", ".jpg", ".jpeg"))
            ]
            for image_path in image_paths:
                handle_image(image_path, idx)
                idx += 1
        else:
            handle_image(image_path, idx)
            idx += 1

    for i in tqdm(range(0, len(images), args.batch_size)):
        image = images[i : i + args.batch_size]
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            with torch.autocast(
                device_type=device, dtype=torch.bfloat16
            ) if "cuda" in device else nullcontext():
                mesh, glob_dict = model.run_image(
                    image,
                    bake_resolution=args.texture_resolution,
                    remesh=args.remesh_option,
                    vertex_count=args.target_vertex_count,
                )
        if torch.cuda.is_available():
            print("Peak Memory:", torch.cuda.max_memory_allocated() / 1024 / 1024, "MB")
        elif torch.backends.mps.is_available():
            print(
                "Peak Memory:", torch.mps.driver_allocated_memory() / 1024 / 1024, "MB"
            )

        if len(image) == 1:
            out_mesh_path = os.path.join(output_dir, str(i), "mesh.glb")
            mesh.export(out_mesh_path, include_normals=True)
            
            # Apply AI texture transfer if enabled
            if args.enable_texture_transfer:
                print("Applying AI texture transfer...")
                try:
                    # Pass SF3D's background-removed image for exact color matching
                    sf3d_image_path = os.path.join(output_dir, str(i), "input.png")
                    textured_mesh_path = apply_ai_texture_transfer(
                        mesh_path=out_mesh_path,
                        reference_image_path=args.reference_image,
                        sf3d_image_path=sf3d_image_path,  # Use SF3D image as primary texture source
                        output_path=os.path.join(output_dir, str(i), "mesh_ai_textured.glb"),
                        texture_size=args.texture_resolution,
                        prompt=args.texture_prompt,
                        device=device
                    )
                    print(f"AI textured mesh saved to: {textured_mesh_path}")
                except Exception as e:
                    print(f"AI texture transfer failed: {e}")
                    print("Continuing with original mesh...")
        else:
            for j in range(len(mesh)):
                out_mesh_path = os.path.join(output_dir, str(i + j), "mesh.glb")
                mesh[j].export(out_mesh_path, include_normals=True)
                
                # Apply AI texture transfer if enabled
                if args.enable_texture_transfer:
                    print(f"Applying AI texture transfer to mesh {j}...")
                    try:
                        # Pass SF3D's background-removed image for exact color matching
                        sf3d_image_path = os.path.join(output_dir, str(i + j), "input.png")
                        textured_mesh_path = apply_ai_texture_transfer(
                            mesh_path=out_mesh_path,
                            reference_image_path=args.reference_image,
                            sf3d_image_path=sf3d_image_path,  # Use SF3D image as primary texture source
                            output_path=os.path.join(output_dir, str(i + j), "mesh_ai_textured.glb"),
                            texture_size=args.texture_resolution,
                            prompt=args.texture_prompt,
                            device=device
                        )
                        print(f"AI textured mesh saved to: {textured_mesh_path}")
                    except Exception as e:
                        print(f"AI texture transfer failed: {e}")
                        print("Continuing with original mesh...")
