#!/usr/bin/env python3
"""
Batch processing script for SF3D with multiple images
Supports various input methods and comprehensive export options
"""

import os
import sys
import argparse
import glob
from pathlib import Path

def find_images(input_path):
    """Find images based on input path"""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        # Find all common image formats in directory
        patterns = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff"]
        images = []
        for pattern in patterns:
            images.extend(glob.glob(os.path.join(input_path, pattern)))
        return sorted(images)
    else:
        # Check if it's a glob pattern
        images = glob.glob(input_path)
        return images if images else []

def generate_batch_command(images, output_dir, options):
    """Generate batch processing command"""
    cmd_parts = ["python", "run.py"] + images
    
    # Add options
    if options.get('export_formats'):
        cmd_parts.extend(["--export-formats"] + options['export_formats'])
    
    if options.get('generate_mtl'):
        cmd_parts.append("--generate-mtl")
    
    if options.get('texture_resolution'):
        cmd_parts.extend(["--texture-resolution", str(options['texture_resolution'])])
    
    if options.get('enable_texture_transfer'):
        cmd_parts.append("--enable-texture-transfer")
        
    if options.get('reference_image'):
        cmd_parts.extend(["--reference-image", options['reference_image']])
    
    if options.get('device'):
        cmd_parts.extend(["--device", options['device']])
    
    cmd_parts.extend(["--output-dir", output_dir])
    
    return " ".join(cmd_parts)

def main():
    parser = argparse.ArgumentParser(description="Batch process multiple images with SF3D")
    parser.add_argument("input", nargs="+", help="Input images, folders, or patterns")
    parser.add_argument("--output-dir", default="batch_output", help="Output directory")
    parser.add_argument("--export-formats", nargs="+", default=["glb"], 
                       choices=["glb", "obj"], help="Export formats")
    parser.add_argument("--generate-mtl", action="store_true", help="Generate MTL files")
    parser.add_argument("--texture-resolution", type=int, default=2048, help="Texture resolution")
    parser.add_argument("--enable-texture-transfer", action="store_true", help="Enable AI texture transfer")
    parser.add_argument("--reference-image", help="Reference image for texture transfer")
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dry-run", action="store_true", help="Show commands without executing")
    parser.add_argument("--parallel", action="store_true", help="Process images in parallel")
    
    args = parser.parse_args()
    
    # Collect all images
    all_images = []
    for input_item in args.input:
        images = find_images(input_item)
        all_images.extend(images)
    
    # Remove duplicates
    all_images = list(set(all_images))
    all_images.sort()
    
    if not all_images:
        print("❌ No images found!")
        return
    
    print(f"🖼️  Found {len(all_images)} images to process:")
    for i, img in enumerate(all_images, 1):
        print(f"   {i}. {img}")
    
    # Prepare options
    options = {
        'export_formats': args.export_formats,
        'generate_mtl': args.generate_mtl,
        'texture_resolution': args.texture_resolution,
        'enable_texture_transfer': args.enable_texture_transfer,
        'reference_image': args.reference_image,
        'device': args.device
    }
    
    # Generate command
    command = generate_batch_command(all_images, args.output_dir, options)
    
    print(f"\n🔧 Command to execute:")
    print(f"   {command}")
    
    if args.dry_run:
        print("\n🔍 Dry run mode - not executing")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n🚀 Processing {len(all_images)} images...")
    print(f"📁 Output directory: {args.output_dir}")
    
    # Execute command
    import subprocess
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ Batch processing completed successfully!")
        
        # Show output structure
        print(f"\n📂 Output structure:")
        for i in range(len(all_images)):
            folder_path = os.path.join(args.output_dir, str(i))
            if os.path.exists(folder_path):
                files = os.listdir(folder_path)
                print(f"   {i}/: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during processing:")
        print(f"   {e.stderr}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
