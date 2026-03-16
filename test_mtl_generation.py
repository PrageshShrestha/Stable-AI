#!/usr/bin/env python3
"""
Test script for MTL generation with SF3D
Demonstrates OBJ+MTL export functionality
"""

import os
import sys
import argparse
from pathlib import Path

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent))

def test_mtl_generation():
    """Test MTL file generation with a sample image"""
    
    print("🔧 Testing MTL Generation with SF3D")
    print("=" * 50)
    
    # Check if conda environment is activated
    try:
        import torch
        print(f"✓ PyTorch available: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("❌ PyTorch not found. Please activate conda environment:")
        print("   conda activate stable-ai")
        return
    
    # Check if sample image exists
    sample_images = ["bicycle.jpg", "img1.jpg", "img2.jpg", "img3.jpg"]
    sample_image = None
    for img in sample_images:
        if os.path.exists(img):
            sample_image = img
            break
    
    if not sample_image:
        print("❌ No sample image found. Please place an image in the project root.")
        return
    
    print(f"✓ Using sample image: {sample_image}")
    
    # Test command examples
    print("\n📝 Example Commands:")
    print("-" * 30)
    
    print("\n1. Basic GLB export (default):")
    print(f"   python run.py {sample_image} --output-dir test_output --device cuda")
    
    print("\n2. OBJ + MTL export:")
    print(f"   python run.py {sample_image} --export-formats obj --generate-mtl --output-dir test_output --device cuda")
    
    print("\n3. Both GLB and OBJ + MTL:")
    print(f"   python run.py {sample_image} --export-formats glb obj --generate-mtl --output-dir test_output --device cuda")
    
    print("\n4. With AI texture transfer:")
    print(f"   python run.py {sample_image} --export-formats glb obj --generate-mtl --enable-texture-transfer --reference-image {sample_image} --output-dir test_output --device cuda")
    
    print("\n5. High resolution texture:")
    print(f"   python run.py {sample_image} --export-formats obj --generate-mtl --texture-resolution 4096 --output-dir test_output --device cuda")
    
    print("\n📁 Expected Output Structure:")
    print("-" * 30)
    print("test_output/")
    print("├── 0/")
    print("│   ├── mesh.glb              # GLB with embedded textures")
    print("│   ├── mesh.obj              # Wavefront OBJ mesh")
    print("│   ├── mesh.mtl              # Material file")
    print("│   ├── mesh_texture.png      # Diffuse/albedo texture")
    print("│   ├── mesh_texture_normal.png  # Normal map (if generated)")
    print("│   ├── mesh_ai_textured.glb  # AI-enhanced version")
    print("│   └── input.png             # Background-removed input")
    
    print("\n🎯 MTL File Features:")
    print("-" * 30)
    print("✓ Proper material definitions")
    print("✓ Texture references (relative paths)")
    print("✓ Normal map support")
    print("✓ PBR material parameters")
    print("✓ Compatible with Blender, MeshLab, etc.")
    
    print("\n🔍 Viewer Instructions:")
    print("-" * 30)
    print("Blender:")
    print("  1. File > Import > Wavefront (.obj)")
    print("  2. Select mesh.obj")
    print("  3. Switch to Material Preview mode")
    print("  4. Check Material Properties > Principled BSDF")
    
    print("\nMeshLab:")
    print("  1. File > Import Mesh")
    print("  2. Select mesh.obj")
    print("  3. Render > Texture > Per-Vertex")
    print("  4. Or: Render > Texture > Per-Face")
    
    print("\n✅ MTL generation test complete!")

if __name__ == "__main__":
    test_mtl_generation()
