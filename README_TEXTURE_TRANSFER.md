# AI Texture Transfer for SF3D

This project adds AI-powered texture transfer capabilities to Stable Fast 3D (SF3D), allowing you to transform grey 3D meshes into realistically textured models using reference images.

## Features

- **AI Texture Generation**: Uses Stable Diffusion with ControlNet to generate realistic textures
- **Reference Image Guidance**: Takes a reference image to guide color and texture generation
- **Depth-Aware Texturing**: Generates depth maps from 3D meshes for accurate texture projection
- **UV Mapping Support**: Automatically generates UV coordinates for proper texture mapping
- **Command Line Interface**: Integrate with existing SF3D pipeline
- **Gradio Web Interface**: User-friendly web interface for texture transfer

## Installation

1. Activate the conda environment:
```bash
conda activate copy_stable
```

2. Install additional dependencies:
```bash
pip install diffusers controlnet-aux opencv-python
```

## Usage

### Command Line Interface

1. **Basic SF3D + Texture Transfer**:
```bash
python run.py input_image.jpg --enable-texture-transfer --reference-image reference.jpg --texture-prompt "realistic wood texture"
```

2. **Full Command Options**:
```bash
python run.py input_image.jpg \
    --enable-texture-transfer \
    --reference-image path/to/reference.jpg \
    --texture-prompt "detailed realistic texture" \
    --texture-resolution 1024 \
    --output-dir output/
```

### Gradio Web Interface

1. Start the Gradio app:
```bash
python gradio_app.py
```

2. In the web interface:
   - Upload your input image
   - Remove background if needed
   - Enable "AI Texture Transfer"
   - Upload a reference image for color guidance
   - Adjust texture description if desired
   - Click "Run" to generate the textured 3D model

## How It Works

1. **SF3D Mesh Generation**: First generates a grey 3D mesh from your input image
2. **Depth Map Generation**: Creates a depth map from the 3D mesh geometry
3. **Reference Image Analysis**: Extracts color information from your reference image
4. **AI Texture Generation**: Uses Stable Diffusion with ControlNet to generate textures
5. **Texture Application**: Applies the generated texture to the 3D mesh with proper UV mapping

## New Command Line Arguments

- `--enable-texture-transfer`: Enable AI texture transfer
- `--reference-image`: Path to reference image for color guidance
- `--texture-prompt`: Text description for texture generation

## Examples

### Wood Texture Transfer
```bash
python run.py chair.jpg --enable-texture-transfer --reference-image wood_grain.jpg --texture-prompt "realistic wood grain texture"
```

### Fabric Texture Transfer
```bash
python run.py sofa.jpg --enable-texture-transfer --reference-image fabric_pattern.jpg --texture-prompt "soft fabric texture with detailed patterns"
```

## Output Files

- `mesh.glb`: Original grey mesh from SF3D
- `mesh_textured.glb`: Final textured mesh (when texture transfer is enabled)

## Requirements

- PyTorch
- Diffusers
- ControlNet-Aux
- OpenCV
- Trimesh
- PIL (Pillow)
- SF3D dependencies

## Tips

1. **Reference Images**: Use high-quality images with clear color patterns for best results
2. **Texture Prompts**: Be specific about material properties (e.g., "glossy metal", "rough concrete")
3. **Resolution**: Higher texture resolution (2048) provides more detail but takes longer
4. **Lighting**: Reference images with consistent lighting work better

## Troubleshooting

- **CUDA Out of Memory**: Try reducing texture resolution or using CPU
- **Poor Texture Quality**: Improve reference image quality or adjust texture prompt
- **UV Mapping Issues**: The system includes fallback vertex coloring if UV mapping fails

## Model Downloads

The first run will automatically download:
- Stable Diffusion Inpainting model (~2GB)
- MiDaS depth estimation model (~100MB)
- ControlNet models (~500MB)

These will be cached locally for future use.
