"""
Simplified AI Texture Transfer Module for SF3D
Uses OpenCV and PIL for basic texture transfer from reference images
"""

import os
import numpy as np
from PIL import Image
import trimesh
import cv2
from typing import Optional, Tuple
import tempfile


class SimpleTextureTransfer:
    def __init__(self):
        """Initialize simple texture transfer"""
        pass
    
    def generate_depth_map(self, mesh: trimesh.Trimesh, size: int = 512) -> np.ndarray:
        """Generate simple depth map from 3D mesh using vertex projection"""
        # Get vertices and faces
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Normalize vertices to [0, 1] range
        vertices_norm = (vertices - vertices.min(axis=0)) / (vertices.max(axis=0) - vertices.min(axis=0) + 1e-8)
        
        # Create depth map using Z coordinates
        depth_map = np.zeros((size, size), dtype=np.float32)
        
        # Project vertices to 2D and use Z as depth
        for vertex in vertices_norm:
            x = int(vertex[0] * (size - 1))
            y = int(vertex[1] * (size - 1))
            z = vertex[2]
            
            if 0 <= x < size and 0 <= y < size:
                depth_map[y, x] = max(depth_map[y, x], z)
        
        # Normalize to 0-255
        depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        
        return depth_map
    
    def extract_dominant_colors(self, reference_image: Image.Image, num_colors: int = 5) -> np.ndarray:
        """Extract dominant colors from reference image using K-means"""
        # Convert to numpy array
        img_array = np.array(reference_image)
        
        # Reshape to list of pixels
        pixels = img_array.reshape(-1, 3)
        
        # Remove black/white pixels (background)
        pixels = pixels[(pixels > 30).all(axis=1) & (pixels < 225).all(axis=1)]
        
        if len(pixels) == 0:
            # Fallback to some default colors
            return np.array([[128, 128, 128], [200, 200, 200], [100, 100, 100]])
        
        # Use K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(pixels.astype(np.float32), num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers.astype(np.uint8)
    
    def create_texture_from_colors(self, depth_map: np.ndarray, colors: np.ndarray, texture_size: int = 1024) -> Image.Image:
        """Create texture based on depth map and extracted colors"""
        # Resize depth map to texture size
        depth_resized = cv2.resize(depth_map, (texture_size, texture_size))
        
        # Create texture by mapping depth to colors
        texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)
        
        # Normalize depth to [0, 1] for color mapping
        depth_norm = depth_resized.astype(np.float32) / 255.0
        
        # Create gradient-like texture using colors
        for i, color in enumerate(colors):
            mask = (depth_norm >= i/len(colors)) & (depth_norm < (i+1)/len(colors))
            texture[mask] = color
        
        # Add some noise for texture variation
        noise = np.random.normal(0, 10, texture.shape).astype(np.int16)
        texture = np.clip(texture.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply Gaussian blur for smoother texture
        texture = cv2.GaussianBlur(texture, (5, 5), 0)
        
        return Image.fromarray(texture)
    
    def apply_texture_to_mesh(self, mesh: trimesh.Trimesh, texture_image: Image.Image) -> trimesh.Trimesh:
        """Apply texture to mesh"""
        try:
            # Try to generate UV coordinates
            mesh = mesh.unwrap_uv()
            
            # Create material with texture
            material = trimesh.visual.texture.SimpleMaterial(image=texture_image)
            
            # Apply texture to mesh
            textured_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                visual=trimesh.visual.texture.TextureVisuals(
                    uv=mesh.visual.uv,
                    image=texture_image,
                    material=material
                )
            )
            
            return textured_mesh
            
        except Exception as e:
            print(f"UV mapping failed: {e}")
            # Fallback to vertex coloring
            return self._apply_vertex_colors(mesh, texture_image)
    
    def _apply_vertex_colors(self, mesh: trimesh.Trimesh, texture_image: Image.Image) -> trimesh.Trimesh:
        """Fallback: Apply colors as vertex colors"""
        # Sample colors from texture
        texture_array = np.array(texture_image)
        
        # Create vertex colors based on position
        vertex_colors = []
        for vertex in mesh.vertices:
            # Map vertex position to texture coordinates
            x = int((vertex[0] - mesh.vertices[:, 0].min()) / 
                   (mesh.vertices[:, 0].max() - mesh.vertices[:, 0].min() + 1e-8) * (texture_array.shape[1] - 1))
            y = int((vertex[1] - mesh.vertices[:, 1].min()) / 
                   (mesh.vertices[:, 1].max() - mesh.vertices[:, 1].min() + 1e-8) * (texture_array.shape[0] - 1))
            
            x = np.clip(x, 0, texture_array.shape[1] - 1)
            y = np.clip(y, 0, texture_array.shape[0] - 1)
            
            vertex_colors.append(texture_array[y, x, :3])
        
        vertex_colors = np.array(vertex_colors)
        
        # Create mesh with vertex colors
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=vertex_colors
        )
        
        return colored_mesh
    
    def generate_texture(
        self, 
        mesh: trimesh.Trimesh, 
        reference_image: Optional[Image.Image] = None,
        texture_size: int = 1024,
        prompt: str = "detailed realistic texture"
    ) -> trimesh.Trimesh:
        """Generate texture for mesh using reference image guidance"""
        
        # Generate depth map from mesh
        print("Generating depth map from mesh...")
        depth_map = self.generate_depth_map(mesh, size=512)
        
        # Extract colors from reference image
        if reference_image:
            print("Extracting colors from reference image...")
            colors = self.extract_dominant_colors(reference_image, num_colors=5)
        else:
            # Use default colors based on prompt
            if "metallic" in prompt.lower():
                colors = np.array([[180, 180, 190], [140, 140, 150], [100, 100, 110], [200, 200, 210], [160, 160, 170]])
            elif "wood" in prompt.lower():
                colors = np.array([[139, 90, 43], [160, 110, 60], [120, 80, 40], [180, 130, 80], [100, 70, 30]])
            elif "fabric" in prompt.lower():
                colors = np.array([[150, 150, 160], [120, 120, 130], [180, 180, 190], [100, 100, 110], [200, 200, 210]])
            else:
                colors = np.array([[128, 128, 128], [150, 150, 150], [100, 100, 100], [180, 180, 180], [80, 80, 80]])
        
        # Create texture from colors and depth
        print("Creating texture from extracted colors...")
        texture_image = self.create_texture_from_colors(depth_map, colors, texture_size)
        
        # Apply texture to mesh
        textured_mesh = self.apply_texture_to_mesh(mesh, texture_image)
        
        return textured_mesh


def apply_texture_transfer(
    mesh_path: str,
    reference_image_path: Optional[str] = None,
    output_path: str = None,
    texture_size: int = 1024,
    prompt: str = "detailed realistic texture",
    device: str = "cuda"
) -> str:
    """
    Apply AI texture transfer to a mesh using simplified approach
    
    Args:
        mesh_path: Path to input mesh file
        reference_image_path: Path to reference image for color guidance
        output_path: Path to save textured mesh
        texture_size: Size of generated texture
        prompt: Text prompt for texture generation
        device: Device to use for processing (ignored in simplified version)
    
    Returns:
        Path to the textured mesh file
    """
    
    # Load mesh
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    
    # Handle scene objects
    if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
        # Extract the first mesh from the scene
        mesh_name = list(mesh.geometry.keys())[0]
        mesh = mesh.geometry[mesh_name]
    
    # Ensure we have a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        print("Warning: Loaded object is not a Trimesh, attempting to convert...")
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            mesh_name = list(mesh.geometry.keys())[0]
            mesh = mesh.geometry[mesh_name]
    
    # Load reference image if provided
    reference_image = None
    if reference_image_path and os.path.exists(reference_image_path):
        reference_image = Image.open(reference_image_path)
        print(f"Using reference image: {reference_image_path}")
    
    # Initialize texture transfer
    texture_transfer = SimpleTextureTransfer()
    
    # Generate texture
    textured_mesh = texture_transfer.generate_texture(
        mesh=mesh,
        reference_image=reference_image,
        texture_size=texture_size,
        prompt=prompt
    )
    
    # Save result
    if output_path is None:
        base_name = os.path.splitext(mesh_path)[0]
        output_path = f"{base_name}_textured.glb"
    
    print(f"Saving textured mesh to {output_path}...")
    textured_mesh.export(output_path, include_normals=True)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply simplified AI texture transfer to 3D meshes")
    parser.add_argument("mesh_path", help="Path to input mesh file")
    parser.add_argument("--reference", help="Path to reference image", default=None)
    parser.add_argument("--output", help="Output path", default=None)
    parser.add_argument("--texture-size", type=int, default=1024, help="Texture size")
    parser.add_argument("--prompt", default="detailed realistic texture", help="Texture prompt")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    result_path = apply_texture_transfer(
        mesh_path=args.mesh_path,
        reference_image_path=args.reference,
        output_path=args.output,
        texture_size=args.texture_size,
        prompt=args.prompt,
        device=args.device
    )
    
    print(f"Texture transfer completed! Result saved to: {result_path}")
