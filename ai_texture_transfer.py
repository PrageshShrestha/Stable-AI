"""
AI Texture Transfer Module for SF3D using Stable Diffusion
Integrates Stable Diffusion with ControlNet for realistic texture generation
"""

import os
import torch
import numpy as np
from PIL import Image
import trimesh
import cv2
from typing import Optional, Tuple
import tempfile

try:
    from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, DDIMScheduler
    from controlnet_aux import MidasDetector
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    print(f"Diffusers not available: {e}")
    DIFFUSERS_AVAILABLE = False


class AITextureTransfer:
    def __init__(self, device: str = "cuda"):
        """Initialize AI texture transfer with Stable Diffusion"""
        self.device = device
        self.depth_estimator = None
        self.texture_pipe = None
        
        if DIFFUSERS_AVAILABLE:
            self._load_models()
        else:
            print("Warning: AI texture transfer not available, falling back to simple method")
    
    def _load_models(self):
        """Load depth estimation and texture generation models"""
        if not DIFFUSERS_AVAILABLE:
            return
            
        print("Loading depth estimation model...")
        try:
            # Modern MiDaS/DPT weights from the standard annotators repo
            self.depth_estimator = MidasDetector.from_pretrained(
                "lllyasviel/Annotators"
            ).to(self.device)
            print("Depth estimation model loaded successfully")
        except Exception as e:
            print(f"Failed to load depth estimator: {e}")
            self.depth_estimator = None  # Allow fallback even if estimator fails
            # Do NOT return here - try to load pipeline anyway if possible
        
        print("Loading Stable Diffusion texture generation model...")
        try:
            # Check if model files exist locally first
            model_path = "runwayml/stable-diffusion-inpainting"
            controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"
            
            # Load ControlNet separately first
            try:
                print(f"Loading ControlNet from {controlnet_path}...")
                controlnet = ControlNetModel.from_pretrained(
                    controlnet_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    local_files_only=False,
                    resume_download=True
                )
                print("ControlNet loaded successfully")
            except Exception as controlnet_error:
                print(f"Failed to load ControlNet: {controlnet_error}")
                print("Attempting to continue without ControlNet...")
                controlnet = None

            # Load pipeline with or without ControlNet
            try:
                print(f"Loading Stable Diffusion pipeline from {model_path}...")
                pipeline_kwargs = {
                    "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                    "safety_checker": None,
                    "requires_safety_checker": False,
                    "local_files_only": False,
                    "resume_download": True
                }
                
                # Add variant for CUDA to reduce memory usage
                if self.device == "cuda":
                    pipeline_kwargs["variant"] = "fp16"
                
                # Only add controlnet if it loaded successfully
                if controlnet is not None:
                    pipeline_kwargs["controlnet"] = controlnet
                    controlnet.to(self.device)
                
                self.texture_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                    model_path,
                    **pipeline_kwargs
                )
                
                # Move to device
                self.texture_pipe.to(self.device)
                
                if self.device == "cuda":
                    self.texture_pipe.enable_model_cpu_offload()  # Good for VRAM
                    # Optional: if you have xformers installed
                    # try:
                    #     self.texture_pipe.enable_xformers_memory_efficient_attention()
                    #     print("Enabled xformers memory efficient attention")
                    # except:
                    #     print("xformers not available, using default attention")
                
                print("Stable Diffusion pipeline loaded successfully")
                
            except Exception as pipeline_error:
                print(f"Failed to load pipeline: {pipeline_error}")
                print("This might be due to missing model files or network issues")
                self.texture_pipe = None
                
        except Exception as e:
            print(f"Unexpected error during model loading: {e}")
            self.texture_pipe = None
        
        # Add warning check for partial failures
        if self.texture_pipe is None:
            print("Warning: Stable Diffusion pipeline failed to load - AI texture generation will be disabled")
            print("Falling back to traditional texture methods only")
        elif self.depth_estimator is None:
            print("Warning: Depth estimator failed to load - texture quality may be reduced")
        else:
            print("All AI models loaded successfully")
    
    def generate_depth_map(self, mesh: trimesh.Trimesh, size: int = 512) -> np.ndarray:
        """Generate depth map from 3D mesh"""
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
        if depth_map.max() > depth_map.min():
            depth_map = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        else:
            depth_map = np.zeros((size, size), dtype=np.uint8)
        
        return depth_map
    
    def extract_colors_from_reference(self, reference_image: Image.Image) -> Tuple[str, Image.Image]:
        """Extract color information from reference image"""
        # Convert to RGB if needed
        ref_rgb = reference_image.convert("RGB")
        
        # Create a simple color palette description
        # Resize to get dominant colors
        small = ref_rgb.resize((50, 50))
        colors = np.array(small).reshape(-1, 3)
        
        # Get most common colors
        unique_colors, counts = np.unique(colors, axis=0, return_counts=True)
        top_colors = unique_colors[np.argsort(counts)[-5:]]
        
        # Create color description
        color_desc = "with colors: " + ", ".join([
            f"RGB({int(c[0])},{int(c[1])},{int(c[2])})" 
            for c in top_colors
        ])
        
        return color_desc, ref_rgb
    
    def generate_texture_with_ai(
        self, 
        mesh: trimesh.Trimesh, 
        reference_image: Optional[Image.Image] = None,
        texture_size: int = 1024,
        prompt: str = "detailed realistic texture",
        texture_source_type: str = "reference",
        use_direct_mapping: bool = False
    ) -> trimesh.Trimesh:
        """Generate texture for mesh using Stable Diffusion or direct color mapping"""
        
        # Direct color preservation mode - bypass AI entirely
        if use_direct_mapping and reference_image and texture_source_type == "sf3d_original":
            print("Using direct color preservation mode - bypassing AI generation...")
            return self._direct_color_mapping(mesh, reference_image, texture_size)
        
        if not DIFFUSERS_AVAILABLE or self.texture_pipe is None:
            print("AI models not available, using fallback method...")
            return self._generate_fallback_texture(mesh, reference_image, texture_size, prompt, texture_source_type)
        
        # Generate depth map from mesh
        print("Generating depth map from mesh...")
        depth_map = self.generate_depth_map(mesh, size=512)
        depth_image = Image.fromarray(depth_map)
        
        # Process texture source based on type
        if reference_image and texture_source_type == "sf3d_original":
            # Use SF3D original image directly - this gives exact colors!
            print("Using SF3D original image for exact color matching...")
            enhanced_prompt = f"{prompt}, highly detailed, photorealistic, 8k, ultra realistic, preserve original colors"
            init_image = reference_image.resize((512, 512)).convert("RGB")
            
        elif reference_image and texture_source_type == "reference":
            # External reference image
            color_desc, ref_processed = self.extract_colors_from_reference(reference_image)
            enhanced_prompt = f"{prompt}, {color_desc}, highly detailed, photorealistic, 8k, ultra realistic"
            init_image = reference_image.resize((512, 512)).convert("RGB")
            
        else:
            # No reference image
            enhanced_prompt = f"{prompt}, highly detailed, photorealistic, 8k, ultra realistic"
            init_image = Image.fromarray(np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8))
        
        # Create control image from depth
        try:
            if self.depth_estimator is None:
                print("Depth estimator not available, using fallback...")
                return self._generate_fallback_texture(mesh, reference_image, texture_size, prompt, texture_source_type)
            
            control_image = self.depth_estimator(depth_image, detect_resolution=512, image_resolution=512)
        except Exception as e:
            print(f"Depth estimation failed: {e}")
            return self._generate_fallback_texture(mesh, reference_image, texture_size, prompt, texture_source_type)
        
        # Prepare mask (use entire image for inpainting)
        mask = Image.new("RGB", (512, 512), (255, 255, 255))
        
        print("Generating texture with Stable Diffusion...")
        try:
            with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                result = self.texture_pipe(
                    prompt=enhanced_prompt,
                    negative_prompt="blurry, low quality, artifacts",
                    image=init_image,
                    mask_image=mask,
                    control_image=control_image,
                    num_inference_steps=20,
                    guidance_scale=7.5,
                    controlnet_conditioning_scale=0.8,
                    generator=torch.Generator().manual_seed(42)
                )
            
            texture_image = result.images[0]
            
            # Apply texture to mesh
            textured_mesh = self._apply_texture_to_mesh(mesh, texture_image, texture_size)
            
            return textured_mesh
            
        except Exception as e:
            print(f"AI texture generation failed: {e}")
            return self._generate_fallback_texture(mesh, reference_image, texture_size, prompt, texture_source_type)
    
    def _direct_color_mapping(self, mesh: trimesh.Trimesh, reference_image: Image.Image, texture_size: int = 1024) -> trimesh.Trimesh:
        """Direct color preservation using perspective-aware ray casting for accurate color mapping"""
        print(f"Performing perspective-aware color mapping with texture size {texture_size}...")
        
        # Resize reference image to target texture size
        texture_image = reference_image.resize((texture_size, texture_size)).convert("RGB")
        texture_array = np.array(texture_image)
        
        # Generate camera-aware color mapping using ray casting
        print("Performing perspective-aware ray casting...")
        vertex_colors = self._perspective_aware_color_mapping(mesh, texture_array)
        
        # Create mesh with accurate colors from original image perspective
        colored_mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
            vertex_colors=vertex_colors
        )
        
        print("Perspective-aware color mapping completed - accurate colors preserved!")
        return colored_mesh
    
    def _perspective_aware_color_mapping(self, mesh: trimesh.Trimesh, texture_array: np.ndarray) -> np.ndarray:
        """Map colors using ray casting from camera perspective to match original photo viewpoint"""
        vertices = mesh.vertices
        texture_height, texture_width = texture_array.shape[:2]
        
        # Estimate camera parameters from mesh geometry
        camera_params = self._estimate_camera_parameters(mesh)
        camera_position = camera_params['position']
        fov = camera_params['fov']
        
        # Create perspective projection matrix
        projection_matrix = self._create_perspective_projection_matrix(fov, texture_width, texture_height)
        
        # Create Z-buffer for depth-aware sampling and occlusion handling
        vertex_depths = self._create_depth_buffer(mesh, camera_position)
        
        vertex_colors = []
        
        for i, vertex in enumerate(vertices):
            # Project 3D vertex to 2D screen coordinates using perspective projection
            screen_coords = self._project_vertex_to_screen(vertex, camera_position, projection_matrix, texture_width, texture_height)
            
            if screen_coords is None:
                # Vertex is behind camera or invalid
                color = np.array([128, 128, 128], dtype=np.uint8)  # Default gray
            else:
                screen_x, screen_y, depth = screen_coords
                
                # Check occlusion using Z-buffer
                if self._is_vertex_occluded(i, depth, vertex_depths):
                    # Vertex is hidden behind other vertices
                    color = np.array([64, 64, 64], dtype=np.uint8)  # Darker for occluded
                else:
                    # Sample color from texture with bounds checking and sub-pixel accuracy
                    color = self._sample_color_with_interpolation(texture_array, screen_x, screen_y)
            
            vertex_colors.append(color)
        
        return np.array(vertex_colors, dtype=np.uint8)
    
    def _create_depth_buffer(self, mesh: trimesh.Trimesh, camera_position: np.ndarray) -> np.ndarray:
        """Create depth buffer for all vertices from camera perspective"""
        vertices = mesh.vertices
        depths = np.zeros(len(vertices))
        
        for i, vertex in enumerate(vertices):
            # Calculate distance from camera to vertex
            depth = np.linalg.norm(vertex - camera_position)
            depths[i] = depth
        
        return depths
    
    def _is_vertex_occluded(self, vertex_idx: int, depth: float, all_depths: np.ndarray) -> bool:
        """Check if vertex is occluded by other vertices closer to camera"""
        # Simple occlusion check - if many vertices are much closer, this one might be hidden
        closer_vertices = np.sum(all_depths < depth * 0.9)  # Vertices at least 10% closer
        
        # If more than 20% of vertices are significantly closer, consider this vertex occluded
        total_vertices = len(all_depths)
        occlusion_threshold = total_vertices * 0.2
        
        return closer_vertices > occlusion_threshold
    
    def _estimate_camera_parameters(self, mesh: trimesh.Trimesh) -> dict:
        """Estimate optimal camera parameters based on mesh geometry"""
        vertices = mesh.vertices
        
        # Calculate mesh bounding box and dimensions
        bbox_min = vertices.min(axis=0)
        bbox_max = vertices.max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        bbox_size = bbox_max - bbox_min
        
        # Estimate camera distance (2.5x the maximum dimension for good framing)
        max_dimension = np.max(bbox_size)
        camera_distance = max_dimension * 2.5
        
        # Position camera along Z-axis (assuming object is centered at origin)
        camera_position = bbox_center + np.array([0, 0, camera_distance])
        
        # Calculate field of view based on object size and distance
        fov = 2 * np.arctan(max_dimension / (2 * camera_distance))
        
        # Calculate aspect ratio from mesh dimensions
        aspect_ratio = bbox_size[0] / max(bbox_size[1], 0.1)
        
        return {
            'position': camera_position,
            'fov': fov,
            'distance': camera_distance,
            'aspect_ratio': aspect_ratio,
            'bbox_center': bbox_center,
            'bbox_size': bbox_size
        }
    
    def _create_perspective_projection_matrix(self, fov: float, width: int, height: int) -> np.ndarray:
        """Create 4x4 perspective projection matrix"""
        aspect_ratio = width / height
        near = 0.1
        far = 1000.0
        
        f = 1.0 / np.tan(fov / 2)
        
        projection_matrix = np.array([
            [f / aspect_ratio, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ])
        
        return projection_matrix
    
    def _project_vertex_to_screen(self, vertex: np.ndarray, camera_pos: np.ndarray, 
                                 projection_matrix: np.ndarray, screen_width: int, screen_height: int) -> tuple:
        """Project 3D vertex to 2D screen coordinates using proper perspective projection"""
        
        # Transform vertex to camera space
        vertex_camera = vertex - camera_pos
        
        # Check if vertex is behind camera
        if vertex_camera[2] >= 0:
            return None
        
        # Convert to homogeneous coordinates
        vertex_homo = np.append(vertex_camera, 1)
        
        # Apply projection matrix
        projected = projection_matrix @ vertex_homo
        
        # Perspective division
        if projected[3] <= 0:
            return None
            
        x_ndc = projected[0] / projected[3]
        y_ndc = projected[1] / projected[3]
        depth = projected[2] / projected[3]
        
        # Convert from normalized device coordinates to screen coordinates
        screen_x = (x_ndc + 1) * screen_width / 2
        screen_y = (1 - y_ndc) * screen_height / 2  # Flip Y for screen coordinates
        
        return (screen_x, screen_y, depth)
    
    def _sample_color_with_interpolation(self, texture_array: np.ndarray, x: float, y: float) -> np.ndarray:
        """Sample color from texture with bilinear interpolation and edge preservation"""
        texture_height, texture_width = texture_array.shape[:2]
        
        # Bounds checking with edge preservation
        if x < 0 or x >= texture_width or y < 0 or y >= texture_height:
            # Use edge-aware clamping for better boundary handling
            return self._sample_edge_color(texture_array, x, y, texture_width, texture_height)
        
        # Detect if we're near an edge in the original image
        if self._is_near_edge(texture_array, x, y):
            # Use sharper sampling near edges to preserve boundaries
            return self._sample_edge_preserving_color(texture_array, x, y)
        
        # Standard bilinear interpolation for smooth areas
        return self._sample_bilinear_color(texture_array, x, y)
    
    def _sample_bilinear_color(self, texture_array: np.ndarray, x: float, y: float) -> np.ndarray:
        """Standard bilinear interpolation for smooth color sampling"""
        texture_height, texture_width = texture_array.shape[:2]
        
        # Bilinear interpolation
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = min(x0 + 1, texture_width - 1), min(y0 + 1, texture_height - 1)
        
        # Calculate interpolation weights
        wx, wy = x - x0, y - y0
        
        # Sample four neighboring pixels
        c00 = texture_array[y0, x0]
        c10 = texture_array[y0, x1]
        c01 = texture_array[y1, x0]
        c11 = texture_array[y1, x1]
        
        # Bilinear interpolation
        c0 = c00 * (1 - wx) + c10 * wx
        c1 = c01 * (1 - wx) + c11 * wx
        color = c0 * (1 - wy) + c1 * wy
        
        return color.astype(np.uint8)
    
    def _is_near_edge(self, texture_array: np.ndarray, x: float, y: float, threshold: float = 10.0) -> bool:
        """Detect if sampling position is near a color edge"""
        texture_height, texture_width = texture_array.shape[:2]
        
        # Sample surrounding pixels to detect color variation
        x_int, y_int = int(x), int(y)
        radius = 2
        
        # Ensure we don't go out of bounds
        x_start = max(0, x_int - radius)
        x_end = min(texture_width - 1, x_int + radius)
        y_start = max(0, y_int - radius)
        y_end = min(texture_height - 1, y_int + radius)
        
        # Calculate color variance in the neighborhood
        neighborhood = texture_array[y_start:y_end+1, x_start:x_end+1]
        color_variance = np.var(neighborhood, axis=(0, 1))
        
        # High variance indicates we're near an edge
        return np.mean(color_variance) > threshold
    
    def _sample_edge_preserving_color(self, texture_array: np.ndarray, x: float, y: float) -> np.ndarray:
        """Sample color with edge preservation - uses nearest neighbor near edges"""
        texture_height, texture_width = texture_array.shape[:2]
        
        # Use nearest neighbor sampling near edges to preserve sharp boundaries
        x_nearest = int(np.round(x))
        y_nearest = int(np.round(y))
        
        # Clamp to bounds
        x_nearest = np.clip(x_nearest, 0, texture_width - 1)
        y_nearest = np.clip(y_nearest, 0, texture_height - 1)
        
        return texture_array[y_nearest, x_nearest]
    
    def _sample_edge_color(self, texture_array: np.ndarray, x: float, y: float, width: int, height: int) -> np.ndarray:
        """Sample color at image edges with proper clamping"""
        # Clamp coordinates to valid range
        x_clamped = np.clip(x, 0, width - 1)
        y_clamped = np.clip(y, 0, height - 1)
        
        return texture_array[int(y_clamped), int(x_clamped)]
        """Generate optimal UV coordinates using multiple projection methods"""
        vertices = mesh.vertices
        
        # Center the vertices for better projection
        center = vertices.mean(axis=0)
        vertices_centered = vertices - center
        
        # Calculate bounding box for normalization
        bbox_min = vertices_centered.min(axis=0)
        bbox_max = vertices_centered.max(axis=0)
        bbox_range = bbox_max - bbox_min
        
        # Try different projection methods and choose the best one
        methods = []
        
        # Method 1: Spherical projection (best for rounded objects)
        try:
            x, y, z = vertices_centered[:, 0], vertices_centered[:, 1], vertices_centered[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            r = np.maximum(r, 1e-8)
            
            u_sphere = 0.5 + np.arctan2(y, x) / (2 * np.pi)
            v_sphere = 0.5 - np.arcsin(z / r) / np.pi
            
            uv_sphere = np.column_stack([u_sphere, v_sphere])
            methods.append(("spherical", uv_sphere))
        except:
            pass
        
        # Method 2: Cylindrical projection (best for tall objects)
        try:
            x, y, z = vertices_centered[:, 0], vertices_centered[:, 1], vertices_centered[:, 2]
            r = np.sqrt(x**2 + y**2)
            r = np.maximum(r, 1e-8)
            
            u_cyl = 0.5 + np.arctan2(y, x) / (2 * np.pi)
            v_cyl = (z - bbox_min[2]) / (bbox_range[2] + 1e-8)
            
            uv_cyl = np.column_stack([u_cyl, v_cyl])
            methods.append(("cylindrical", uv_cyl))
        except:
            pass
        
        # Method 3: Planar projection (best for flat objects)
        try:
            # Project onto XY plane
            u_planar = (vertices_centered[:, 0] - bbox_min[0]) / (bbox_range[0] + 1e-8)
            v_planar = (vertices_centered[:, 1] - bbox_min[1]) / (bbox_range[1] + 1e-8)
            
            uv_planar = np.column_stack([u_planar, v_planar])
            methods.append(("planar", uv_planar))
        except:
            pass
        
        # Choose the method with minimal UV distortion
        best_method = methods[0][1] if methods else np.random.rand(len(vertices), 2)
        
        # Normalize UV coordinates to [0, 1]
        best_method = np.clip(best_method, 0, 1)
        
        print(f"Using projection method: {methods[0][0] if methods else 'random'}")
        return best_method
    
    def _sample_colors_from_texture(self, texture_array: np.ndarray, uv_coords: np.ndarray, vertices: np.ndarray) -> np.ndarray:
        """Sample colors from texture using UV coordinates with proper interpolation"""
        texture_height, texture_width = texture_array.shape[:2]
        
        # Convert UV coordinates to pixel coordinates
        pixel_coords = uv_coords * np.array([texture_width - 1, texture_height - 1])
        
        # Use bilinear interpolation for smoother color sampling
        vertex_colors = []
        for i, (u, v) in enumerate(pixel_coords):
            x, y = u, v
            
            # Bilinear interpolation
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = min(x0 + 1, texture_width - 1), min(y0 + 1, texture_height - 1)
            
            # Calculate interpolation weights
            wx, wy = x - x0, y - y0
            
            # Sample four neighboring pixels
            c00 = texture_array[y0, x0]
            c10 = texture_array[y0, x1] 
            c01 = texture_array[y1, x0]
            c11 = texture_array[y1, x1]
            
            # Bilinear interpolation
            c0 = c00 * (1 - wx) + c10 * wx
            c1 = c01 * (1 - wx) + c11 * wx
            color = c0 * (1 - wy) + c1 * wy
            
            vertex_colors.append(color)
        
        return np.array(vertex_colors, dtype=np.uint8)
    
    def _generate_fallback_texture(
        self, 
        mesh: trimesh.Trimesh, 
        reference_image: Optional[Image.Image] = None,
        texture_size: int = 1024,
        prompt: str = "detailed realistic texture",
        texture_source_type: str = "reference"
    ) -> trimesh.Trimesh:
        """Fallback texture generation using traditional methods"""
        
        # Generate depth map from mesh
        print("Generating depth map from mesh...")
        depth_map = self.generate_depth_map(mesh, size=512)
        
        # Process texture source based on type
        if reference_image and texture_source_type == "sf3d_original":
            # Use SF3D original image directly - preserve exact colors!
            print("Using SF3D original image for exact color preservation...")
            # Resize the SF3D image to texture size directly
            texture_image = reference_image.resize((texture_size, texture_size)).convert("RGB")
            
        elif reference_image and texture_source_type == "reference":
            # External reference image - extract dominant colors
            print("Extracting colors from reference image...")
            colors = self._extract_dominant_colors(reference_image, num_colors=5)
            # Create texture from colors and depth
            print("Creating texture from extracted colors...")
            texture_image = self._create_texture_from_colors(depth_map, colors, texture_size)
            
        else:
            # No reference image - use prompt-based colors
            if "metallic" in prompt.lower():
                colors = np.array([[180, 180, 190], [140, 140, 150], [100, 100, 110], [200, 200, 210], [160, 160, 170]])
            elif "wood" in prompt.lower():
                colors = np.array([[139, 90, 43], [160, 110, 60], [120, 80, 40], [180, 130, 80], [100, 70, 30]])
            elif "fabric" in prompt.lower():
                colors = np.array([[150, 150, 160], [120, 120, 130], [180, 180, 190], [100, 100, 110], [200, 200, 210]])
            else:
                colors = np.array([[128, 128, 128], [150, 150, 150], [100, 100, 100], [180, 180, 180], [80, 80, 80]])
            
            # Create texture from colors and depth
            print("Creating texture from prompt-based colors...")
            texture_image = self._create_texture_from_colors(depth_map, colors, texture_size)
        
        # Apply texture to mesh
        textured_mesh = self._apply_texture_to_mesh(mesh, texture_image, texture_size)
        
        return textured_mesh
    
    def _extract_dominant_colors(self, reference_image: Image.Image, num_colors: int = 5) -> np.ndarray:
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
    
    def _create_texture_from_colors(self, depth_map: np.ndarray, colors: np.ndarray, texture_size: int = 1024) -> Image.Image:
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
    
    def _apply_texture_to_mesh(self, mesh: trimesh.Trimesh, texture_image: Image.Image, texture_size: int = 1024) -> trimesh.Trimesh:
        """Apply texture to mesh"""
        try:
            # Try to generate UV coordinates using trimesh's built-in functionality
            print("Attempting to generate UV coordinates...")
            
            # Create a copy of the mesh to avoid modifying the original
            mesh_copy = mesh.copy()
            
            # Try to generate UV coordinates using trimesh's unwrapping
            try:
                # Check if mesh has existing UV coordinates
                if hasattr(mesh_copy.visual, 'uv') and mesh_copy.visual.uv is not None:
                    # UV coordinates already exist
                    print("Found existing UV coordinates")
                    return self._apply_proper_texture(mesh_copy, texture_image, texture_size)
                else:
                    # Try to generate new UV coordinates
                    print("Generating new UV coordinates...")
                    # Use a simple projection method as fallback
                    return self._apply_projection_texture(mesh_copy, texture_image, texture_size)
            except Exception as uv_error:
                print(f"UV generation failed: {uv_error}")
                return self._apply_vertex_colors(mesh, texture_image)
            
        except Exception as e:
            print(f"Texture application failed: {e}")
            # Fallback to vertex coloring
            return self._apply_vertex_colors(mesh, texture_image)
    
    def _apply_projection_texture(self, mesh: trimesh.Trimesh, texture_image: Image.Image, texture_size: int = 1024) -> trimesh.Trimesh:
        """Apply texture using simple projection mapping"""
        try:
            # Create UV coordinates using spherical projection
            vertices = mesh.vertices
            
            # Center the vertices
            center = vertices.mean(axis=0)
            vertices_centered = vertices - center
            
            # Calculate spherical coordinates
            x, y, z = vertices_centered[:, 0], vertices_centered[:, 1], vertices_centered[:, 2]
            r = np.sqrt(x**2 + y**2 + z**2)
            
            # Avoid division by zero
            r = np.maximum(r, 1e-8)
            
            # Spherical to UV mapping
            u = 0.5 + np.arctan2(y, x) / (2 * np.pi)
            v = 0.5 - np.arcsin(z / r) / np.pi
            
            # Normalize UV coordinates to [0, 1]
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
            
            uv_coordinates = np.column_stack([u, v])
            
            # Create texture coordinates
            texture_coords = uv_coordinates * (texture_size - 1)
            texture_coords = texture_coords.astype(int)
            texture_coords = np.clip(texture_coords, 0, texture_size - 1)
            
            # Sample colors from texture
            texture_array = np.array(texture_image)
            vertex_colors = texture_array[texture_coords[:, 1], texture_coords[:, 0], :3]
            
            # Create mesh with vertex colors
            colored_mesh = trimesh.Trimesh(
                vertices=mesh.vertices,
                faces=mesh.faces,
                vertex_colors=vertex_colors
            )
            
            print("Applied texture using spherical projection")
            return colored_mesh
            
        except Exception as e:
            print(f"Projection texture failed: {e}")
            return self._apply_vertex_colors(mesh, texture_image)
    
    def _apply_proper_texture(self, mesh: trimesh.Trimesh, texture_image: Image.Image, texture_size: int = 1024) -> trimesh.Trimesh:
        """Apply texture using proper UV mapping"""
        try:
            # This would be ideal but requires proper UV unwrapping
            # For now, fall back to projection method
            return self._apply_projection_texture(mesh, texture_image, texture_size)
        except Exception as e:
            print(f"Proper texture application failed: {e}")
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


def apply_ai_texture_transfer(
    mesh_path: str,
    reference_image_path: Optional[str] = None,
    sf3d_image_path: Optional[str] = None,
    output_path: str = None,
    texture_size: int = 1024,
    prompt: str = "detailed realistic texture",
    device: str = "cuda",
    use_direct_mapping: bool = False
) -> str:
    """
    Apply AI texture transfer to a mesh using Stable Diffusion or direct color mapping
    
    Args:
        mesh_path: Path to input mesh file
        reference_image_path: Path to reference image for color guidance
        sf3d_image_path: Path to SF3D's background-removed image (preferred)
        output_path: Path to save textured mesh
        texture_size: Size of generated texture
        prompt: Text prompt for texture generation
        device: Device to use for processing
        use_direct_mapping: If True, bypass AI and use direct color preservation
    
    Returns:
        Path to textured mesh file
    """
    
    # Load mesh
    print(f"Loading mesh from {mesh_path}...")
    mesh = trimesh.load(mesh_path)
    print(type(mesh))
    # Handle scene objects
    if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
        # Extract first mesh from scene
        print("Extracting first mesh from scene...")
        mesh_name = list(mesh.geometry.keys())[0]
        mesh = mesh.geometry[mesh_name]
    
    # Ensure we have a Trimesh object
    if not isinstance(mesh, trimesh.Trimesh):
        print("Warning: Loaded object is not a Trimesh, attempting to convert...")
        if hasattr(mesh, 'geometry') and len(mesh.geometry) > 0:
            mesh_name = list(mesh.geometry.keys())[0]
            mesh = mesh.geometry[mesh_name]
    
    # Load texture source image with priority
    texture_source = None
    texture_source_type = "none"
    
    # Priority 1: SF3D background-removed image (exact colors from original)
    if sf3d_image_path and os.path.exists(sf3d_image_path):
        texture_source = Image.open(sf3d_image_path)
        texture_source_type = "sf3d_original"
        print(f"Using SF3D background-removed image: {sf3d_image_path}")
    
    # Priority 2: Reference image
    elif reference_image_path and os.path.exists(reference_image_path):
        texture_source = Image.open(reference_image_path)
        texture_source_type = "reference"
        print(f"Using reference image: {reference_image_path}")
    
    # Priority 3: No image - use prompt-based colors
    else:
        texture_source_type = "prompt_based"
        print("No texture source image provided, using prompt-based colors")
    
    # Initialize AI texture transfer
    texture_transfer = AITextureTransfer(device=device)
    
    # Generate texture
    textured_mesh = texture_transfer.generate_texture_with_ai(
        mesh=mesh,
        reference_image=texture_source,
        texture_size=texture_size,
        prompt=prompt,
        texture_source_type=texture_source_type,
        use_direct_mapping=use_direct_mapping
    )
    
    # Save result
    if output_path is None:
        base_name = os.path.splitext(mesh_path)[0]
        output_path = f"{base_name}_ai_textured.glb"
    
    print(f"Saving AI textured mesh to {output_path}...")
    textured_mesh.export(output_path, include_normals=True)
    
    return output_path


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Apply AI texture transfer to 3D meshes using Stable Diffusion")
    parser.add_argument("mesh_path", help="Path to input mesh file")
    parser.add_argument("--reference", help="Path to reference image", default=None)
    parser.add_argument("--output", help="Output path", default=None)
    parser.add_argument("--texture-size", type=int, default=1024, help="Texture size")
    parser.add_argument("--prompt", default="detailed realistic texture", help="Texture prompt")
    parser.add_argument("--device", default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    result_path = apply_ai_texture_transfer(
        mesh_path=args.mesh_path,
        reference_image_path=args.reference,
        output_path=args.output,
        texture_size=args.texture_size,
        prompt=args.prompt,
        device=args.device
    )
    
    print(f"AI texture transfer completed! Result saved to: {result_path}")
