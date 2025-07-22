'''
enhanced_noise_modeling.py
Complete implementation of CORTO Figure 8 Noise Pipeline
with 8-step systematic noise modeling
'''

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass

@dataclass
class NoiseParameters:
    """Noise parameters following CORTO Figure 8"""
    # Generic blur (1)
    generic_blur_sigma: float = 1.0
    
    # Motion blur (2)  
    motion_blur_angle: float = 0.0
    motion_blur_length: int = 5
    
    # Generic noise (3)
    noise_type: str = 'gaussian'
    noise_mean: float = 0.0
    noise_variance: float = 0.01
    
    # Gamma correction (4)
    gamma_value: float = 2.2
    
    # Dead pixels (5)
    dead_pixel_probability: float = 0.001
    
    # Dead buckets (6)
    dead_bucket_probability: float = 0.0005
    dead_bucket_size: Tuple[int, int] = (8, 8)
    
    # Blooming (7)
    blooming_threshold: float = 0.9
    blooming_spread: int = 3
    
    # Radiation effects (8)
    radiation_line_probability: float = 0.002
    radiation_intensity: float = 1.0

class EnhancedNoiseModeling:
    """
    Complete CORTO Figure 8 Noise Modeling Pipeline
    
    Implements all 8 noise sources systematically:
    1. Generic blur
    2. Motion blur  
    3. Generic noise
    4. Gamma correction
    5. Dead pixels
    6. Dead buckets
    7. Blooming
    8. Radiation effects
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def apply_complete_noise_pipeline(self, 
                                    image: np.ndarray,
                                    noise_params: NoiseParameters,
                                    apply_all: bool = True,
                                    selected_steps: Optional[List[int]] = None) -> np.ndarray:
        """
        Apply complete 8-step noise pipeline per CORTO Figure 8
        
        Args:
            image: Input clean image [0,1] range
            noise_params: Noise parameters
            apply_all: Apply all 8 steps
            selected_steps: List of specific steps to apply (1-8)
        """
        
        if selected_steps is None and apply_all:
            selected_steps = list(range(1, 9))
        elif selected_steps is None:
            selected_steps = []
            
        self.logger.info(f"Applying noise pipeline steps: {selected_steps}")
        
        # Ensure image is in float32 [0,1] range
        noisy_image = image.astype(np.float32)
        if noisy_image.max() > 1.0:
            noisy_image = noisy_image / 255.0
            
        # Step 1: Generic blur
        if 1 in selected_steps:
            noisy_image = self._apply_generic_blur(noisy_image, noise_params)
            
        # Step 2: Motion blur
        if 2 in selected_steps:
            noisy_image = self._apply_motion_blur(noisy_image, noise_params)
            
        # Step 3: Generic noise
        if 3 in selected_steps:
            noisy_image = self._apply_generic_noise(noisy_image, noise_params)
            
        # Step 4: Gamma correction
        if 4 in selected_steps:
            noisy_image = self._apply_gamma_correction(noisy_image, noise_params)
            
        # Step 5: Dead pixels
        if 5 in selected_steps:
            noisy_image = self._apply_dead_pixels(noisy_image, noise_params)
            
        # Step 6: Dead buckets  
        if 6 in selected_steps:
            noisy_image = self._apply_dead_buckets(noisy_image, noise_params)
            
        # Step 7: Blooming
        if 7 in selected_steps:
            noisy_image = self._apply_blooming(noisy_image, noise_params)
            
        # Step 8: Radiation effects
        if 8 in selected_steps:
            noisy_image = self._apply_radiation_effects(noisy_image, noise_params)
            
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image
        
    def _apply_generic_blur(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 1: Generic blur using Gaussian filter"""
        kernel_size = int(params.generic_blur_sigma * 4) * 2 + 1  # Ensure odd
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), params.generic_blur_sigma)
        
    def _apply_motion_blur(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 2: Motion blur using directional kernel"""
        # Create motion blur kernel
        kernel_size = params.motion_blur_length
        kernel = np.zeros((kernel_size, kernel_size))
        
        # Calculate motion direction
        angle_rad = np.radians(params.motion_blur_angle)
        center = kernel_size // 2
        
        for i in range(kernel_size):
            x = int(center + (i - center) * np.cos(angle_rad))
            y = int(center + (i - center) * np.sin(angle_rad))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1
                
        # Normalize kernel
        kernel = kernel / np.sum(kernel) if np.sum(kernel) > 0 else kernel
        
        # Apply motion blur
        return cv2.filter2D(image, -1, kernel)
        
    def _apply_generic_noise(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 3: Generic noise addition"""
        if params.noise_type == 'gaussian':
            noise = np.random.normal(params.noise_mean, np.sqrt(params.noise_variance), image.shape)
        elif params.noise_type == 'poisson':
            # Poisson noise simulation
            scaled = image * 100  # Scale for Poisson
            noise = (np.random.poisson(scaled) - scaled) / 100
        elif params.noise_type == 'salt_pepper':
            noise = np.zeros_like(image)
            # Salt noise
            salt_coords = np.random.random(image.shape) < params.noise_variance / 2
            noise[salt_coords] = 1
            # Pepper noise  
            pepper_coords = np.random.random(image.shape) < params.noise_variance / 2
            noise[pepper_coords] = -1
        else:
            noise = np.zeros_like(image)
            
        return image + noise
        
    def _apply_gamma_correction(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 4: Gamma correction"""
        return np.power(image, 1.0 / params.gamma_value)
        
    def _apply_dead_pixels(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 5: Dead pixels simulation"""
        dead_mask = np.random.random(image.shape[:2]) < params.dead_pixel_probability
        
        noisy_image = image.copy()
        if len(image.shape) == 3:
            for c in range(image.shape[2]):
                noisy_image[:, :, c][dead_mask] = 0
        else:
            noisy_image[dead_mask] = 0
            
        return noisy_image
        
    def _apply_dead_buckets(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 6: Dead buckets (entire rows/columns)"""
        noisy_image = image.copy()
        h, w = image.shape[:2]
        
        # Dead rows
        n_dead_rows = int(h * params.dead_bucket_probability)
        if n_dead_rows > 0:
            dead_rows = np.random.choice(h, n_dead_rows, replace=False)
            if len(image.shape) == 3:
                noisy_image[dead_rows, :, :] = 0
            else:
                noisy_image[dead_rows, :] = 0
                
        # Dead columns
        n_dead_cols = int(w * params.dead_bucket_probability)
        if n_dead_cols > 0:
            dead_cols = np.random.choice(w, n_dead_cols, replace=False)
            if len(image.shape) == 3:
                noisy_image[:, dead_cols, :] = 0
            else:
                noisy_image[:, dead_cols] = 0
                
        return noisy_image
        
    def _apply_blooming(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 7: Blooming effects for saturated pixels"""
        noisy_image = image.copy()
        
        # Find saturated pixels
        if len(image.shape) == 3:
            saturated = np.any(image > params.blooming_threshold, axis=2)
        else:
            saturated = image > params.blooming_threshold
            
        # Apply blooming effect
        if np.any(saturated):
            # Create blooming kernel
            kernel_size = params.blooming_spread * 2 + 1
            kernel = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
            
            # Convolve saturated areas
            if len(image.shape) == 3:
                for c in range(image.shape[2]):
                    bloom_effect = cv2.filter2D(saturated.astype(np.float32), -1, kernel)
                    noisy_image[:, :, c] = np.maximum(noisy_image[:, :, c], bloom_effect * 0.3)
            else:
                bloom_effect = cv2.filter2D(saturated.astype(np.float32), -1, kernel)
                noisy_image = np.maximum(noisy_image, bloom_effect * 0.3)
                
        return noisy_image
        
    def _apply_radiation_effects(self, image: np.ndarray, params: NoiseParameters) -> np.ndarray:
        """Step 8: Radiation effects (random lines)"""
        noisy_image = image.copy()
        h, w = image.shape[:2]
        
        # Random horizontal lines
        n_lines = int(h * params.radiation_line_probability)
        if n_lines > 0:
            line_rows = np.random.choice(h, n_lines, replace=False)
            for row in line_rows:
                # Random line intensity
                intensity = np.random.uniform(0.5, 1.0) * params.radiation_intensity
                if len(image.shape) == 3:
                    noisy_image[row, :, :] = np.minimum(noisy_image[row, :, :] + intensity, 1.0)
                else:
                    noisy_image[row, :] = np.minimum(noisy_image[row, :] + intensity, 1.0)
                    
        # Random vertical lines (less frequent)
        n_vlines = int(w * params.radiation_line_probability * 0.3)
        if n_vlines > 0:
            line_cols = np.random.choice(w, n_vlines, replace=False)
            for col in line_cols:
                intensity = np.random.uniform(0.3, 0.7) * params.radiation_intensity
                if len(image.shape) == 3:
                    noisy_image[:, col, :] = np.minimum(noisy_image[:, col, :] + intensity, 1.0)
                else:
                    noisy_image[:, col] = np.minimum(noisy_image[:, col] + intensity, 1.0)
                    
        return noisy_image

    def create_systematic_noise_variations(self, 
                                         image: np.ndarray,
                                         base_params: NoiseParameters,
                                         variation_factor: float = 0.3,
                                         n_variations: int = 5) -> List[np.ndarray]:
        """Create systematic noise variations for validation"""
        
        variations = []
        
        for i in range(n_variations):
            # Create parameter variation
            varied_params = NoiseParameters(
                generic_blur_sigma=base_params.generic_blur_sigma * (1 + np.random.uniform(-variation_factor, variation_factor)),
                motion_blur_angle=np.random.uniform(0, 360),
                motion_blur_length=max(1, int(base_params.motion_blur_length * (1 + np.random.uniform(-variation_factor, variation_factor)))),
                noise_variance=base_params.noise_variance * (1 + np.random.uniform(-variation_factor, variation_factor)),
                gamma_value=base_params.gamma_value * (1 + np.random.uniform(-variation_factor/2, variation_factor/2)),
                dead_pixel_probability=base_params.dead_pixel_probability * (1 + np.random.uniform(-variation_factor, variation_factor)),
                dead_bucket_probability=base_params.dead_bucket_probability * (1 + np.random.uniform(-variation_factor, variation_factor)),
                blooming_threshold=np.clip(base_params.blooming_threshold + np.random.uniform(-0.1, 0.1), 0.7, 1.0),
                radiation_line_probability=base_params.radiation_line_probability * (1 + np.random.uniform(-variation_factor, variation_factor))
            )
            
            # Apply noise pipeline
            noisy_image = self.apply_complete_noise_pipeline(image, varied_params)
            variations.append(noisy_image)
            
        self.logger.info(f"Created {len(variations)} systematic noise variations")
        return variations
