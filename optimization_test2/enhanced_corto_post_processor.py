'''
enhanced_corto_post_processor.py
Complete implementation of CORTO Figure 12 Post-Processing Pipeline
with systematic domain randomization and onboard readiness
'''

import numpy as np
import cv2
import json
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging

@dataclass
class ProcessingParameters:
    """Storage for pipeline parameters enabling onboard implementation"""
    gamma1: float
    gamma2: float  
    gamma3: float
    gamma4: float
    alpha_u: float
    alpha_v: float
    gamma_i: int
    scale_factor: float
    target_size: int
    blob_strategy: str
    padding_method: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'gamma1': self.gamma1,
            'gamma2': self.gamma2,
            'gamma3': self.gamma3, 
            'gamma4': self.gamma4,
            'alpha_u': self.alpha_u,
            'alpha_v': self.alpha_v,
            'gamma_i': self.gamma_i,
            'scale_factor': self.scale_factor,
            'target_size': self.target_size,
            'blob_strategy': self.blob_strategy,
            'padding_method': self.padding_method
        }

class EnhancedCORTOPostProcessor:
    """
    Complete CORTO Figure 12 Post-Processing Pipeline Implementation
    
    Features:
    - Systematic blob analysis (N-th biggest blobs grouping)
    - Domain randomization with multiple strategies
    - Onboard implementation readiness
    - Complete label transformation pipeline
    - Reversible transformations for ML outputs
    """
    
    def __init__(self, target_size: int = 128, enable_domain_randomization: bool = True):
        self.target_size = target_size
        self.enable_domain_randomization = enable_domain_randomization
        self.processing_params: Optional[ProcessingParameters] = None
        
        # Domain randomization parameters
        self.padding_strategies = ['random', 'center', 'corner', 'systematic']
        self.blob_strategies = ['largest', 'n_largest', 'grouped']
        
        # Statistics for systematic domain randomization
        self.generation_stats = {
            'total_processed': 0,
            'padding_distribution': {strategy: 0 for strategy in self.padding_strategies},
            'blob_distribution': {strategy: 0 for strategy in self.blob_strategies}
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_image_label_pair_enhanced(self, 
                                        image: np.ndarray, 
                                        labels: Dict,
                                        target_size: Optional[int] = None,
                                        domain_randomization_mode: str = 'systematic',
                                        blob_strategy: str = 'grouped') -> Tuple[np.ndarray, Dict, ProcessingParameters]:
        """
        Enhanced S0→S1→S2 pipeline with systematic domain randomization
        
        Args:
            image: Input image in S0 space
            labels: Dictionary of labels
            target_size: Final target size (M)
            domain_randomization_mode: 'random', 'systematic', 'balanced'
            blob_strategy: 'largest', 'n_largest', 'grouped'
            
        Returns:
            Processed image, updated labels, processing parameters
        """
        if target_size is None:
            target_size = self.target_size
            
        self.logger.info(f"Processing image {image.shape} with {blob_strategy} blob strategy")
        
        # S0 → S1: Enhanced binarization, blob analysis, systematic padding
        image_s1, labels_s1, transform_s1 = self._transform_s0_to_s1_enhanced(
            image, labels, blob_strategy, domain_randomization_mode
        )
        
        # S1 → S2: Resize to target size
        image_s2, labels_s2, transform_s2 = self._transform_s1_to_s2_enhanced(
            image_s1, labels_s1, target_size
        )
        
        # Create comprehensive processing parameters for onboard implementation
        self.processing_params = ProcessingParameters(
            gamma1=transform_s1['Gamma1'],
            gamma2=transform_s1['Gamma2'],
            gamma3=transform_s1['Gamma3'],
            gamma4=transform_s1['Gamma4'],
            alpha_u=transform_s1['alpha_u'],
            alpha_v=transform_s1['alpha_v'],
            gamma_i=transform_s1['gamma_i'],
            scale_factor=transform_s2['scale_factor'],
            target_size=target_size,
            blob_strategy=blob_strategy,
            padding_method=domain_randomization_mode
        )
        
        # Update statistics for domain randomization monitoring
        self._update_generation_stats(blob_strategy, domain_randomization_mode)
        
        return image_s2, labels_s2, self.processing_params

    def _transform_s0_to_s1_enhanced(self, 
                                   image: np.ndarray, 
                                   labels: Dict,
                                   blob_strategy: str,
                                   padding_mode: str) -> Tuple[np.ndarray, Dict, Dict]:
        """Enhanced S0 → S1 transformation with systematic blob analysis"""
        
        # 1. Enhanced Binarization with adaptive thresholding
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Adaptive threshold for better blob detection
        binary = cv2.adaptiveThreshold(
            (gray * 255).astype(np.uint8),
            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        ) / 255.0
        
        # 2. Complete Blob Analysis - N-th biggest blobs grouping
        bounding_box = self._enhanced_blob_analysis(binary, blob_strategy)
        
        # 3. Systematic Padding with domain randomization
        gamma1, gamma2, gamma3, gamma4 = bounding_box
        alpha_u, alpha_v, gamma_i = self._systematic_padding(
            gamma3, gamma4, padding_mode
        )
        
        # 4. Create padded image
        padded_image = self._create_padded_image(
            image, gamma1, gamma2, gamma3, gamma4, alpha_u, alpha_v, gamma_i
        )
        
        # 5. Update labels with enhanced transformation
        labels_s1 = self._update_labels_s0_to_s1_enhanced(
            labels, gamma1, gamma2, alpha_u, alpha_v
        )
        
        transform_s1 = {
            'Gamma1': gamma1, 'Gamma2': gamma2,
            'Gamma3': gamma3, 'Gamma4': gamma4,
            'alpha_u': alpha_u, 'alpha_v': alpha_v,
            'gamma_i': gamma_i,
            'blob_strategy': blob_strategy,
            'padding_mode': padding_mode
        }
        
        return padded_image, labels_s1, transform_s1

    def _enhanced_blob_analysis(self, binary: np.ndarray, strategy: str) -> List[float]:
        """Complete blob analysis implementation per CORTO requirements"""
        
        contours, _ = cv2.findContours(
            (binary * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return [0, 0, binary.shape[1], binary.shape[0]]
        
        if strategy == 'largest':
            # Original implementation - largest contour only
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            return [x, y, w, h]
            
        elif strategy == 'n_largest':
            # N largest contours analysis
            n = min(3, len(contours))  # Top 3 contours
            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:n]
            
            # Find combined bounding box
            all_points = np.concatenate(sorted_contours)
            x, y, w, h = cv2.boundingRect(all_points)
            return [x, y, w, h]
            
        elif strategy == 'grouped':
            # CORTO Paper: "N-th biggest blobs grouped together"
            # Group blobs by proximity and size
            blob_data = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > binary.size * 0.001:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    blob_data.append({'contour': contour, 'area': area, 'bbox': [x, y, w, h]})
            
            if not blob_data:
                return [0, 0, binary.shape[1], binary.shape[0]]
            
            # Sort by area and group largest blobs
            blob_data.sort(key=lambda x: x['area'], reverse=True)
            n_group = min(5, len(blob_data))  # Group top 5 blobs
            
            # Create combined bounding box
            min_x = min(blob['bbox'][0] for blob in blob_data[:n_group])
            min_y = min(blob['bbox'][1] for blob in blob_data[:n_group])
            max_x = max(blob['bbox'][0] + blob['bbox'][2] for blob in blob_data[:n_group])
            max_y = max(blob['bbox'][1] + blob['bbox'][3] for blob in blob_data[:n_group])
            
            return [min_x, min_y, max_x - min_x, max_y - min_y]
        
        else:
            raise ValueError(f"Unknown blob strategy: {strategy}")

    def _systematic_padding(self, gamma3: float, gamma4: float, mode: str) -> Tuple[float, float, int]:
        """Systematic padding for domain randomization"""
        
        max_dim = max(gamma3, gamma4)
        
        if mode == 'random':
            # Standard random padding
            alpha_u = np.random.uniform(0, max_dim - gamma3)
            alpha_v = np.random.uniform(0, max_dim - gamma4)
            
        elif mode == 'center':
            # Centered padding
            alpha_u = (max_dim - gamma3) / 2
            alpha_v = (max_dim - gamma4) / 2
            
        elif mode == 'corner':
            # Corner padding (systematic corners)
            corner = self.generation_stats['total_processed'] % 4
            if corner == 0:  # Top-left
                alpha_u, alpha_v = 0, 0
            elif corner == 1:  # Top-right
                alpha_u, alpha_v = max_dim - gamma3, 0
            elif corner == 2:  # Bottom-left
                alpha_u, alpha_v = 0, max_dim - gamma4
            else:  # Bottom-right
                alpha_u, alpha_v = max_dim - gamma3, max_dim - gamma4
                
        elif mode == 'systematic':
            # Systematic grid sampling for balanced dataset
            grid_size = 3  # 3x3 grid
            total_positions = grid_size * grid_size
            position = self.generation_stats['total_processed'] % total_positions
            
            row = position // grid_size
            col = position % grid_size
            
            alpha_u = col * (max_dim - gamma3) / (grid_size - 1) if grid_size > 1 else 0
            alpha_v = row * (max_dim - gamma4) / (grid_size - 1) if grid_size > 1 else 0
            
        else:
            raise ValueError(f"Unknown padding mode: {mode}")
        
        # Calculate final padded size
        gamma_i = int(max_dim + max(alpha_u, alpha_v))
        
        return alpha_u, alpha_v, gamma_i

    def _create_padded_image(self, image: np.ndarray, 
                           gamma1: float, gamma2: float, 
                           gamma3: float, gamma4: float,
                           alpha_u: float, alpha_v: float, 
                           gamma_i: int) -> np.ndarray:
        """Create padded image with enhanced boundary handling"""
        
        # Create padded image with proper dimensions
        if len(image.shape) == 3:
            padded_image = np.zeros((gamma_i, gamma_i, image.shape[2]), dtype=image.dtype)
        else:
            padded_image = np.zeros((gamma_i, gamma_i), dtype=image.dtype)
        
        # Calculate placement coordinates
        start_x = int(alpha_u)
        start_y = int(alpha_v)
        end_x = start_x + int(gamma3)
        end_y = start_y + int(gamma4)
        
        # Ensure coordinates are within bounds
        start_x = max(0, min(start_x, gamma_i))
        start_y = max(0, min(start_y, gamma_i))
        end_x = max(start_x, min(end_x, gamma_i))
        end_y = max(start_y, min(end_y, gamma_i))
        
        # Extract and place original content
        src_start_x = max(0, int(gamma1))
        src_start_y = max(0, int(gamma2))
        src_end_x = min(image.shape[1], src_start_x + int(gamma3))
        src_end_y = min(image.shape[0], src_start_y + int(gamma4))
        
        if len(image.shape) == 3:
            padded_image[start_y:end_y, start_x:end_x] = \
                image[src_start_y:src_end_y, src_start_x:src_end_x]
        else:
            padded_image[start_y:end_y, start_x:end_x] = \
                image[src_start_y:src_end_y, src_start_x:src_end_x]
        
        return padded_image

    def _transform_s1_to_s2_enhanced(self, image_s1: np.ndarray, 
                                   labels_s1: Dict,
                                   target_size: int) -> Tuple[np.ndarray, Dict, Dict]:
        """Enhanced S1 → S2 transformation with quality preservation"""
        
        current_size = image_s1.shape[0]
        scale_factor = target_size / current_size
        
        # Enhanced resizing with quality preservation
        if len(image_s1.shape) == 3:
            resized = cv2.resize(image_s1, (target_size, target_size), 
                               interpolation=cv2.INTER_CUBIC)
        else:
            resized = cv2.resize(image_s1, (target_size, target_size), 
                               interpolation=cv2.INTER_CUBIC)
        
        # Enhanced label transformation
        labels_s2 = self._update_labels_s1_to_s2_enhanced(labels_s1, scale_factor)
        
        transform_s2 = {
            'scale_factor': scale_factor,
            'target_size': target_size,
            'interpolation': 'cubic'
        }
        
        return resized, labels_s2, transform_s2

    def _update_labels_s0_to_s1_enhanced(self, labels: Dict, 
                                       gamma1: float, gamma2: float,
                                       alpha_u: float, alpha_v: float) -> Dict:
        """Enhanced label transformation S0 → S1 with validation"""
        
        labels_s1 = {}
        translation = np.array([alpha_u - gamma1, alpha_v - gamma2])
        
        # Enhanced transformations with validation
        for key, value in labels.items():
            if key in ['CoB', 'CoM', 'center_of_brightness', 'center_of_mass']:
                if isinstance(value, (list, np.ndarray)) and len(value) >= 2:
                    labels_s1[key] = (np.array(value[:2]) + translation).tolist()
                else:
                    labels_s1[key] = value
                    
            elif key in ['delta', 'correction_vector']:
                labels_s1[key] = value  # Unchanged per Appendix A
                
            elif key in ['range', 'distance', 'phase_angle']:
                labels_s1[key] = value  # Unchanged per Appendix A
                
            elif key in ['position', 'xyz_position']:
                labels_s1[key] = value  # Unchanged per Appendix A
                
            elif key in ['angles', 'spherical_angles']:
                labels_s1[key] = value  # Unchanged per Appendix A
                
            else:
                # Pass through unknown labels
                labels_s1[key] = value
        
        return labels_s1

    def _update_labels_s1_to_s2_enhanced(self, labels_s1: Dict, 
                                       scale_factor: float) -> Dict:
        """Enhanced label transformation S1 → S2 with comprehensive scaling"""
        
        labels_s2 = {}
        
        for key, value in labels_s1.items():
            if key in ['CoB', 'CoM', 'center_of_brightness', 'center_of_mass']:
                if isinstance(value, (list, np.ndarray)) and len(value) >= 2:
                    labels_s2[key] = (np.array(value[:2]) * scale_factor).tolist()
                else:
                    labels_s2[key] = value
                    
            elif key in ['delta', 'correction_vector']:
                if isinstance(value, (list, np.ndarray)):
                    labels_s2[key] = (np.array(value) * scale_factor).tolist()
                else:
                    labels_s2[key] = value
                    
            elif key in ['range', 'distance']:
                labels_s2[key] = float(value) * scale_factor if isinstance(value, (int, float)) else value
                
            elif key in ['position', 'xyz_position']:
                if isinstance(value, (list, np.ndarray)):
                    labels_s2[key] = (np.array(value) * scale_factor).tolist()
                else:
                    labels_s2[key] = value
                    
            elif key in ['angles', 'spherical_angles', 'phase_angle']:
                labels_s2[key] = value  # Unchanged per Appendix A
                
            else:
                labels_s2[key] = value
        
        return labels_s2

    def _update_generation_stats(self, blob_strategy: str, padding_mode: str):
        """Update domain randomization statistics"""
        self.generation_stats['total_processed'] += 1
        if blob_strategy in self.generation_stats['blob_distribution']:
            self.generation_stats['blob_distribution'][blob_strategy] += 1
        if padding_mode in self.generation_stats['padding_distribution']:
            self.generation_stats['padding_distribution'][padding_mode] += 1

    def create_balanced_dataset_enhanced(self, 
                                       image_list: List[np.ndarray],
                                       label_list: List[Dict],
                                       balance_factor: int = 3,
                                       strategy_distribution: Dict[str, float] = None) -> Tuple[List[np.ndarray], List[Dict], List[ProcessingParameters]]:
        """Create systematically balanced dataset with controlled domain randomization"""
        
        if strategy_distribution is None:
            strategy_distribution = {
                'random': 0.3,
                'systematic': 0.4,
                'center': 0.15,
                'corner': 0.15
            }
        
        balanced_images = []
        balanced_labels = []
        balanced_params = []
        
        blob_strategies = ['largest', 'n_largest', 'grouped']
        
        for idx, (image, labels) in enumerate(zip(image_list, label_list)):
            # Original image-label pair
            processed_image, processed_labels, params = self.process_image_label_pair_enhanced(
                image, labels, blob_strategy='grouped', domain_randomization_mode='systematic'
            )
            balanced_images.append(processed_image)
            balanced_labels.append(processed_labels)
            balanced_params.append(params)
            
            # Generate additional samples with systematic variation
            for i in range(balance_factor - 1):
                # Systematic strategy selection
                padding_mode = np.random.choice(
                    list(strategy_distribution.keys()),
                    p=list(strategy_distribution.values())
                )
                blob_strategy = blob_strategies[i % len(blob_strategies)]
                
                processed_image, processed_labels, params = self.process_image_label_pair_enhanced(
                    image, labels.copy(), 
                    blob_strategy=blob_strategy,
                    domain_randomization_mode=padding_mode
                )
                
                balanced_images.append(processed_image)
                balanced_labels.append(processed_labels)
                balanced_params.append(params)
        
        self.logger.info(f"Created balanced dataset: {len(balanced_images)} samples")
        self.logger.info(f"Generation statistics: {self.generation_stats}")
        
        return balanced_images, balanced_labels, balanced_params

    def inverse_transform_ml_output(self, ml_output: Dict, 
                                  processing_params: ProcessingParameters) -> Dict:
        """Complete inverse transformation for ML outputs per Appendix A equations"""
        
        physical_output = {}
        
        # Extract processing parameters
        gamma_i = processing_params.gamma_i
        target_size = processing_params.target_size
        scale_factor = processing_params.scale_factor
        
        for key, value in ml_output.items():
            if key in ['CoB', 'CoM', 'center_of_brightness', 'center_of_mass']:
                # Equation A15: Position S2 → S0
                if isinstance(value, (list, np.ndarray)) and len(value) >= 2:
                    s0_value = np.array(value[:2]) * gamma_i / (scale_factor * target_size)
                    physical_output[key] = s0_value.tolist()
                else:
                    physical_output[key] = value
                    
            elif key in ['delta', 'correction_vector']:
                # Delta transformation
                if isinstance(value, (list, np.ndarray)):
                    s0_value = np.array(value) * gamma_i / (scale_factor * target_size)
                    physical_output[key] = s0_value.tolist()
                else:
                    physical_output[key] = value
                    
            elif key in ['range', 'distance']:
                # Equation A18: Range S2 → S0
                if isinstance(value, (int, float)):
                    s0_value = float(value) * gamma_i / (scale_factor * target_size)
                    physical_output[key] = s0_value
                else:
                    physical_output[key] = value
                    
            elif key in ['spherical_coords']:
                # Equation A16: Spherical coordinates transformation
                if isinstance(value, (list, np.ndarray)) and len(value) >= 3:
                    phi1, phi2, rho = value[:3]
                    rho_s0 = rho * gamma_i / (scale_factor * target_size)
                    physical_output[key] = [phi1, phi2, rho_s0]
                else:
                    physical_output[key] = value
                    
            else:
                physical_output[key] = value
        
        return physical_output

    def generate_optical_observables(self, delta_s2: np.ndarray, rho_s2: float,
                                   processing_params: ProcessingParameters,
                                   camera_matrix: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        """Generate optical observables per Appendix A equations A17-A19"""
        
        # Equation A17: Optical observables in S0
        gamma_i = processing_params.gamma_i
        scale_factor = processing_params.scale_factor
        target_size = processing_params.target_size
        
        CoB_s0_u = delta_s2[0] * gamma_i / (scale_factor * target_size)
        CoB_s0_v = delta_s2[1] * gamma_i / (scale_factor * target_size)
        
        o_uv_s0 = np.array([CoB_s0_u, CoB_s0_v, 1.0])
        
        # Equation A18: Range in S0
        rho_s0 = rho_s2 * gamma_i / (scale_factor * target_size)
        
        # Equation A19: Transform to camera frame
        K_inv = np.linalg.inv(camera_matrix)
        o_ImP = K_inv @ o_uv_s0
        position_cam = o_ImP * rho_s0
        
        return o_uv_s0, rho_s0, position_cam

    def save_processing_parameters(self, filepath: str, include_stats: bool = True):
        """Save comprehensive processing parameters for onboard implementation"""
        save_data = {
            'processing_params': self.processing_params.to_dict() if self.processing_params else None,
            'target_size': self.target_size,
            'domain_randomization': self.enable_domain_randomization
        }
        
        if include_stats:
            save_data['generation_stats'] = self.generation_stats
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        self.logger.info(f"Processing parameters saved to: {filepath}")

    def get_generation_statistics(self) -> Dict:
        """Get comprehensive domain randomization statistics"""
        return {
            'total_processed': self.generation_stats['total_processed'],
            'padding_distribution': self.generation_stats['padding_distribution'],
            'blob_distribution': self.generation_stats['blob_distribution'],
            'balance_ratio': self._calculate_balance_ratio()
        }

    def _calculate_balance_ratio(self) -> float:
        """Calculate balance ratio for domain randomization assessment"""
        if self.generation_stats['total_processed'] == 0:
            return 0.0
        
        padding_counts = list(self.generation_stats['padding_distribution'].values())
        if not padding_counts:
            return 0.0
        
        max_count = max(padding_counts)
        min_count = min(padding_counts)
        
        return min_count / max_count if max_count > 0 else 0.0
