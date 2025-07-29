""""
Compact CORTO Post-Processing Pipeline Implementation
Based on CORTO paper Figure 12 and Appendix A - FULL PRECISION VERSION
"""

import numpy as np, cv2, json, os
from typing import Tuple, Dict, List

class CORTOPostProcessor:
    """Compact post-processing pipeline for CORTO synthetic images - FULL PRECISION"""

    def __init__(self, target_size: int = 128):
        self.target_size = target_size
        self.pipeline_params = {}

    def process_image_label_pair(self, image: np.ndarray, labels: Dict, target_size: int = None) -> Tuple[np.ndarray, Dict]:
        """Complete S0 → S1 → S2 pipeline in one function - PRECISION PRESERVED"""
        if target_size is None:
            target_size = self.target_size

        # S0 → S1: Enhanced precision transformation
        image_s1, labels_s1, transform_s1 = self._precision_s0_to_s1_transform(image, labels)
        
        # S1 → S2: Enhanced precision resize
        image_s2, labels_s2, transform_s2 = self._precision_s1_to_s2_transform(image_s1, labels_s1, target_size)

        # Store parameters
        self.pipeline_params = {'transform_s1': transform_s1, 'transform_s2': transform_s2, 'original_size': image.shape[:2]}

        return image_s2, labels_s2



    def process_image_label_pair_no_stretch(self, image: np.ndarray, labels: Dict, target_size: int = None, mask_path: str = None) -> Tuple[np.ndarray, Dict]:
        """Complete S0 → S1 → S2 pipeline WITHOUT min-max stretching - MASK AWARE"""
        if target_size is None:
            target_size = self.target_size

        # ✅ DÜZELTİLDİ: Mask-aware processing
        image_s1, labels_s1, transform_s1 = self._precision_s0_to_s1_transform_no_stretch(image, labels, mask_path)
        image_s2, labels_s2, transform_s2 = self._precision_s1_to_s2_transform(image_s1, labels_s1, target_size)

        self.pipeline_params = {'transform_s1': transform_s1, 'transform_s2': transform_s2, 'original_size': image.shape[:2]}
        return image_s2, labels_s2

    def _precision_s0_to_s1_transform(self, image: np.ndarray, labels: Dict) -> Tuple[np.ndarray, Dict, Dict]:
        """Enhanced S0→S1 with FULL PRECISION blob analysis and label updates"""
        
        # 1. Enhanced binarization with precision
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        _, binary = cv2.threshold(gray, 0.1, 1.0, cv2.THRESH_BINARY)
        
        # 2. PRECISION ENHANCED blob analysis - N-th largest blob selection
        bounding_box = self._enhanced_precision_blob_analysis(binary, n_blobs=1)
        Gamma1, Gamma2, Gamma3, Gamma4 = bounding_box
        
        # 3. PRECISION random padding calculation
        max_dim = max(Gamma3, Gamma4)
        
        # Enhanced random padding with boundary checks
        if max_dim > Gamma3:
            alpha_u = np.random.uniform(0.0, float(max_dim - Gamma3))
        else:
            alpha_u = 0.0
            
        if max_dim > Gamma4:
            alpha_v = np.random.uniform(0.0, float(max_dim - Gamma4))
        else:
            alpha_v = 0.0
        
        # Calculate precise gamma_i
        gamma_i = max_dim + int(np.ceil(max(alpha_u, alpha_v)))
        
        # 4. Create padded image with precision
        if len(image.shape) == 3:
            padded_image = np.zeros((gamma_i, gamma_i, image.shape[2]), dtype=image.dtype)
        else:
            padded_image = np.zeros((gamma_i, gamma_i), dtype=image.dtype)
        
        # Precise placement calculations
        start_x = int(np.round(alpha_u))
        start_y = int(np.round(alpha_v))
        end_x = start_x + Gamma3
        end_y = start_y + Gamma4
        
        # Boundary safety checks
        end_x = min(end_x, gamma_i)
        end_y = min(end_y, gamma_i)
        
        # Place content with precision
        if len(image.shape) == 3:
            padded_image[start_y:end_y, start_x:end_x] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]
        else:
            padded_image[start_y:end_y, start_x:end_x] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]
        
        # 5. PRECISION label updates (Appendix A equations A1-A7)
        labels_s1 = self._precision_update_labels_s0_to_s1(labels, Gamma1, Gamma2, alpha_u, alpha_v)
        
        # Store precise transformation parameters
        transform_s1 = {
            'Gamma1': float(Gamma1), 'Gamma2': float(Gamma2), 
            'Gamma3': float(Gamma3), 'Gamma4': float(Gamma4),
            'alpha_u': float(alpha_u), 'alpha_v': float(alpha_v), 
            'gamma_i': int(gamma_i)
        }
        
        return padded_image, labels_s1, transform_s1

    def _precision_s0_to_s1_transform_no_stretch(self, image: np.ndarray, labels: Dict, mask_path: str = None) -> Tuple[np.ndarray, Dict, Dict]:
        """Enhanced S0→S1 WITHOUT stretching - MASK AWARE"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        # ✅ DÜZELTİLDİ: Mask dosyası varsa kullan
        if mask_path and os.path.exists(mask_path):
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask_img is not None:
                if mask_img.shape != gray.shape:
                    mask_img = cv2.resize(mask_img, (gray.shape[1], gray.shape[0]))
                binary = (mask_img > 0).astype(np.float32)
            else:
                # Fallback to adaptive threshold
                threshold_value = max(10, np.mean(gray) * 0.1)
                _, binary = cv2.threshold(gray, threshold_value, 1.0, cv2.THRESH_BINARY)
        else:
            # ✅ DÜZELTİLDİ: Adaptive threshold
            threshold_value = max(10, np.mean(gray) * 0.1)
            _, binary = cv2.threshold(gray, threshold_value, 1.0, cv2.THRESH_BINARY)
        
        # Geri kalan kod aynı kalıyor...
        bounding_box = self._enhanced_precision_blob_analysis(binary, n_blobs=1)
        Gamma1, Gamma2, Gamma3, Gamma4 = bounding_box
        
        max_dim = max(Gamma3, Gamma4)
        
        if max_dim > Gamma3:
            alpha_u = np.random.uniform(0.0, float(max_dim - Gamma3))
        else:
            alpha_u = 0.0
            
        if max_dim > Gamma4:
            alpha_v = np.random.uniform(0.0, float(max_dim - Gamma4))
        else:
            alpha_v = 0.0
        
        gamma_i = max_dim + int(np.ceil(max(alpha_u, alpha_v)))
        
        if len(image.shape) == 3:
            padded_image = np.zeros((gamma_i, gamma_i, image.shape[2]), dtype=image.dtype)
        else:
            padded_image = np.zeros((gamma_i, gamma_i), dtype=image.dtype)
        
        start_x = int(np.round(alpha_u))
        start_y = int(np.round(alpha_v))
        end_x = start_x + Gamma3
        end_y = start_y + Gamma4
        
        end_x = min(end_x, gamma_i)
        end_y = min(end_y, gamma_i)
        
        if len(image.shape) == 3:
            padded_image[start_y:end_y, start_x:end_x] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]
        else:
            padded_image[start_y:end_y, start_x:end_x] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]
        
        labels_s1 = self._precision_update_labels_s0_to_s1(labels, Gamma1, Gamma2, alpha_u, alpha_v)
        
        transform_s1 = {
            'Gamma1': float(Gamma1), 'Gamma2': float(Gamma2), 
            'Gamma3': float(Gamma3), 'Gamma4': float(Gamma4),
            'alpha_u': float(alpha_u), 'alpha_v': float(alpha_v), 
            'gamma_i': int(gamma_i),
            'no_stretch': True,
            'mask_used': mask_path is not None and os.path.exists(mask_path or "")
        }
        
        return padded_image, labels_s1, transform_s1


    def _precision_s1_to_s2_transform(self, image_s1: np.ndarray, labels_s1: Dict, target_size: int) -> Tuple[np.ndarray, Dict, Dict]:
        """Enhanced S1→S2 with PRECISION scaling"""
        current_size = image_s1.shape[0]
        scale_factor = float(target_size) / float(current_size)  # Precise float division
        
        # Precise resize
        if len(image_s1.shape) == 3:
            resized = cv2.resize(image_s1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            resized = cv2.resize(image_s1, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        
        # PRECISION label updates (Appendix A equations A8-A14)
        labels_s2 = self._precision_update_labels_s1_to_s2(labels_s1, scale_factor)
        
        transform_s2 = {'scale_factor': float(scale_factor), 'target_size': int(target_size)}
        
        return resized, labels_s2, transform_s2

    def _enhanced_precision_blob_analysis(self, binary_image: np.ndarray, n_blobs: int = 1) -> List[int]:
        """Enhanced blob analysis with N-th largest blob selection - FULL PRECISION"""
        contours, _ = cv2.findContours((binary_image * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return [0, 0, binary_image.shape[1], binary_image.shape[0]]
        
        # Sort contours by area (largest first)
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Select N blobs and group them
        selected_blobs = []
        for i in range(min(n_blobs, len(contours_sorted))):
            if cv2.contourArea(contours_sorted[i]) > 0:  # Valid contour check
                selected_blobs.append(contours_sorted[i])
        
        if selected_blobs:
            # Group all selected blobs together for precise bounding rect
            all_points = np.vstack(selected_blobs)
            x, y, w, h = cv2.boundingRect(all_points)
            return [int(x), int(y), int(w), int(h)]
        
        return [0, 0, binary_image.shape[1], binary_image.shape[0]]

    def _precision_update_labels_s0_to_s1(self, labels: Dict, Gamma1: float, Gamma2: float, alpha_u: float, alpha_v: float) -> Dict:
        """PRECISION label updates for S0 → S1 (Appendix A equations A1-A7)"""
        labels_s1 = {}
        
        # Precise translation vector calculation
        translation = np.array([alpha_u, alpha_v], dtype=np.float64) - np.array([Gamma1, Gamma2], dtype=np.float64)
        
        # Center of Brightness (CoB) - Equation A1
        if 'CoB' in labels:
            cob_array = np.array(labels['CoB'], dtype=np.float64)
            labels_s1['CoB'] = (cob_array + translation).tolist()
        
        # Center of Mass (CoM) - Equation A2
        if 'CoM' in labels:
            com_array = np.array(labels['CoM'], dtype=np.float64)
            labels_s1['CoM'] = (com_array + translation).tolist()
        
        # Delta (δ) - Equation A3 (unchanged in S0→S1)
        if 'delta' in labels:
            labels_s1['delta'] = labels['delta']
        
        # Range (ρ) - Equation A4 (unchanged in S0→S1)
        if 'range' in labels:
            labels_s1['range'] = float(labels['range'])
        
        # Position [X,Y,Z] - Equation A5 (unchanged in S0→S1)
        if 'position' in labels:
            labels_s1['position'] = labels['position']
        
        # Angles [φ1, φ2] - Equation A6 (unchanged in S0→S1)
        if 'angles' in labels:
            labels_s1['angles'] = labels['angles']
        
        # Phase angle (Ψ) - Equation A7 (unchanged in S0→S1)
        if 'phase_angle' in labels:
            labels_s1['phase_angle'] = float(labels['phase_angle'])
        
        return labels_s1

    def _precision_update_labels_s1_to_s2(self, labels_s1: Dict, scale_factor: float) -> Dict:
        """PRECISION label updates for S1 → S2 (Appendix A equations A8-A14)"""
        labels_s2 = {}
        
        # Center of Brightness (CoB) - Equation A8
        if 'CoB' in labels_s1:
            cob_array = np.array(labels_s1['CoB'], dtype=np.float64)
            labels_s2['CoB'] = (cob_array * scale_factor).tolist()
        
        # Center of Mass (CoM) - Equation A9
        if 'CoM' in labels_s1:
            com_array = np.array(labels_s1['CoM'], dtype=np.float64)
            labels_s2['CoM'] = (com_array * scale_factor).tolist()
        
        # Delta (δ) - Equation A10
        if 'delta' in labels_s1:
            if isinstance(labels_s1['delta'], (list, np.ndarray)):
                delta_array = np.array(labels_s1['delta'], dtype=np.float64)
                labels_s2['delta'] = (delta_array * scale_factor).tolist()
            else:
                labels_s2['delta'] = float(labels_s1['delta']) * scale_factor
        
        # Range (ρ) - Equation A11
        if 'range' in labels_s1:
            labels_s2['range'] = float(labels_s1['range']) * scale_factor
        
        # Position [X,Y,Z] - Equation A12
        if 'position' in labels_s1:
            if isinstance(labels_s1['position'], (list, np.ndarray)):
                pos_array = np.array(labels_s1['position'], dtype=np.float64)
                labels_s2['position'] = (pos_array * scale_factor).tolist()
            else:
                labels_s2['position'] = float(labels_s1['position']) * scale_factor
        
        # Angles [φ1, φ2] - Equation A13 (unchanged in S1→S2)
        if 'angles' in labels_s1:
            labels_s2['angles'] = labels_s1['angles']
        
        # Phase angle (Ψ) - Equation A14 (unchanged in S1→S2)
        if 'phase_angle' in labels_s1:
            labels_s2['phase_angle'] = float(labels_s1['phase_angle'])
        
        return labels_s2

    def inverse_transform_s2_to_s0(self, position_s2: np.ndarray) -> np.ndarray:
        """Inverse S2→S0 transformation - PRECISION"""
        if 'transform_s2' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")
        
        scale_factor = float(self.pipeline_params['transform_s2']['scale_factor'])
        position_s0 = position_s2.astype(np.float64) / scale_factor
        return position_s0

    def inverse_transform_spherical_s2_to_s0(self, spherical_s2: np.ndarray) -> np.ndarray:
        """Inverse spherical transformation - PRECISION"""
        if 'transform_s2' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")
        
        phi1, phi2, rho = spherical_s2.astype(np.float64)
        scale_factor = float(self.pipeline_params['transform_s2']['scale_factor'])
        rho_s0 = rho / scale_factor
        
        # Precise trigonometric calculations
        x = rho_s0 * np.cos(phi2) * np.cos(phi1)
        y = rho_s0 * np.cos(phi2) * np.sin(phi1)
        z = rho_s0 * np.sin(phi2)
        
        return np.array([x, y, z], dtype=np.float64)

    def generate_optical_observables(self, delta_s2: np.ndarray, rho_s2: float) -> Tuple[np.ndarray, float]:
        """Generate optical observables - PRECISION (Equations A17-A18)"""
        if 'transform_s1' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")
        
        gamma_i = float(self.pipeline_params['transform_s1']['gamma_i'])
        scale_factor = float(self.pipeline_params['transform_s2']['scale_factor'])
        target_size = float(self.target_size)
        
        # Precise calculations
        denominator = scale_factor * target_size
        CoB_s0_u = delta_s2[0] * gamma_i / denominator
        CoB_s0_v = delta_s2[1] * gamma_i / denominator
        
        o_uv_s0 = np.array([CoB_s0_u, CoB_s0_v, 1.0], dtype=np.float64)
        rho_s0 = float(rho_s2) * gamma_i / denominator
        
        return o_uv_s0, rho_s0

    def transform_to_camera_frame(self, o_uv_s0: np.ndarray, rho_s0: float, K_inv: np.ndarray) -> np.ndarray:
        """Transform to camera frame - PRECISION"""
        o_uv_s0 = o_uv_s0.astype(np.float64)
        K_inv = K_inv.astype(np.float64)
        rho_s0 = float(rho_s0)
        
        o_ImP = K_inv @ o_uv_s0
        position_cam = o_ImP * rho_s0
        
        return position_cam

    def create_balanced_dataset(self, image_list: List[np.ndarray], label_list: List[Dict], balance_factor: int = 3) -> Tuple[List[np.ndarray], List[Dict]]:
        """Create balanced dataset with DETERMINISTIC padding"""
        balanced_images, balanced_labels = [], []
        
        for i, (image, labels) in enumerate(zip(image_list, label_list)):
            # Original image-label pair
            balanced_images.append(image)
            balanced_labels.append(labels)
            
            # Generate additional samples with deterministic random padding
            for j in range(balance_factor - 1):
                # Set deterministic seed for reproducible results
                np.random.seed(42 + i * balance_factor + j)
                processed_image, processed_labels = self.process_image_label_pair(image, labels)
                balanced_images.append(processed_image)
                balanced_labels.append(processed_labels)
        
        return balanced_images, balanced_labels

    def save_processing_parameters(self, filepath: str):
        """Save processing parameters with precision"""
        # Convert numpy types to ensure JSON serialization
        serializable_params = self._convert_to_serializable(self.pipeline_params)
        with open(filepath, 'w') as f:
            json.dump(serializable_params, f, indent=4)

    def load_processing_parameters(self, filepath: str):
        """Load processing parameters"""
        with open(filepath, 'r') as f:
            self.pipeline_params = json.load(f)

    def _convert_to_serializable(self, obj):
        """Convert numpy types to serializable types"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        return obj

# Utility functions with PRECISION
def prepare_ml_dataset(image_files: List[str], label_files: List[str], output_dir: str) -> Tuple[List[np.ndarray], List[Dict]]:
    """Prepare ML dataset using PRECISION CORTO pipeline"""
    processor = CORTOPostProcessor(target_size=128)
    processed_images, processed_labels = [], []

    for image_path, label_path in zip(image_files, label_files):
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = json.load(f)
        
        # Set deterministic seed for reproducible dataset
        np.random.seed(42)
        processed_image, processed_label = processor.process_image_label_pair(image, labels)
        processed_images.append(processed_image)
        processed_labels.append(processed_label)

    # Create balanced dataset with deterministic results
    balanced_images, balanced_labels = processor.create_balanced_dataset(processed_images, processed_labels, balance_factor=3)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    np.savez(os.path.join(output_dir, 'processed_dataset.npz'), images=balanced_images, labels=balanced_labels)
    processor.save_processing_parameters(os.path.join(output_dir, 'processing_params.json'))

    print(f"Processed {len(balanced_images)} image-label pairs with PRECISION")
    return balanced_images, balanced_labels

def validate_processing_pipeline():
    """Validate the PRECISION post-processing pipeline"""
    # Set deterministic seed for reproducible validation
    np.random.seed(42)
    
    test_image = np.random.random((512, 512, 3))
    test_labels = {'CoB': [256.0, 256.0], 'CoM': [260.0, 260.0], 'range': 100.0, 'position': [10.0, 20.0, 30.0], 'angles': [0.5, 0.3], 'phase_angle': 0.8}
    
    processor = CORTOPostProcessor(target_size=128)
    processed_image, processed_labels = processor.process_image_label_pair(test_image, test_labels)
    
    test_position_s2 = np.array([64.0, 64.0, 50.0])
    position_s0 = processor.inverse_transform_s2_to_s0(test_position_s2)
    
    print(f"PRECISION VALIDATION:")
    print(f"Original: {test_image.shape}, Processed: {processed_image.shape}")
    print(f"CoB: {test_labels['CoB']} → {processed_labels['CoB']}")
    print(f"Inverse: S2={test_position_s2} → S0={position_s0}")
    
    return processed_image, processed_labels

# For backward compatibility
def create_corto_post_processor(target_size=128):
    """Factory function for creating PRECISION processor"""
    return CORTOPostProcessor(target_size=target_size)
