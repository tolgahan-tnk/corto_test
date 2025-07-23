"""
corto_post_processor.py
CORTO Post-Processing Pipeline Implementation
Based on CORTO paper Figure 12 and Appendix A
"""

import numpy as np
import cv2
import json
from typing import Tuple, Dict, List

class CORTOPostProcessor:
    """
    Post-processing pipeline for CORTO synthetic images
    Implements Figure 12 and Appendix A from CORTO paper
    """

    def __init__(self, target_size: int = 128):
        self.target_size = target_size  # M in the paper
        self.pipeline_params = {}

    def process_image_label_pair(self, image: np.ndarray, labels: Dict,
                                 target_size: int = None) -> Tuple[np.ndarray, Dict]:
        """
        Complete post-processing pipeline: S0 → S1 → S2

        Args:
            image: Input image in S0 space (Nu × Nv)
            labels: Dictionary of labels (CoB, CoM, range, etc.)
            target_size: Final target size (M)

        Returns:
            Processed image and updated labels
        """
        if target_size is None:
            target_size = self.target_size

        # S0 → S1: Binarization, blob analysis, random padding
        image_s1, labels_s1, transform_s1 = self._transform_s0_to_s1(image, labels)

        # S1 → S2: Resize to target size
        image_s2, labels_s2, transform_s2 = self._transform_s1_to_s2(
            image_s1, labels_s1, target_size
        )

        # Store transformation parameters for inverse transformation
        self.pipeline_params = {
            'transform_s1': transform_s1,
            'transform_s2': transform_s2,
            'original_size': image.shape[:2]
        }

        return image_s2, labels_s2

    def _transform_s0_to_s1(self, image: np.ndarray, labels: Dict) -> Tuple[np.ndarray, Dict, Dict]:
        """
        S0 → S1 transformation: Binarization, blob analysis, random padding
        Enhanced with N-th largest blob selection as per CORTO paper
        """
        # 1. Binarization
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Threshold for body detection
        _, binary = cv2.threshold(gray, 0.1, 1.0, cv2.THRESH_BINARY)

        # 2. Enhanced blob analysis - find N-th biggest blob (CORTO paper enhancement)
        bounding_box = self._enhanced_blob_analysis(binary, n_blobs=1)
        
        # Extract bounding box parameters (Γ1, Γ2, Γ3, Γ4)
        Gamma1, Gamma2, Gamma3, Gamma4 = bounding_box

        # 3. Random padding to make square
        # Calculate padding needed
        max_dim = max(Gamma3, Gamma4)

        # Random padding factors (αu, αv)
        alpha_u = np.random.uniform(0, max_dim - Gamma3)
        alpha_v = np.random.uniform(0, max_dim - Gamma4)

        # Target size with padding
        gamma_i = max_dim + int(max(alpha_u, alpha_v))

        # Create padded image
        padded_image = np.zeros((gamma_i, gamma_i, image.shape[2] if len(image.shape) == 3 else 1))

        # Place original content
        start_x = int(alpha_u)
        start_y = int(alpha_v)
        end_x = start_x + Gamma3
        end_y = start_y + Gamma4

        if len(image.shape) == 3:
            padded_image[start_y:end_y, start_x:end_x] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]
        else:
            padded_image[start_y:end_y, start_x:end_x, 0] = image[Gamma2:Gamma2+Gamma4, Gamma1:Gamma1+Gamma3]

        # 4. Update labels (S0 → S1 transformation from Appendix A)
        labels_s1 = self._update_labels_s0_to_s1(labels, Gamma1, Gamma2, alpha_u, alpha_v)

        # Store transformation parameters
        transform_s1 = {
            'Gamma1': Gamma1,
            'Gamma2': Gamma2,
            'Gamma3': Gamma3,
            'Gamma4': Gamma4,
            'alpha_u': alpha_u,
            'alpha_v': alpha_v,
            'gamma_i': gamma_i
        }

        return padded_image, labels_s1, transform_s1

    def _transform_s1_to_s2(self, image_s1: np.ndarray, labels_s1: Dict,
                              target_size: int) -> Tuple[np.ndarray, Dict, Dict]:
        """
        S1 → S2 transformation: Resize to target size
        """
        # Get current size
        current_size = image_s1.shape[0]  # Should be square from S1

        # Resize image
        if len(image_s1.shape) == 3:
            resized = cv2.resize(image_s1, (target_size, target_size))
        else:
            resized = cv2.resize(image_s1[:,:,0], (target_size, target_size))

        # Calculate scale factor
        scale_factor = target_size / current_size

        # Update labels (S1 → S2 transformation from Appendix A)
        labels_s2 = self._update_labels_s1_to_s2(labels_s1, scale_factor)

        # Store transformation parameters
        transform_s2 = {
            'scale_factor': scale_factor,
            'target_size': target_size
        }

        return resized, labels_s2, transform_s2

    def _update_labels_s0_to_s1(self, labels: Dict, Gamma1: float, Gamma2: float,
                                alpha_u: float, alpha_v: float) -> Dict:
        """
        Update labels for S0 → S1 transformation (Appendix A equations A1-A7)
        """
        labels_s1 = {}

        # Translation vector
        translation = np.array([alpha_u, alpha_v]) - np.array([Gamma1, Gamma2])

        # Center of Brightness (CoB) - Equation A1
        if 'CoB' in labels:
            labels_s1['CoB'] = np.array(labels['CoB']) + translation

        # Center of Mass (CoM) - Equation A2
        if 'CoM' in labels:
            labels_s1['CoM'] = np.array(labels['CoM']) + translation

        # Delta (δ) - Equation A3 (unchanged)
        if 'delta' in labels:
            labels_s1['delta'] = labels['delta']

        # Range (ρ) - Equation A4 (unchanged)
        if 'range' in labels:
            labels_s1['range'] = labels['range']

        # Position [X,Y,Z] - Equation A5 (unchanged)
        if 'position' in labels:
            labels_s1['position'] = labels['position']

        # Angles [φ1, φ2] - Equation A6 (unchanged)
        if 'angles' in labels:
            labels_s1['angles'] = labels['angles']

        # Phase angle (Ψ) - Equation A7 (unchanged)
        if 'phase_angle' in labels:
            labels_s1['phase_angle'] = labels['phase_angle']

        return labels_s1

    def _update_labels_s1_to_s2(self, labels_s1: Dict, scale_factor: float) -> Dict:
        """
        Update labels for S1 → S2 transformation (Appendix A equations A8-A14)
        """
        labels_s2 = {}

        # Center of Brightness (CoB) - Equation A8
        if 'CoB' in labels_s1:
            labels_s2['CoB'] = np.array(labels_s1['CoB']) * scale_factor

        # Center of Mass (CoM) - Equation A9
        if 'CoM' in labels_s1:
            labels_s2['CoM'] = np.array(labels_s1['CoM']) * scale_factor

        # Delta (δ) - Equation A10
        if 'delta' in labels_s1:
            labels_s2['delta'] = np.array(labels_s1['delta']) * scale_factor

        # Range (ρ) - Equation A11
        if 'range' in labels_s1:
            labels_s2['range'] = labels_s1['range'] * scale_factor

        # Position [X,Y,Z] - Equation A12
        if 'position' in labels_s1:
            labels_s2['position'] = np.array(labels_s1['position']) * scale_factor

        # Angles [φ1, φ2] - Equation A13 (unchanged)
        if 'angles' in labels_s1:
            labels_s2['angles'] = labels_s1['angles']

        # Phase angle (Ψ) - Equation A14 (unchanged)
        if 'phase_angle' in labels_s1:
            labels_s2['phase_angle'] = labels_s1['phase_angle']

        return labels_s2

    def inverse_transform_s2_to_s0(self, position_s2: np.ndarray) -> np.ndarray:
        """
        Inverse transformation from S2 to S0 (Equation A15)
        """
        if 'transform_s2' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")

        # Get transformation parameters
        scale_factor = self.pipeline_params['transform_s2']['scale_factor']

        # S2 → S0 transformation
        position_s0 = position_s2 / scale_factor

        return position_s0

    def inverse_transform_spherical_s2_to_s0(self, spherical_s2: np.ndarray) -> np.ndarray:
        """
        Inverse transformation for spherical coordinates (Equation A16)
        """
        if 'transform_s2' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")

        # Get transformation parameters
        scale_factor = self.pipeline_params['transform_s2']['scale_factor']

        # Spherical coordinates: [φ1, φ2, ρ]
        phi1, phi2, rho = spherical_s2

        # Transform range
        rho_s0 = rho / scale_factor

        # Angles remain unchanged
        spherical_s0 = np.array([phi1, phi2, rho_s0])

        # Convert to Cartesian if needed
        x = rho_s0 * np.cos(phi2) * np.cos(phi1)
        y = rho_s0 * np.cos(phi2) * np.sin(phi1)
        z = rho_s0 * np.sin(phi2)

        return np.array([x, y, z])

    def generate_optical_observables(self, delta_s2: np.ndarray, rho_s2: float) -> Tuple[np.ndarray, float]:
        """
        Generate optical observables from ML output (Equations A17-A18)
        """
        if 'transform_s1' not in self.pipeline_params:
            raise ValueError("No transformation parameters stored")

        # Get stored parameters
        gamma_i = self.pipeline_params['transform_s1']['gamma_i']
        scale_factor = self.pipeline_params['transform_s2']['scale_factor']

        # Calculate CoB in S0
        CoB_s0_u = delta_s2[0] * gamma_i / (scale_factor * self.target_size)
        CoB_s0_v = delta_s2[1] * gamma_i / (scale_factor * self.target_size)

        # Optical observables vector - Equation A17
        o_uv_s0 = np.array([CoB_s0_u, CoB_s0_v, 1.0])

        # Range in S0 - Equation A18
        rho_s0 = rho_s2 * gamma_i / (scale_factor * self.target_size)

        return o_uv_s0, rho_s0

    def transform_to_camera_frame(self, o_uv_s0: np.ndarray, rho_s0: float,
                                    K_inv: np.ndarray) -> np.ndarray:
        """
        Transform optical observables to camera frame (Equation A19)
        """
        # Transform to image plane coordinates
        o_ImP = K_inv @ o_uv_s0

        # Convert to 3D position using range
        position_cam = o_ImP * rho_s0

        return position_cam

    def create_balanced_dataset(self, image_list: List[np.ndarray],
                                label_list: List[Dict],
                                balance_factor: int = 3) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Create balanced dataset using multiple random padding (Data Augmentation)
        """
        balanced_images = []
        balanced_labels = []

        for image, labels in zip(image_list, label_list):
            # Original image-label pair
            balanced_images.append(image)
            balanced_labels.append(labels)

            # Generate additional samples with different random padding
            for _ in range(balance_factor - 1):
                # Re-process with different random padding
                processed_image, processed_labels = self.process_image_label_pair(
                    image, labels
                )
                balanced_images.append(processed_image)
                balanced_labels.append(processed_labels)

        return balanced_images, balanced_labels

    def save_processing_parameters(self, filepath: str):
        """Save processing parameters for reproducibility"""
        with open(filepath, 'w') as f:
            json.dump(self.pipeline_params, f, indent=4, default=str)

    def load_processing_parameters(self, filepath: str):
        """Load processing parameters"""
        with open(filepath, 'r') as f:
            self.pipeline_params = json.load(f)

    def _get_nth_largest_blob(self, binary_image, n=1):
        """Get N-th largest blob as specified in CORTO paper"""
        contours, _ = cv2.findContours(
            (binary_image * 255).astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) < n:
            return contours[0] if contours else None
        
        # Sort by area and get N-th largest
        contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
        return contours_sorted[n-1]

    def _enhanced_blob_analysis(self, binary_image, n_blobs=1):
        """Enhanced blob analysis following CORTO paper methodology"""
        selected_blobs = []
        
        for i in range(n_blobs):
            blob = self._get_nth_largest_blob(binary_image, i+1)
            if blob is not None:
                selected_blobs.append(blob)
        
        # Group blobs together
        if selected_blobs:
            all_points = np.vstack(selected_blobs)
            x, y, w, h = cv2.boundingRect(all_points)
            return [x, y, w, h]
        
        return [0, 0, binary_image.shape[1], binary_image.shape[0]]

# Example usage for machine learning dataset preparation
def prepare_ml_dataset(image_dir: str, label_dir: str, output_dir: str):
    """
    Prepare ML dataset using CORTO post-processing pipeline
    """
    processor = CORTOPostProcessor(target_size=128)

    # Process all images
    processed_images = []
    processed_labels = []

    # Example processing loop
    for i, (image_path, label_path) in enumerate(zip(image_files, label_files)):
        # Load image and labels
        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            labels = json.load(f)

        # Process through pipeline
        processed_image, processed_label = processor.process_image_label_pair(image, labels)

        processed_images.append(processed_image)
        processed_labels.append(processed_label)

    # Create balanced dataset
    balanced_images, balanced_labels = processor.create_balanced_dataset(
        processed_images, processed_labels, balance_factor=3
    )

    # Save processed dataset
    np.savez(os.path.join(output_dir, 'processed_dataset.npz'),
             images=balanced_images, labels=balanced_labels)

    # Save processing parameters
    processor.save_processing_parameters(
        os.path.join(output_dir, 'processing_params.json')
    )

    print(f"Processed {len(balanced_images)} image-label pairs")
    return balanced_images, balanced_labels

# Validation example
def validate_processing_pipeline():
    """Validate the post-processing pipeline"""
    # Create synthetic test data
    test_image = np.random.random((512, 512, 3))
    test_labels = {
        'CoB': [256, 256],
        'CoM': [260, 260],
        'range': 100.0,
        'position': [10, 20, 30],
        'angles': [0.5, 0.3],
        'phase_angle': 0.8
    }

    processor = CORTOPostProcessor(target_size=128)

    # Process image
    processed_image, processed_labels = processor.process_image_label_pair(
        test_image, test_labels
    )

    # Test inverse transformation
    test_position_s2 = np.array([64, 64, 50])
    position_s0 = processor.inverse_transform_s2_to_s0(test_position_s2)

    print(f"Original image shape: {test_image.shape}")
    print(f"Processed image shape: {processed_image.shape}")
    print(f"Original CoB: {test_labels['CoB']}")
    print(f"Processed CoB: {processed_labels['CoB']}")
    print(f"Inverse transform test - S2: {test_position_s2}, S0: {position_s0}")

    return processed_image, processed_labels
