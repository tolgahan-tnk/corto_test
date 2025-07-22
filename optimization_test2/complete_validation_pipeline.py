'''
complete_validation_pipeline.py
Complete implementation of CORTO Figure 15 Validation Pipeline
with systematic template generation and progressive filtering
'''

import numpy as np
import cv2
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
from scipy.signal import correlate2d
import logging
from dataclasses import dataclass
from enhanced_corto_post_processor import EnhancedCORTOPostProcessor

@dataclass
class ValidationResult:
    """Comprehensive validation result storage"""
    utc_time: str
    img_filename: str
    template_configs: List[Dict]
    noise_combinations: List[Dict]
    best_ncc: float
    best_nrmse: float
    best_ssim: float
    composite_score: float
    composite_score_normalized: float
    validation_status: str
    processing_pipeline: str
    detailed_metrics: Dict
    template_selection_path: List[int]
    
    def to_dict(self) -> Dict:
        return {
            'utc_time': self.utc_time,
            'img_filename': self.img_filename,
            'template_configs': self.template_configs,
            'noise_combinations': self.noise_combinations,
            'best_ncc': self.best_ncc,
            'best_nrmse': self.best_nrmse,
            'best_ssim': self.best_ssim,
            'composite_score': self.composite_score,
            'composite_score_normalized': self.composite_score_normalized,
            'validation_status': self.validation_status,
            'processing_pipeline': self.processing_pipeline,
            'detailed_metrics': self.detailed_metrics,
            'template_selection_path': self.template_selection_path
        }

class CompleteValidationPipeline:
    """
    Complete CORTO Figure 15 Validation Pipeline Implementation
    
    Features:
    - Systematic template generation with rendering variations
    - Progressive filtering: NCC → NRMSE → SSIM  
    - Noise combinations systematic exploration
    - Template-Real image alignment with same transformations
    - Comprehensive validation metrics
    """
    
    def __init__(self, post_processor: EnhancedCORTOPostProcessor):
        self.post_processor = post_processor
        self.validation_results: List[ValidationResult] = []
        
        # Template generation parameters
        self.template_generation_config = {
            'rendering_variations': {
                'sun_intensity': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                'albedo_variations': [0.05, 0.068, 0.08, 0.1, 0.12, 0.15],
                'brdf_roughness': [0.3, 0.5, 0.7, 0.9, 1.0],
                'gamma_correction': [1.8, 2.0, 2.2, 2.4],
                'film_exposure': [0.025, 0.039, 0.05, 0.065]
            },
            'material_variations': {
                'scattering_functions': ['principled', 'lambert', 'oren_nayar'],
                'surface_properties': ['smooth', 'rough', 'mixed']
            }
        }
        
        # Noise combination parameters per CORTO Table 1
        self.noise_combinations_config = {
            'gaussian_mean': [0.01, 0.09, 0.17, 0.25],
            'gaussian_variance': [1e-5, 1e-4, 1e-3],
            'blur': [0.6, 0.8, 1.0, 1.2],
            'brightness': [1.00, 1.17, 1.33, 1.50]
        }
        
        # Progressive filtering parameters
        self.filtering_config = {
            'ncc_templates': 50,  # N templates for NCC filtering
            'nrmse_selection': 10,  # M best images after NRMSE
            'noise_combinations_per_template': 5,  # J noise combinations
            'ssim_final_selection': 3  # L final best images
        }
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def generate_template_configurations(self, base_config: Dict) -> List[Dict]:
        """Generate systematic template configurations with rendering variations"""
        
        templates = []
        render_vars = self.template_generation_config['rendering_variations']
        
        # Generate all combinations of rendering parameters
        import itertools
        
        # Create parameter combinations
        param_combinations = list(itertools.product(
            render_vars['sun_intensity'],
            render_vars['albedo_variations'], 
            render_vars['brdf_roughness'],
            render_vars['gamma_correction'],
            render_vars['film_exposure']
        ))
        
        # Limit to manageable number of templates
        max_templates = self.filtering_config['ncc_templates']
        if len(param_combinations) > max_templates:
            # Systematic sampling for balanced coverage
            step = len(param_combinations) // max_templates
            param_combinations = param_combinations[::step][:max_templates]
        
        self.logger.info(f"Generating {len(param_combinations)} template configurations")
        
        for i, (sun_int, albedo, roughness, gamma, exposure) in enumerate(param_combinations):
            template_config = base_config.copy()
            template_config.update({
                'template_id': i,
                'sun_intensity': sun_int,
                'albedo': albedo,
                'brdf_roughness': roughness,
                'gamma_correction': gamma,
                'film_exposure': exposure,
                'rendering_seed': i  # For reproducible variations
            })
            templates.append(template_config)
        
        return templates

    def generate_noise_combinations(self) -> List[Dict]:
        """Generate systematic noise combinations per CORTO Table 1"""
        
        noise_config = self.noise_combinations_config
        
        import itertools
        combinations = list(itertools.product(
            noise_config['gaussian_mean'],
            noise_config['gaussian_variance'],
            noise_config['blur'],
            noise_config['brightness']
        ))
        
        noise_combinations = []
        for i, (g_mean, g_var, blur, brightness) in enumerate(combinations):
            noise_combinations.append({
                'combination_id': i,
                'gaussian_mean': g_mean,
                'gaussian_variance': g_var,
                'blur': blur,
                'brightness': brightness
            })
        
        self.logger.info(f"Generated {len(noise_combinations)} noise combinations")
        return noise_combinations

    def apply_noise_combination(self, image: np.ndarray, noise_config: Dict) -> np.ndarray:
        """Apply specific noise combination to image"""
        
        noisy_image = image.copy().astype(np.float32)
        
        # 1. Gaussian noise
        noise = np.random.normal(
            noise_config['gaussian_mean'], 
            np.sqrt(noise_config['gaussian_variance']),
            image.shape
        )
        noisy_image += noise
        
        # 2. Blur
        kernel_size = int(noise_config['blur'] * 2) * 2 + 1  # Ensure odd
        noisy_image = cv2.GaussianBlur(noisy_image, (kernel_size, kernel_size), 0)
        
        # 3. Brightness adjustment
        noisy_image *= noise_config['brightness']
        
        # Clip to valid range
        noisy_image = np.clip(noisy_image, 0, 1)
        
        return noisy_image.astype(image.dtype)

    def compute_normalized_cross_correlation(self, real_img: np.ndarray, 
                                           template_img: np.ndarray) -> Tuple[float, Tuple[int, int]]:
        """Compute NCC with optimal alignment per CORTO Equation 1"""
        
        # Ensure same data type
        real_img = real_img.astype(np.float32)
        template_img = template_img.astype(np.float32)
        
        # Resize template to match real image if necessary
        if template_img.shape != real_img.shape:
            template_img = cv2.resize(template_img, (real_img.shape[1], real_img.shape[0]))
        
        # Compute normalized cross-correlation
        correlation = cv2.matchTemplate(real_img, template_img, cv2.TM_CCOEFF_NORMED)
        
        # Find best match
        _, max_val, _, max_loc = cv2.minMaxLoc(correlation)
        
        return float(max_val), max_loc

    def compute_normalized_rmse(self, real_img: np.ndarray, template_img: np.ndarray) -> float:
        """Compute NRMSE per CORTO Equation 2"""
        
        # Ensure same size
        if template_img.shape != real_img.shape:
            template_img = cv2.resize(template_img, (real_img.shape[1], real_img.shape[0]))
        
        # Convert to same type
        real_img = real_img.astype(np.float64)
        template_img = template_img.astype(np.float64)
        
        # Calculate RMSE
        mse = np.mean((real_img - template_img) ** 2)
        rmse = np.sqrt(mse)
        
        # Normalize by image range
        img_range = real_img.max() - real_img.min()
        nrmse = rmse / img_range if img_range > 0 else 0.0
        
        return float(nrmse)

    def compute_structural_similarity(self, real_img: np.ndarray, 
                                    template_img: np.ndarray) -> float:
        """Compute SSIM per CORTO Equation 3"""
        
        # Ensure same size
        if template_img.shape != real_img.shape:
            template_img = cv2.resize(template_img, (real_img.shape[1], real_img.shape[0]))
        
        # Convert to same type
        real_img = real_img.astype(np.float64)
        template_img = template_img.astype(np.float64)
        
        # Calculate SSIM
        try:
            ssim_score = ssim(
                real_img, template_img,
                data_range=real_img.max() - real_img.min(),
                channel_axis=None if len(real_img.shape) == 2 else -1
            )
        except Exception as e:
            self.logger.warning(f"SSIM calculation failed: {e}")
            ssim_score = 0.0
        
        return float(ssim_score)

    def apply_same_transformation_pipeline(self, real_img: np.ndarray, 
                                         synthetic_img: np.ndarray,
                                         labels: Dict) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
        """Apply SAME S0→S1→S2 transformation to both real and synthetic images"""
        
        self.logger.info("Applying same transformation pipeline to both images")
        
        # Process real image
        if len(real_img.shape) == 2:
            real_rgb = cv2.cvtColor((real_img / real_img.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            real_rgb = real_img
            
        real_processed, real_labels, real_params = self.post_processor.process_image_label_pair_enhanced(
            real_rgb, labels.copy(), domain_randomization_mode='center', blob_strategy='grouped'
        )
        
        # Process synthetic image with SAME parameters
        if len(synthetic_img.shape) == 2:
            synthetic_rgb = cv2.cvtColor(synthetic_img, cv2.COLOR_GRAY2RGB)
        else:
            synthetic_rgb = synthetic_img
            
        synthetic_processed, synthetic_labels, synthetic_params = self.post_processor.process_image_label_pair_enhanced(
            synthetic_rgb, labels.copy(), domain_randomization_mode='center', blob_strategy='grouped'
        )
        
        # Convert back to grayscale for comparison
        if len(real_processed.shape) == 3:
            real_processed = cv2.cvtColor(real_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if len(synthetic_processed.shape) == 3:
            synthetic_processed = cv2.cvtColor(synthetic_processed.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        return real_processed, synthetic_processed, real_labels, synthetic_labels

    def run_complete_validation_pipeline(self, 
                                       real_img_path: str,
                                       template_img_paths: List[str],
                                       base_config: Dict,
                                       utc_time: str,
                                       img_filename: str) -> Optional[ValidationResult]:
        """
        Run complete CORTO Figure 15 validation pipeline
        
        Pipeline Steps:
        1. Generate N template configurations
        2. Apply NCC for initial filtering and cropping
        3. Apply NRMSE to select M best templates  
        4. Apply J noise combinations to M templates
        5. Apply SSIM to select L final best results
        """
        
        try:
            self.logger.info(f"Starting complete validation pipeline for {img_filename}")
            
            # Load real image
            real_img = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
            if real_img is None:
                self.logger.error(f"Could not load real image: {real_img_path}")
                return None
            
            # Step 1: Generate template configurations
            template_configs = self.generate_template_configurations(base_config)
            self.logger.info(f"Generated {len(template_configs)} template configurations")
            
            # Step 2: Load and process template images with NCC filtering
            ncc_results = []
            for i, (template_path, config) in enumerate(zip(template_img_paths, template_configs)):
                template_img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template_img is None:
                    continue
                
                # Apply same transformation pipeline
                labels = {'CoB': [real_img.shape[1]//2, real_img.shape[0]//2], 'range': 1000.0}
                real_proc, template_proc, real_labels, template_labels = \
                    self.apply_same_transformation_pipeline(real_img, template_img, labels)
                
                # Compute NCC
                ncc_score, alignment = self.compute_normalized_cross_correlation(real_proc, template_proc)
                
                ncc_results.append({
                    'template_id': i,
                    'config': config,
                    'ncc_score': ncc_score,
                    'alignment': alignment,
                    'real_processed': real_proc,
                    'template_processed': template_proc,
                    'template_path': template_path
                })
            
            # Sort by NCC and select top N
            ncc_results.sort(key=lambda x: x['ncc_score'], reverse=True)
            top_ncc = ncc_results[:self.filtering_config['ncc_templates']]
            
            self.logger.info(f"NCC filtering: Selected {len(top_ncc)} templates")
            
            # Step 3: NRMSE filtering to select M best
            nrmse_results = []
            for result in top_ncc:
                nrmse_score = self.compute_normalized_rmse(
                    result['real_processed'], result['template_processed']
                )
                result['nrmse_score'] = nrmse_score
                nrmse_results.append(result)
            
            # Sort by NRMSE (lower is better) and select M best
            nrmse_results.sort(key=lambda x: x['nrmse_score'])
            best_nrmse = nrmse_results[:self.filtering_config['nrmse_selection']]
            
            self.logger.info(f"NRMSE filtering: Selected {len(best_nrmse)} templates")
            
            # Step 4: Generate and apply J noise combinations  
            noise_combinations = self.generate_noise_combinations()
            # Limit to manageable number
            max_noise = self.filtering_config['noise_combinations_per_template']
            if len(noise_combinations) > max_noise:
                step = len(noise_combinations) // max_noise
                noise_combinations = noise_combinations[::step][:max_noise]
            
            noisy_results = []
            for template_result in best_nrmse:
                for noise_config in noise_combinations:
                    # Apply noise to template
                    noisy_template = self.apply_noise_combination(
                        template_result['template_processed'].astype(np.float32) / 255.0,
                        noise_config
                    )
                    noisy_template = (noisy_template * 255).astype(np.uint8)
                    
                    # Compute SSIM
                    ssim_score = self.compute_structural_similarity(
                        template_result['real_processed'], noisy_template
                    )
                    
                    noisy_results.append({
                        'template_id': template_result['template_id'],
                        'template_config': template_result['config'],
                        'noise_config': noise_config,
                        'ncc_score': template_result['ncc_score'],
                        'nrmse_score': template_result['nrmse_score'],
                        'ssim_score': ssim_score,
                        'real_processed': template_result['real_processed'],
                        'noisy_template': noisy_template
                    })
            
            # Step 5: SSIM filtering to select L final best
            noisy_results.sort(key=lambda x: x['ssim_score'], reverse=True)
            final_best = noisy_results[:self.filtering_config['ssim_final_selection']]
            
            self.logger.info(f"SSIM filtering: Selected {len(final_best)} final results")
            
            # Select overall best result
            if not final_best:
                self.logger.warning("No valid results from validation pipeline")
                return None
            
            best_result = final_best[0]
            
            # Compute composite scores
            composite_score = (
                best_result['ssim_score'] + 
                best_result['ncc_score'] + 
                (1 - best_result['nrmse_score'])
            ) / 3
            
            # Normalized composite score (NRMSE clamped to [0,1])
            nrmse_norm = min(best_result['nrmse_score'] / 5.0, 1.0)
            composite_score_normalized = (
                best_result['ssim_score'] + 
                best_result['ncc_score'] + 
                (1.0 - nrmse_norm)
            ) / 3
            
            # Create comprehensive validation result
            validation_result = ValidationResult(
                utc_time=utc_time,
                img_filename=img_filename,
                template_configs=[r['config'] for r in top_ncc],
                noise_combinations=noise_combinations,
                best_ncc=best_result['ncc_score'],
                best_nrmse=best_result['nrmse_score'],
                best_ssim=best_result['ssim_score'],
                composite_score=composite_score,
                composite_score_normalized=composite_score_normalized,
                validation_status='SUCCESS' if composite_score > 0.6 else 'LOW_SIMILARITY',
                processing_pipeline='CORTO_Figure_15_Complete',
                detailed_metrics={
                    'ncc_candidates': len(ncc_results),
                    'nrmse_selected': len(best_nrmse),
                    'noise_combinations_tested': len(noisy_results),
                    'final_candidates': len(final_best),
                    'template_selection_metrics': {
                        'ncc_range': [min(r['ncc_score'] for r in ncc_results), 
                                     max(r['ncc_score'] for r in ncc_results)],
                        'nrmse_range': [min(r['nrmse_score'] for r in nrmse_results),
                                       max(r['nrmse_score'] for r in nrmse_results)],
                        'ssim_range': [min(r['ssim_score'] for r in noisy_results),
                                      max(r['ssim_score'] for r in noisy_results)]
                    }
                },
                template_selection_path=[
                    len(ncc_results), len(best_nrmse), len(noisy_results), len(final_best)
                ]
            )
            
            self.validation_results.append(validation_result)
            
            self.logger.info(f"Validation completed: NCC={best_result['ncc_score']:.4f}, "
                           f"NRMSE={best_result['nrmse_score']:.4f}, "
                           f"SSIM={best_result['ssim_score']:.4f}, "
                           f"Composite={composite_score:.4f}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error in complete validation pipeline: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_validation_summary(self) -> Dict:
        """Get comprehensive validation summary"""
        if not self.validation_results:
            return {
                'total_validations': 0,
                'successful_validations': 0,
                'success_rate': 0,
                'average_metrics': {},
                'pipeline_performance': {},
                'template_generation': 'SYSTEMATIC',
                'noise_modeling': 'COMPLETE'
            }
        
        total = len(self.validation_results)
        successful = len([r for r in self.validation_results if r.validation_status == 'SUCCESS'])
        
        # Calculate average metrics
        avg_ncc = np.mean([r.best_ncc for r in self.validation_results])
        avg_nrmse = np.mean([r.best_nrmse for r in self.validation_results])
        avg_ssim = np.mean([r.best_ssim for r in self.validation_results])
        avg_composite = np.mean([r.composite_score for r in self.validation_results])
        avg_composite_norm = np.mean([r.composite_score_normalized for r in self.validation_results])
        
        # Pipeline performance analysis
        avg_template_path = np.mean([r.template_selection_path for r in self.validation_results], axis=0)
        
        return {
            'total_validations': total,
            'successful_validations': successful,
            'success_rate': successful / total,
            'average_metrics': {
                'ncc': avg_ncc,
                'nrmse': avg_nrmse,
                'ssim': avg_ssim,
                'composite_score': avg_composite,
                'composite_score_normalized': avg_composite_norm
            },
            'pipeline_performance': {
                'avg_ncc_candidates': avg_template_path[0],
                'avg_nrmse_selected': avg_template_path[1], 
                'avg_noise_combinations': avg_template_path[2],
                'avg_final_selected': avg_template_path[3],
                'filtering_efficiency': avg_template_path[3] / avg_template_path[0] if avg_template_path[0] > 0 else 0
            },
            'template_generation': 'SYSTEMATIC',
            'noise_modeling': 'COMPLETE',
            'validation_pipeline': 'CORTO_FIGURE_15_COMPLETE'
        }

    def save_validation_results(self, filepath: str):
        """Save comprehensive validation results"""
        results_data = {
            'validation_results': [r.to_dict() for r in self.validation_results],
            'summary': self.get_validation_summary(),
            'configuration': {
                'template_generation': self.template_generation_config,
                'noise_combinations': self.noise_combinations_config,
                'filtering_config': self.filtering_config
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=4, default=str)
        
        self.logger.info(f"Validation results saved to: {filepath}")
