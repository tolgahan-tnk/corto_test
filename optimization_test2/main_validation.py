"""
main_validation.py
CORTO Validation Framework Main Execution Script
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.getcwd())

# Import the framework
from corto_validation_framework import CORTOValidationFramework

def main():
    """Main execution function"""
    print("🚀 Starting CORTO Validation Framework")
    print(f"Working directory: {os.getcwd()}")
    
    # Check if config file exists
    config_path = 'config.json'
    if not Path(config_path).exists():
        print(f"⚠️ Config file not found: {config_path}")
        print("Creating default config...")
        create_default_config()
    
    try:
        # Initialize framework
        print("\n📋 Initializing CORTO Validation Framework...")
        framework = CORTOValidationFramework(
            config_path=config_path,
            camera_type='SRC'
        )
        
        # Check if PDS data directory exists
        pds_path = Path(framework.config['pds_images_path'])
        if not pds_path.exists():
            print(f"❌ PDS data directory not found: {pds_path}")
            print("Please ensure your PDS IMG files are in the correct directory.")
            print("Update the 'pds_images_path' in config.json if needed.")
            return
        
        # Step 1: Extract UTC database from PDS files
        print(f"\n📊 Extracting UTC database from: {pds_path}")
        utc_db = framework.extract_pds_database()
        
        if utc_db is None or len(utc_db) == 0:
            print("❌ No valid PDS records found. Check your PDS files.")
            return
        
        print(f"✅ Found {len(utc_db)} valid PDS records")
        
        # Step 2: Run batch photometric simulations
        print(f"\n🎬 Running photometric simulations for {len(utc_db)} records...")
        simulation_results = framework.run_photometric_simulation_batch()
        
        if not simulation_results:
            print("❌ No successful simulations. Check your CORTO/Blender setup.")
            return
        
        print(f"✅ Completed {len(simulation_results)} successful simulations")
        
        # Step 3: Run CORTO validation framework
        print(f"\n🔍 Running CORTO validation on {len(simulation_results)} simulations...")
        validation_results = framework.run_corto_validation()
        
        # Step 4: Generate summary report
        print(f"\n📋 Validation completed!")
        if validation_results:
            passed = sum(1 for r in validation_results if r['validation_passed'])
            total = len(validation_results)
            success_rate = (passed / total * 100) if total > 0 else 0
            
            print(f"Results: {passed}/{total} validations passed")
            print(f"Success rate: {success_rate:.1f}%")
            
            # Print detailed statistics
            if validation_results:
                import numpy as np
                ncc_scores = [r['ncc_score'] for r in validation_results]
                nrmse_scores = [r['nrmse_score'] for r in validation_results]
                ssim_scores = [r['ssim_score'] for r in validation_results]
                
                print(f"\n📈 Metric Averages:")
                print(f"  NCC:   {np.mean(ncc_scores):.3f} ± {np.std(ncc_scores):.3f}")
                print(f"  NRMSE: {np.mean(nrmse_scores):.3f} ± {np.std(nrmse_scores):.3f}")
                print(f"  SSIM:  {np.mean(ssim_scores):.3f} ± {np.std(ssim_scores):.3f}")
        
        print("\n🎉 CORTO Validation Framework completed successfully!")
        print(f"📁 Results saved in: {framework.config['validation_output_path']}")
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

def create_default_config():
    """Create default config.json file"""
    import json
    
    default_config = {
        "input_path": "./input/S07_Mars_Phobos_Deimos",
        "output_path": "./output/photometric_validation",
        "spice_data_path": "./spice_kernels",
        "pds_images_path": "./PDS_Data",
        "validation_output_path": "./validation_results",
        "real_images_path": "./real_hrsc_images",
        "body_files": [
            "g_phobos_287m_spc_0000n00000_v002.obj",
            "Mars_65k.obj",
            "g_deimos_162m_spc_0000n00000_v001.obj"
        ],
        "scene_file": "scene_mmx.json",
        "geometry_file": "geometry_mmx.json"
    }
    
    with open('config.json', 'w') as f:
        json.dump(default_config, f, indent=4)
    
    print("✅ Created default config.json")
    print("📝 Please review and update paths as needed")

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✨ Process completed successfully!")
    else:
        print("\n💥 Process failed. Check error messages above.")
        sys.exit(1)