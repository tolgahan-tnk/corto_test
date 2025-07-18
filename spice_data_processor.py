"""
spice_data_processor.py
SPICE Data Processing for HRSC Photometric Simulation
Updated with correct kernel list and download functions from SPICE_Data_Preparation_for_SIG.ipynb
"""
"""
Enhanced SPICE Data Processor with HRSC SRC Camera Support
IK kernel dosyasından kamera parametrelerini otomatik olarak çeker
"""

import spiceypy as spice
import numpy as np
import os
import time
import math
from datetime import datetime
import requests
import pathlib
import fnmatch
import re
from typing import Union, Dict, Any
import json

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj

class SpiceDataProcessor:
    """Process SPICE data for photometric HRSC simulations with SRC camera support"""

    def __init__(self, base_path="spice_data"):
        self.base_path = pathlib.Path(base_path)
        self.kernels_loaded = False
        self.loaded_kernels = []
        self.current_time_range = None
        self._remote_spk_list_cache = None
        
        # Cache for camera parameters
        self._camera_params = {}

        # SPICE Kernels
        self.STATIC_KERNELS = [
            "generic_kernels/lsk/naif0012.tls",
            "generic_kernels/pck/pck00010.tpc",
            "MEX/kernels/sclk/MEX_250716_STEP.TSC",
            "MEX/kernels/fk/MEX_V16.TF",
            "MEX/kernels/ik/MEX_HRSC_V09.TI",  # IK kernel for camera parameters
            "MEX/kernels/spk/MEX_STRUCT_V01.BSP",
            "generic_kernels/spk/satellites/mar099.bsp"
        ]

        self.MIRRORS = [
            "https://naif.jpl.nasa.gov/pub/naif",
            "https://spiftp.esac.esa.int/data/SPICE",
        ]

        # Camera ID mappings
        self.CAMERA_IDS = {
            'HRSC_HEAD': -41210,
            'HRSC_SRC': -41220,
            'HRSC_NADIR': -41215,
            'HRSC_RED': -41212,
            'HRSC_GREEN': -41216,
            'HRSC_BLUE': -41214,
            'HRSC_IR': -41218
        }

        # Regex patterns for kernel selection
        self._PRED_RE = re.compile(r"_FDRMCS_", re.IGNORECASE)

    def _download(self, rel_path: str, full_path: pathlib.Path):
        """Downloads a kernel file from a mirror if it doesn't already exist."""
        if full_path.exists() and full_path.stat().st_size > 0:
            return
        
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        for mirror in self.MIRRORS:
            try:
                url = f"{mirror}/{rel_path}"
                print(f"Downloading: {url}")
                r = requests.get(url, timeout=(10, 300))
                r.raise_for_status()
                full_path.write_bytes(r.content)
                return
            except Exception as e:
                print(f"Failed to download from {mirror}: {e}")
                time.sleep(1)
        
        raise RuntimeError(f"Failed to download kernel: {rel_path}")

    def _get_camera_parameters(self, camera_name: str) -> Dict[str, Any]:
        """Extract camera parameters from IK kernel"""
        if camera_name in self._camera_params:
            return self._camera_params[camera_name]
        
        if not self.kernels_loaded:
            print("Warning: Kernels not loaded. Loading static kernels for camera parameters...")
            self._load_static_kernels()
        
        camera_id = self.CAMERA_IDS.get(camera_name)
        if camera_id is None:
            raise ValueError(f"Unknown camera: {camera_name}")
        
        try:
            # Extract parameters from IK kernel
            params = {}
            
            # Focal length
            focal_length = spice.gdpool(f'INS{camera_id}_FOCAL_LENGTH', 0, 1)
            params['focal_length'] = focal_length[0]
            
            # F/ratio
            try:
                f_ratio = spice.gdpool(f'INS{camera_id}_F/RATIO', 0, 1)
                params['f_ratio'] = f_ratio[0]
            except:
                params['f_ratio'] = None
            
            # FOV angular size
            try:
                fov_size = spice.gdpool(f'INS{camera_id}_FOV_ANGULAR_SIZE', 0, 2)
                params['fov_angular_size'] = fov_size  # [cross-track, along-track]
                params['fov_degrees'] = [math.degrees(fov_size[0]), math.degrees(fov_size[1])]
            except:
                params['fov_angular_size'] = None
                params['fov_degrees'] = None
            
            # IFOV
            try:
                ifov = spice.gdpool(f'INS{camera_id}_IFOV', 0, 1)
                params['ifov'] = ifov[0]
                params['ifov_degrees'] = math.degrees(ifov[0])
            except:
                params['ifov'] = None
                params['ifov_degrees'] = None
            
            # Pixel size
            try:
                pixel_size = spice.gdpool(f'INS{camera_id}_PIXEL_SIZE', 0, 2)
                params['pixel_size'] = pixel_size  # [cross-track, along-track] in microns
            except:
                params['pixel_size'] = None
            
            # Detector size
            try:
                pixel_samples = spice.gipool(f'INS{camera_id}_PIXEL_SAMPLES', 0, 1)
                pixel_lines = spice.gipool(f'INS{camera_id}_PIXEL_LINES', 0, 1)
                params['detector_size'] = [pixel_samples[0], pixel_lines[0]]
            except:
                params['detector_size'] = None
            
            # CCD center
            try:
                ccd_center = spice.gdpool(f'INS{camera_id}_CCD_CENTER', 0, 2)
                params['ccd_center'] = ccd_center
            except:
                params['ccd_center'] = None
            
            # Wavelength info
            try:
                band_center = spice.gdpool(f'INS{camera_id}_FILTER_BANDCENTER', 0, 1)
                bandwidth = spice.gdpool(f'INS{camera_id}_FILTER_BANDWIDTH', 0, 1)
                params['wavelength'] = {
                    'band_center': band_center[0],
                    'bandwidth': bandwidth[0]
                }
            except:
                params['wavelength'] = None
            
            # Cache the parameters
            self._camera_params[camera_name] = params
            
            print(f"Extracted camera parameters for {camera_name}:")
            print(f"  Focal Length: {params['focal_length']} mm")
            print(f"  F/ratio: {params['f_ratio']}")
            print(f"  FOV: {params['fov_degrees']} degrees" if params['fov_degrees'] is not None else "  FOV: N/A")
            print(f"  IFOV: {params['ifov_degrees']} degrees/pixel" if params['ifov_degrees'] is not None else "  IFOV: N/A")
            print(f"  Pixel Size: {params['pixel_size']} microns" if params['pixel_size'] is not None else "  Pixel Size: N/A")
            print(f"  Detector Size: {params['detector_size']} pixels" if params['detector_size'] is not None else "  Detector Size: N/A")
            
            return params
            
        except Exception as e:
            print(f"Error extracting camera parameters for {camera_name}: {e}")
            return {}

    def get_hrsc_camera_config(self, camera_type='SRC') -> Dict[str, Any]:
        """
        Get HRSC camera configuration for CORTO
        
        Args:
            camera_type: 'SRC' for Super Resolution Channel, 'HEAD' for main camera
        """
        camera_name = f'HRSC_{camera_type}'
        params = self._get_camera_parameters(camera_name)
        
        if not params:
            # Fallback to default values
            if camera_type == 'SRC':
                return self._get_default_src_config()
            else:
                return self._get_default_hrsc_config()
        
        # Convert to CORTO format
        corto_config = {
            'fov': params['fov_degrees'][0] if params['fov_degrees'] is not None else 0.54,  # Cross-track FOV
            'res_x': params['detector_size'][0] if params['detector_size'] is not None else 1024,
            'res_y': params['detector_size'][1] if params['detector_size'] is not None else 1024,
            'film_exposure': 0.039,  # From photometric parameters
            'sensor': 'BW',  # Black and white for SRC
            'clip_start': 0.1,
            'clip_end': 10000.0,
            'bit_encoding': '16',
            'viewtransform': 'Standard',
            'focal_length_mm': params['focal_length'],
            'f_ratio': params['f_ratio'],
            'pixel_size_microns': params['pixel_size'],
            'ifov_rad': params['ifov'],
            'wavelength_nm': params['wavelength']['band_center'] if params['wavelength'] is not None else None
        }
        
        # Calculate K matrix (camera intrinsics)
        if params['focal_length'] and params['pixel_size'] is not None:
            # Convert focal length from mm to pixels
            focal_length_pixels = params['focal_length'] / (params['pixel_size'][0] / 1000)  # mm to pixels
            
            # Principal point (assume center of detector)
            cx = params['detector_size'][0] / 2 if params['detector_size'] is not None else 512
            cy = params['detector_size'][1] / 2 if params['detector_size'] is not None else 512
            
            # K matrix: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            corto_config['K'] = [
                [focal_length_pixels, 0, cx],
                [0, focal_length_pixels, cy],
                [0, 0, 1]
            ]
        
        return corto_config

    def _get_default_src_config(self) -> Dict[str, Any]:
        """Default SRC camera configuration based on IK kernel values"""
        return {
            'fov': 0.54,  # degrees
            'res_x': 1024,
            'res_y': 1024,
            'film_exposure': 0.039,
            'sensor': 'BW',
            'clip_start': 0.1,
            'clip_end': 10000.0,
            'bit_encoding': '16',
            'viewtransform': 'Standard',
            'focal_length_mm': 984.76,
            'f_ratio': 9.2,
            'pixel_size_microns': [9.0, 9.0],
            'ifov_rad': 0.00000914,
            'K': [
                [1222.0, 0, 512.0],  # fx, 0, cx
                [0, 1222.0, 512.0],  # 0, fy, cy  
                [0, 0, 1]            # 0, 0, 1
            ]
        }

    def _get_default_hrsc_config(self) -> Dict[str, Any]:
        """Default HRSC main camera configuration"""
        return {
            'fov': 11.9,  # degrees
            'res_x': 5184,
            'res_y': 1,
            'film_exposure': 0.039,
            'sensor': 'BW',
            'clip_start': 0.1,
            'clip_end': 10000.0,
            'bit_encoding': '16',
            'viewtransform': 'Standard',
            'focal_length_mm': 175.0,
            'f_ratio': 5.6,
            'pixel_size_microns': [7.0, 7.0],
            'ifov_rad': 0.000040,
            'K': [
                [2500.0, 0, 2592.0],  # fx, 0, cx
                [0, 2500.0, 0.5],     # 0, fy, cy  
                [0, 0, 1]             # 0, 0, 1
            ]
        }

    def _load_static_kernels(self):
        """Load only static kernels for parameter extraction"""
        for kernel in self.STATIC_KERNELS:
            full_path = self.base_path / kernel
            self._download(kernel, full_path)
            if full_path.exists():
                try:
                    spice.furnsh(str(full_path))
                except:
                    print(f"Warning: Could not load kernel {kernel}")

    def _get_remote_file_list(self) -> list[str]:
        """Fetches the list of SPK files from the ESA server and caches it"""
        if self._remote_spk_list_cache is not None:
            return self._remote_spk_list_cache

        spk_dir_url = "https://spiftp.esac.esa.int/data/SPICE/MARS-EXPRESS/kernels/spk/"

        print(f"Fetching remote file list from: {spk_dir_url}")
        try:
            response = requests.get(spk_dir_url, timeout=20)
            response.raise_for_status()
            filenames = re.findall(r'href="([^"]+\.BSP)"', response.text, re.IGNORECASE)
            print(f"Found {len(filenames)} SPK files on the remote server.")
            self._remote_spk_list_cache = filenames
            return filenames
        except requests.exceptions.RequestException as e:
            print(f"WARNING: Could not connect to the remote server. Will use local files only. Error: {e}")
            self._remote_spk_list_cache = []
            return self._remote_spk_list_cache

    def _assert_reconstructed(self, p: pathlib.Path) -> None:
        """Check if kernel is reconstructed (not predicted)"""
        if self._PRED_RE.search(p.name):
            raise ValueError(f"{p.name} is a *predicted* kernel (FDRMCS). Reconstructed ephemeris required.")

    def _select_orbit_kernel(self, year: int, month: int) -> pathlib.Path:
        """Return *one* reconstructed MEX SPK for the given calendar month."""
        root = self.base_path / "MEX/kernels/spk"
        
        # ROB yearly kernels
        if 2005 <= year <= 2011:
            rob = root / f"MEX_ROB_{year%100:02d}0101_{year%100:02d}1231_003.BSP"
            if rob.exists():
                self._assert_reconstructed(rob)
                return rob

        if 2012 <= year <= 2013:
            rob = root / f"MEX_ROB_{year%100:02d}0101_{year%100:02d}1231_001.BSP"
            if rob.exists():
                self._assert_reconstructed(rob)
                return rob

        # Monthly ORMM kernels
        tag = f"{year%100:02d}{month:02d}"
        patterns = [
            f"ORMM_T19_{tag}01000000_*.BSP",
            f"ORMM__{tag}01000000_*.BSP",
        ]

        candidates = []
        
        # First, search locally
        for pat in patterns:
            candidates.extend(root.glob(pat))
            if candidates:
                break

        # Check server if not found locally
        if not candidates:
            print(f"Local kernel not found for {year}-{month:02d}. Checking remote server...")
            remote_files = self._get_remote_file_list()

            for pat in patterns:
                matches_on_server = fnmatch.filter(remote_files, pat)
                if matches_on_server:
                    print(f"Found {len(matches_on_server)} match(es) for pattern '{pat}' on server.")
                    for remote_filename in matches_on_server:
                        rel_path_str = f"MEX/kernels/spk/{remote_filename}"
                        full_path = root / remote_filename
                        self._download(rel_path_str, full_path)
                        if full_path.exists():
                            candidates.append(full_path)
                    print("Stopping search as a suitable kernel type was found and downloaded.")
                    break

        # Filter candidates (remove predicted kernels)
        candidates = [p for p in candidates if not self._PRED_RE.search(p.name)]
        if not candidates:
            raise FileNotFoundError(f"No reconstructed kernel found for {year}-{month:02d} locally or on the remote server.")

        # Select the highest version
        best = max(candidates, key=lambda p: int(p.stem.split("_")[-1]))
        self._assert_reconstructed(best)
        return best

    def _select_attitude_kernel(self, year: int, month: int) -> str:
        """Return the measured attitude CK that covers the given date."""
        yearly_map = {
            2003: "ATNM_MEASURED_030602_040101_V03.BC",
            2004: "ATNM_MEASURED_040101_050101_V03.BC",
            2005: "ATNM_MEASURED_050101_060101_V03.BC",
            2006: "ATNM_MEASURED_060101_070101_V03.BC",
            2007: "ATNM_MEASURED_070101_080101_V03.BC",
            2008: "ATNM_MEASURED_080101_090101_V03.BC",
            2009: "ATNM_MEASURED_090101_100101_V03.BC",
            2010: "ATNM_MEASURED_100101_110101_V03.BC",
            2011: "ATNM_MEASURED_110101_120101_V03.BC",
            2012: "ATNM_MEASURED_2012_V04.BC",
            2013: "ATNM_MEASURED_2013_V04.BC",
            2014: "ATNM_MEASURED_140101_150101_V03.BC",
            2015: "ATNM_MEASURED_2015_V04.BC",
            2016: "ATNM_MEASURED_2016_V04.BC",
            2017: "ATNM_MEASURED_170101_171231_V01.BC",
            2018: "ATNM_MEASURED_180101_181231_V01.BC",
            2019: "ATNM_MEASURED_2019_V03.BC",
            2020: "ATNM_MEASURED_2020_V03.BC",
            2021: "ATNM_MEASURED_2021_V02.BC",
            2022: "ATNM_MEASURED_2022_V01.BC",
            2023: "ATNM_MEASURED_230101_231231_V01.BC",
            2024: "ATNM_MEASURED_240101_250101_V01.BC",
        }
        
        if year in yearly_map:
            return f"MEX/kernels/ck/{yearly_map[year]}"
        
        # 2025 monthly files
        month_map = {
            1: "ATNM_MEASURED_250101_250131_V01.BC",
            2: "ATNM_MEASURED_250201_250228_V01.BC",
            3: "ATNM_MEASURED_250301_250331_V01.BC",
            4: "ATNM_MEASURED_250401_250427_V01.BC",
            5: "ATNM_MEASURED_250511_250531_V01.BC",
            6: "ATNM_MEASURED_250601_250610_V01.BC",
        }
        
        if year == 2025 and month in month_map:
            return f"MEX/kernels/ck/{month_map[month]}"
        
        raise ValueError(f"No CK available for {year}-{month:02d}")

    def _get_required_kernels(self, utc_time):
        """Get list of required kernels for given UTC time"""
        year = int(utc_time[0:4])
        month = int(utc_time[5:7])
        
        required_kernels = self.STATIC_KERNELS.copy()
        
        try:
            orbit_kernel = self._select_orbit_kernel(year, month)
            required_kernels.append(str(orbit_kernel))
            
            attitude_kernel_rel = self._select_attitude_kernel(year, month)
            attitude_kernel_full = self.base_path / attitude_kernel_rel
            required_kernels.append(str(attitude_kernel_full))
            
        except (FileNotFoundError, ValueError) as e:
            print(f"Warning: Could not determine dynamic kernels: {e}")
            
        return required_kernels

    def _kernels_changed(self, utc_time):
        """Check if kernels need to be reloaded for this time"""
        required_kernels = self._get_required_kernels(utc_time)
        
        if not self.kernels_loaded:
            return True
            
        if set(required_kernels) != set(self.loaded_kernels):
            return True
            
        return False

    def load_kernels(self, utc_time):
        """Load required SPICE kernels for given UTC time"""
        if self._kernels_changed(utc_time):
            print(f"Kernels need to be reloaded for time: {utc_time}")
            
            if self.kernels_loaded:
                print("Clearing existing kernels...")
                spice.kclear()
                self.loaded_kernels = []
                self.kernels_loaded = False

            year = int(utc_time[0:4])
            month = int(utc_time[5:7])
            
            print(f"Loading kernels for {year}-{month:02d}...")
            
            # Load static kernels
            for kernel in self.STATIC_KERNELS:
                full_path = self.base_path / kernel
                self._download(kernel, full_path)
                if full_path.exists():
                    spice.furnsh(str(full_path))
                    self.loaded_kernels.append(str(full_path))
                else:
                    print(f"Warning: Could not load kernel {kernel}")

            # Load dynamic kernels
            try:
                orbit_kernel = self._select_orbit_kernel(year, month)
                spice.furnsh(str(orbit_kernel))
                self.loaded_kernels.append(str(orbit_kernel))
                print(f"Loaded orbit kernel: {orbit_kernel}")

                attitude_kernel_rel = self._select_attitude_kernel(year, month)
                attitude_kernel_full = self.base_path / attitude_kernel_rel
                self._download(attitude_kernel_rel, attitude_kernel_full)
                spice.furnsh(str(attitude_kernel_full))
                self.loaded_kernels.append(str(attitude_kernel_full))
                print(f"Loaded attitude kernel: {attitude_kernel_full}")

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load dynamic kernels: {e}")
                print("Continuing with static kernels only...")

            self.kernels_loaded = True
            print(f"SPICE kernels loaded successfully. Total kernels: {len(self.loaded_kernels)}")
        else:
            print(f"Kernels already loaded for time range covering: {utc_time}")

    def get_spice_data(self, utc_time):
        """Get SPICE data for all bodies at given UTC time."""
        self.load_kernels(utc_time)
        
        et = spice.utc2et(utc_time)
        print(f"UTC Time: {utc_time}")
        print(f"Ephemeris Time: {et}")
        spice.boddef('MEX_HRSC_BASE', -41200)
        observer = 'MEX_HRSC_BASE'
        abcorr = 'CN+S'

        results = {}

        try:
            # Phobos
            pos_phobos, lt_phobos = spice.spkpos('PHOBOS', et, 'J2000', abcorr, observer)
            et_phobos = et - lt_phobos
            att_phobos = spice.pxform('IAU_PHOBOS', 'J2000', et_phobos)
            q_phobos = spice.m2q(att_phobos)

            # Mars
            pos_mars, lt_mars = spice.spkpos('MARS', et, 'J2000', abcorr, observer)
            et_mars = et - lt_mars
            att_mars = spice.pxform('IAU_MARS', 'J2000', et_mars)
            q_mars = spice.m2q(att_mars)

            # Deimos
            pos_deimos, lt_deimos = spice.spkpos('DEIMOS', et, 'J2000', abcorr, observer)
            et_deimos = et - lt_deimos
            att_deimos = spice.pxform('IAU_DEIMOS', 'J2000', et_deimos)
            q_deimos = spice.m2q(att_deimos)

            # Sun
            pos_sun, lt_sun = spice.spkpos('SUN', et, 'J2000', abcorr, observer)

            # HRSC Camera attitude
            att_hrsc = spice.pxform('MEX_HRSC_SRC', 'J2000', et)
            q_hrsc = spice.m2q(att_hrsc)

            # Sun lamp orientation calculation
            q_sun_lamp = self._calculate_sun_alignment_quaternion(pos_sun)

            results = {
                'et': et,
                'phobos': {'position': pos_phobos.tolist(), 'quaternion': q_phobos.tolist()},
                'mars':   {'position': pos_mars.tolist(),   'quaternion': q_mars.tolist()},
                'deimos': {'position': pos_deimos.tolist(), 'quaternion': q_deimos.tolist()},
                'sun':    {'position': pos_sun.tolist(),    'quaternion': q_sun_lamp.tolist()},
                'hrsc':   {'position': np.array([0, 0, 0]).tolist(), 'quaternion': q_hrsc.tolist()}
            }

        except Exception as e:
            print(f"SPICE verisi alınırken hata oluştu: {e}")
            results = {
                'et': et,
                'phobos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
                'mars':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
                'deimos': {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
                'sun':    {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]},
                'hrsc':   {'position': [0.0, 0.0, 0.0], 'quaternion': [1.0, 0.0, 0.0, 0.0]}
            }

        return results

    def _calculate_sun_alignment_quaternion(self, pos_sun):
        """Calculate sun lamp orientation quaternion"""
        norm = np.linalg.norm(pos_sun)
        if norm == 0:
            return np.array([1, 0, 0, 0])
            
        unit_vec = pos_sun / norm
        light_direction = -unit_vec
        start_vec = np.array([0.0, 0.0, -1.0])
        
        dot = np.dot(start_vec, light_direction)

        if dot < -0.999999:
            cross_product = np.cross(np.array([1.0, 0.0, 0.0]), start_vec)
            if np.linalg.norm(cross_product) < 0.000001:
                cross_product = np.cross(np.array([0.0, 1.0, 0.0]), start_vec)
            rotation_axis = cross_product / np.linalg.norm(cross_product)
            return np.array([0.0] + list(rotation_axis))

        cross_product = np.cross(start_vec, light_direction)
        w_component = math.sqrt((1.0 + dot) * 2.0)

        w = w_component / 2.0
        x = cross_product[0] / w_component
        y = cross_product[1] / w_component
        z = cross_product[2] / w_component

        quaternion = np.array([w, x, y, z])
        return quaternion / np.linalg.norm(quaternion)

    def create_corto_geometry_data(self, utc_times):
        """Create CORTO-compatible geometry data from multiple UTC times"""
        geometry_data = {
            "sun": {"position": []},
            "camera": {"position": [], "orientation": []},
            "body_1": {"position": [], "orientation": []},
            "body_2": {"position": [], "orientation": []},
            "body_3": {"position": [], "orientation": []}
        }

        for utc_time in utc_times:
            print(f"Processing UTC time: {utc_time}")
            spice_data = self.get_spice_data(utc_time)

            geometry_data["sun"]["position"].append(spice_data["sun"]["position"])
            geometry_data["camera"]["position"].append(spice_data["hrsc"]["position"])
            geometry_data["camera"]["orientation"].append(spice_data["hrsc"]["quaternion"])
            geometry_data["body_1"]["position"].append(spice_data["phobos"]["position"])
            geometry_data["body_1"]["orientation"].append(spice_data["phobos"]["quaternion"])
            geometry_data["body_2"]["position"].append(spice_data["mars"]["position"])
            geometry_data["body_2"]["orientation"].append(spice_data["mars"]["quaternion"])
            geometry_data["body_3"]["position"].append(spice_data["deimos"]["position"])
            geometry_data["body_3"]["orientation"].append(spice_data["deimos"]["quaternion"])

        return geometry_data

    def save_camera_config(self, camera_type='SRC', output_path='camera_config.json'):
        """Save camera configuration to JSON file for CORTO"""
        config = self.get_hrsc_camera_config(camera_type)
        
        # Convert numpy types to JSON-serializable types
        config_safe = convert_numpy_types(config)
        
        with open(output_path, 'w') as f:
            json.dump(config_safe, f, indent=4)
        
        print(f"Camera configuration saved to {output_path}")
        return config

    def cleanup(self):
        """Clean up SPICE kernels"""
        spice.kclear()
        self.kernels_loaded = False
        self.loaded_kernels = []
        self.current_time_range = None
        self._camera_params = {}

# Test the enhanced processor
if __name__ == "__main__":
    processor = SpiceDataProcessor()
    
    try:
        # Test camera parameter extraction
        print("Testing HRSC SRC camera parameter extraction...")
        src_config = processor.get_hrsc_camera_config('SRC')
        print(f"SRC Configuration: {json.dumps(src_config, indent=2)}")
        
        # Save camera config
        processor.save_camera_config('SRC', 'hrsc_src_config.json')
        
        # Test with UTC time
        utc_time = "2019-06-07T10:46:42.8575Z"
        spice_data = processor.get_spice_data(utc_time)
        print(f"✅ Successfully processed SPICE data for: {utc_time}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        processor.cleanup()
