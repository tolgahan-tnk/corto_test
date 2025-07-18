"""
SPICE Data Processing for HRSC Photometric Simulation
Based on paste.txt - Mars Express HRSC data processing
Updated with correct kernel list and download functions from SPICE_Data_Preparation_for_SIG.ipynb
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
from typing import Union

class SpiceDataProcessor:
    """Process SPICE data for photometric HRSC simulations"""

    def __init__(self, base_path="spice_data"):
        self.base_path = pathlib.Path(base_path)
        self.kernels_loaded = False
        self._remote_spk_list_cache = None

        # SPICE Kernels - Updated from SPICE_Data_Preparation_for_SIG.ipynb
        self.STATIC_KERNELS = [
            # --- text kernels ---------------------------------------------------------
            "generic_kernels/lsk/naif0012.tls",         # 1.  LSK  (time system)
            "generic_kernels/pck/pck00010.tpc",         # 2.  PCK  (body constants)
            "MEX/kernels/sclk/MEX_250716_STEP.TSC",     # 3.  SCLK (on-board clock) # It can be updated automatically
            "MEX/kernels/fk/MEX_V16.TF",                # 4.  FK   (reference frames)
            "MEX/kernels/ik/MEX_HRSC_V09.TI",           # 5.  IK   (instrument geometry)
            # --- binary SPKs ----------------------------------------------------------
            "MEX/kernels/spk/MEX_STRUCT_V01.BSP",       # 6b. structure SPK (THIS ONE COMES BEFORE mar099.bsp :) )
            "generic_kernels/spk/satellites/mar099.bsp" # 6a. natural bodies
        ]

        self.MIRRORS = [
            "https://naif.jpl.nasa.gov/pub/naif",
            "https://spiftp.esac.esa.int/data/SPICE",
        ]

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

    def _get_remote_file_list(self) -> list[str]:
        """
        Fetches the list of SPK files from the ESA server and caches it
        to avoid redundant downloads during a single run.
        """
        if self._remote_spk_list_cache is not None:
            return self._remote_spk_list_cache

        # Use the ESA server from the mirrors list
        spk_dir_url = "https://spiftp.esac.esa.int/data/SPICE/MARS-EXPRESS/kernels/spk/"

        print(f"Fetching remote file list from: {spk_dir_url}")
        try:
            response = requests.get(spk_dir_url, timeout=20)
            response.raise_for_status()

            # Find .bsp files in hrefs using a simple regex (case-insensitive)
            filenames = re.findall(r'href="([^"]+\.BSP)"', response.text, re.IGNORECASE)

            print(f"Found {len(filenames)} SPK files on the remote server.")
            self._remote_spk_list_cache = filenames
            return filenames

        except requests.exceptions.RequestException as e:
            print(f"WARNING: Could not connect to the remote server. Will use local files only. Error: {e}")
            self._remote_spk_list_cache = []  # On error, use an empty list to prevent retries
            return self._remote_spk_list_cache

    def _assert_reconstructed(self, p: pathlib.Path) -> None:
        """Check if kernel is reconstructed (not predicted)"""
        if self._PRED_RE.search(p.name):
            raise ValueError(
                f"{p.name} is a *predicted* kernel (FDRMCS). "
                "Reconstructed ephemeris required."
            )

    def _select_orbit_kernel(self, year: int, month: int) -> pathlib.Path:
        """
        Return *one* reconstructed MEX SPK for the given calendar month.
        If the file is not found locally, it checks the ESA server and downloads it.
        """
        root = self.base_path / "MEX/kernels/spk"
        
        # 1) ROB yearly kernels
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

        # 2) Monthly ORMM kernels
        tag = f"{year%100:02d}{month:02d}"  # YYMM
        patterns = [
            f"ORMM_T19_{tag}01000000_*.BSP",      # Preferred modern format
            f"ORMM__{tag}01000000_*.BSP",         # Fallback legacy format
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

            # Search for each pattern in order of preference
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

                    # Stop searching once we find matches for the preferred pattern
                    print("Stopping search as a suitable kernel type was found and downloaded.")
                    break

        # Filter candidates (remove predicted kernels)
        candidates = [p for p in candidates if not self._PRED_RE.search(p.name)]
        if not candidates:
            raise FileNotFoundError(
                f"No reconstructed kernel found for {year}-{month:02d} locally or on the remote server."
            )

        # Select the highest version from the found candidates
        best = max(candidates, key=lambda p: int(p.stem.split("_")[-1]))
        self._assert_reconstructed(best)
        return best

    def _select_attitude_kernel(self, year: int, month: int) -> str:
        """Return the measured attitude CK that covers the given date."""
        
        # Yearly kernels (2003-2024)
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
            1:  "ATNM_MEASURED_250101_250131_V01.BC",
            2:  "ATNM_MEASURED_250201_250228_V01.BC",
            3:  "ATNM_MEASURED_250301_250331_V01.BC",
            4:  "ATNM_MEASURED_250401_250427_V01.BC",
            5:  "ATNM_MEASURED_250511_250531_V01.BC",
            6:  "ATNM_MEASURED_250601_250610_V01.BC",
        }
        
        if year == 2025 and month in month_map:
            return f"MEX/kernels/ck/{month_map[month]}"
        
        raise ValueError(f"No CK available for {year}-{month:02d}")

    def load_kernels(self, utc_time):
        """Load required SPICE kernels for given UTC time"""
        if not self.kernels_loaded:
            spice.kclear()

            # Load static kernels
            for kernel in self.STATIC_KERNELS:
                full_path = self.base_path / kernel
                self._download(kernel, full_path)
                if full_path.exists():
                    spice.furnsh(str(full_path))
                else:
                    print(f"Warning: Could not load kernel {kernel}")

            # Load dynamic kernels based on time
            year = int(utc_time[0:4])
            month = int(utc_time[5:7])

            try:
                # Load orbit kernel
                orbit_kernel = self._select_orbit_kernel(year, month)
                spice.furnsh(str(orbit_kernel))
                print(f"Loaded orbit kernel: {orbit_kernel}")

                # Load attitude kernel
                attitude_kernel_rel = self._select_attitude_kernel(year, month)
                attitude_kernel_full = self.base_path / attitude_kernel_rel
                self._download(attitude_kernel_rel, attitude_kernel_full)
                spice.furnsh(str(attitude_kernel_full))
                print(f"Loaded attitude kernel: {attitude_kernel_full}")

            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load dynamic kernels: {e}")
                print("Continuing with static kernels only...")

            self.kernels_loaded = True
            print("SPICE kernels loaded successfully")

    def extract_utc_from_pds_label(self, pds_file_path):
        """Extract UTC time from PDS label file"""
        with open(pds_file_path, 'r') as f:
            content = f.read()

        # Extract UTC_MEAN_TIME from PDS label
        for line in content.split('\n'):
            if 'UTC_MEAN_TIME' in line:
                # Parse: UTC_MEAN_TIME = 2019-06-07T10:46:42.8575
                utc_time = line.split('=')[1].strip().strip('"')
                return utc_time + 'Z'  # Add Z for UTC designation

        raise ValueError("UTC_MEAN_TIME not found in PDS label")

    def get_spice_data(self, utc_time):
        """
        Get SPICE data for all bodies at given UTC time.
        Returns positions and attitudes in J2000 frame as JSON-serializable lists.
        """
        # Bu kısımlar aynı kalır
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

            # === DEĞİŞİKLİK BURADA ===
            # Tüm NumPy dizilerini .tolist() ile listeye çeviriyoruz.
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
            # Hata durumunda da JSON uyumlu boş veri döndürdüğümüzden emin oluyoruz.
            # NumPy dizilerini yine .tolist() ile listeye çeviriyoruz.
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
        # Normalize sun position vector
        norm = np.linalg.norm(pos_sun)
        if norm == 0:
            return np.array([1, 0, 0, 0])  # Identity quaternion
            
        unit_vec = pos_sun / norm

        # Light direction (Sun → observer)
        light_direction = -unit_vec

        # Blender sun lamp default direction
        start_vec = np.array([0.0, 0.0, -1.0])

        # Calculate quaternion rotation
        dot = np.dot(start_vec, light_direction)

        if dot < -0.999999:  # 180 degree rotation
            cross_product = np.cross(np.array([1.0, 0.0, 0.0]), start_vec)
            if np.linalg.norm(cross_product) < 0.000001:
                cross_product = np.cross(np.array([0.0, 1.0, 0.0]), start_vec)
            rotation_axis = cross_product / np.linalg.norm(cross_product)
            return np.array([0.0] + list(rotation_axis))

        # Standard quaternion calculation
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
            "body_1": {"position": [], "orientation": []},  # Phobos
            "body_2": {"position": [], "orientation": []},  # Mars
            "body_3": {"position": [], "orientation": []}   # Deimos
        }

        for utc_time in utc_times:
            spice_data = self.get_spice_data(utc_time)

            # Sun position
            geometry_data["sun"]["position"].append(spice_data["sun"]["position"].tolist())

            # Camera (HRSC) position and orientation
            geometry_data["camera"]["position"].append(spice_data["hrsc"]["position"].tolist())
            geometry_data["camera"]["orientation"].append(spice_data["hrsc"]["quaternion"].tolist())

            # Bodies
            geometry_data["body_1"]["position"].append(spice_data["phobos"]["position"].tolist())
            geometry_data["body_1"]["orientation"].append(spice_data["phobos"]["quaternion"].tolist())

            geometry_data["body_2"]["position"].append(spice_data["mars"]["position"].tolist())
            geometry_data["body_2"]["orientation"].append(spice_data["mars"]["quaternion"].tolist())

            geometry_data["body_3"]["position"].append(spice_data["deimos"]["position"].tolist())
            geometry_data["body_3"]["orientation"].append(spice_data["deimos"]["quaternion"].tolist())

        return geometry_data

    def cleanup(self):
        """Clean up SPICE kernels"""
        spice.kclear()
        self.kernels_loaded = False

# Usage example
def process_hrsc_observation(pds_label_path, output_geometry_path):
    """Process HRSC observation for CORTO simulation"""
    processor = SpiceDataProcessor()

    try:
        # Extract UTC time from PDS label
        utc_time = processor.extract_utc_from_pds_label(pds_label_path)

        # Get SPICE data
        spice_data = processor.get_spice_data(utc_time)

        # Create CORTO geometry data
        geometry_data = processor.create_corto_geometry_data([utc_time])

        # Save to JSON for CORTO
        import json
        with open(output_geometry_path, 'w') as f:
            json.dump(geometry_data, f, indent=4)

        print(f"Geometry data saved to {output_geometry_path}")
        return geometry_data

    except Exception as e:
        print(f"Error processing HRSC observation: {e}")
        return None

    finally:
        # Clean up SPICE kernels
        processor.cleanup()

# Example usage with default UTC time
if __name__ == "__main__":
    # Example with default UTC time from the reference code
    default_utc = "2019-06-07T10:46:42.8575Z"
    
    processor = SpiceDataProcessor()
    try:
        spice_data = processor.get_spice_data(default_utc)
        print("\nSPICE Data Retrieved Successfully:")
        print(f"Phobos position: {spice_data['phobos']['position']}")
        print(f"Mars position: {spice_data['mars']['position']}")
        print(f"Sun position: {spice_data['sun']['position']}")
        
        # Create geometry data for CORTO
        geometry_data = processor.create_corto_geometry_data([default_utc])
        print(f"\nGeometry data created for {len(geometry_data['sun']['position'])} time points")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        processor.cleanup()