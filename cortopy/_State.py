from __future__ import annotations
from typing import Any, List, Mapping, Optional, Tuple, Union, overload

import json
import numpy as np
import bpy
import os
from pathlib import Path


class State:
    """
    State class: manages scene settings such as environment properties, geometry, and input models.
    """

    # *************************************************************************
    # *     Constructors & Destructors
    # *************************************************************************

    # @overload
    def __init__(
        self,
        scene: str = None,
        geometry: str = None,
        body: Union[str, list] = None,
        scenario: str = None,
        spice_time: str = None,
        kernels: List[str] | None = None,
    ) -> None:
        """
        Constructor for the class STATE defining scene parameters

        Args:
            scene (str): path to json file with scene settings
            geometry (str): path to json file with geometry settings
            body (str or list): path to file with body .blend model
            spice_time (str, optional): time string used to query SPICE kernels
            kernels (list, optional): list of SPICE kernel paths to load
        """
        # Generate a single/multiple nd n_bodies
        if isinstance(body, str):
            self.n_bodies = 1
        elif isinstance(body, list) and len(body)>1:
            self.n_bodies = len(body)
        else:
            TypeError("You have used a list to specify a single body")

        # Get the repository root directory
        corto_path = Path(__file__).resolve().parent.parent
        print(f"Script is running in: {corto_path}")

        scene_file = os.path.join(corto_path, "input", scenario, "scene", scene)
        geometry_file = os.path.join(corto_path, "input", scenario, "geometry", geometry)
        # body_file_according to single or multiple bodies
        if self.n_bodies == 1:
            body_file = os.path.join(corto_path, "input", scenario, "body", "shape", body)
        else: #n_bodies>1
            body_file = []
            for ii in range(0,self.n_bodies):
                body_file.append(os.path.join(corto_path, "input", scenario, "body", "shape", body[ii]))

        # Import inputs
        self.import_scene(scene_file)
        self.import_geometry(geometry_file)
        if self.n_bodies == 1:
            self.import_body(body_file)
        else: #n_bodies>1
            for ii in range(0,self.n_bodies):
                self.import_body(body_file[ii])
        self.path = {}
        self.add_path("corto_path", corto_path)
        self.add_path("input_path", os.path.join(corto_path, "input", scenario))
        self.add_path("output_path", os.path.join(corto_path, "output", scenario))

        # SPICE geometry
        self.spice_time = spice_time
        self.kernels = None
        if spice_time and kernels:
            self.kernels = [
                k if os.path.isabs(k) else os.path.join(corto_path, k)
                for k in kernels
            ]
            import spiceypy as spice
            from . import _Spice

            _Spice.load_kernels(self.kernels)
            et = spice.utc2et(self.spice_time)

            def _update_pose(item: dict):
                if item and {'target', 'observer', 'frame'} <= item.keys():
                    pos, quat = _Spice.get_pose(item['target'], item['observer'], item['frame'], et)
                    item['position'] = np.array([pos])
                    item['orientation'] = np.array([quat])

            _update_pose(self.geometry.get('camera'))
            if self.n_bodies == 1:
                _update_pose(self.geometry.get('body'))
            else:
                for ii in range(self.n_bodies):
                    _update_pose(self.geometry.get(f'body_{ii+1}'))
            _update_pose(self.geometry.get('sun'))

    def import_geometry(self, geometry: str) -> None:
        """
        Import scene geometry configuration

        Args:
            geometry (str): path to json file with geometry settings
        """
        try:
            with open(geometry, "r") as json_file:
                self.geometry = json.load(json_file)
            # Sun
            self.geometry["sun"]["position"] = np.array(
                self.geometry["sun"]["position"]
            )
            # Body
            if self.n_bodies ==1:
                self.geometry["body"]["position"] = np.array(
                    self.geometry["body"]["position"]
                )
                self.geometry["body"]["orientation"] = np.array(
                    self.geometry["body"]["orientation"]
                )
            else:
                for ii in range(0,self.n_bodies):
                    body_key = f"body_{ii+1}"    
                    self.geometry[body_key]["position"] = np.array(
                        self.geometry[body_key]["position"]
                    )
                    self.geometry[body_key]["orientation"] = np.array(
                        self.geometry[body_key]["orientation"]
                    )
            # Camera
            self.geometry["camera"]["position"] = np.array(
                self.geometry["camera"]["position"]
            )
            self.geometry["camera"]["orientation"] = np.array(
                self.geometry["camera"]["orientation"]
            )

        except:
            self.geometry = {}
            ValueError('Geometry dictionary is empty')

    def import_scene(self, scene: str) -> None:
        """
        Import scene settings

        Args:
            scene (str): path to json file with scene settings
        """
        try:
            with open(scene, "r") as json_file:
                settings_json = json.load(json_file)
                self.properties_cam = settings_json["camera_settings"]
                self.properties_cam["K"] = eval(self.properties_cam["K"]) # TODO: generalize handling of these type of variables from json
                self.properties_sun = settings_json["sun_settings"]
                if self.n_bodies ==1:
                    self.properties_body = settings_json["body_settings"]
                else:
                    for ii in range(0,self.n_bodies):
                        property_key = f"body_settings_{ii+1}"    
                        setattr(self, f"properties_body_{ii+1}", settings_json[property_key])
                self.properties_rendering = settings_json["rendering_settings"]
        except:
            self.properties_cam = {}
            self.properties_sun = {}
            self.properties_body = {}
            self.properties_rendering = {}

    def import_body(self, body_filepath: str, load_mode: str = "obj") -> None:
        """Import body

        Args:
            body_filepath (str): filepath of the body
            load_mode (str, optional): loading mode flag. Defaults to 'obj'.
        """
        if load_mode == "obj":
            bpy.ops.wm.obj_import(filepath=body_filepath)
            bpy.ops.object.shade_smooth() # Default setup smooth property of the mesh
            print(f"Imported .obj file from {body_filepath}")
        elif load_mode == ".blend":
            print("Not implemented")

    def add_path(self, name: str, path: str) -> None:
        """add path into the State dictionary containing all relevant paths

        Args:
            name (str): name of the key in the dictionary
            path (str): associated path
        """        
        if self.path:
            # Update the dictionary with the new name and path
            self.path.update({name: path})
        else:
            # Initialize the dictionary with the new name and path
            self.path = {name: path}
