"""
Atmosphere helper module for Blender/cortopy scenes.

Creates a volumetric atmosphere shell around a celestial body using an
exponential density profile:

    density(z) = beta0 * exp(-max(r - R_body, 0) / H)

The atmosphere is a UV sphere with a Principled Volume shader.
No surface shader — volume only.

This module is body-agnostic: parameters (radius, color, scale height)
are passed in by the tutorial script, making it reusable for Mars,
Earth, Titan, etc.

Dependencies:
    - bpy (Blender Python API)
    - cortopy (for Shading.create_node / link_nodes helpers)
"""

import bpy
import math


# ================================================================
# PUBLIC API
# ================================================================

def create_atmosphere(
    name: str,
    center_object: str,
    body_radius: float,
    atmosphere_ratio: float = 1.0177,
    beta0: float = 3e-4,
    scale_height: float = 11.0,
    anisotropy: float = 0.6,
    color: tuple = (0.8, 0.5, 0.3),
    segments: int = 512,
    ring_count: int = 256,
):
    """
    Create a volumetric atmosphere around a celestial body.

    The atmosphere sphere is created at scene scale (Approach B):
    radius = body_radius * atmosphere_ratio, with object scale = 1.

    Args:
        name:              Blender object name for the atmosphere
        center_object:     Name of the body mesh to center on (e.g. "Mars_65k_km")
        body_radius:       Body radius in scene BU (= km if OBJ is in km)
        atmosphere_ratio:  R_atm / R_body (default 1.0177 ≈ +60 km for Mars)
        beta0:             Surface density coefficient (tunable)
        scale_height:      Atmospheric scale height in km (default 11 km)
        anisotropy:        Henyey-Greenstein phase function [-1, 1]
        color:             Scattering color RGB tuple (Mars dust default)
        segments:          UV sphere longitude segments
        ring_count:        UV sphere latitude rings

    Returns:
        bpy.types.Object: The atmosphere Blender object
    """

    # --- Compute radii in scene BU ---
    # Scene radius = OBJ_radius * obj_scale
    # But since we create at scene scale directly:
    #   R_body_bu = body_radius  (because after set_scale, 1 BU ≈ 1 km)
    R_body_bu = body_radius
    R_atm_bu = body_radius * atmosphere_ratio

    # --- Step 1: Create UV sphere ---
    atm_obj = _create_sphere(name, center_object, R_atm_bu, segments, ring_count)

    # --- Step 2: Create volume material ---
    material = _create_volume_material(
        name=f"{name}_Material",
        center_obj_name=center_object,
        R_body_bu=R_body_bu,
        H_bu=scale_height,
        beta0=beta0,
        anisotropy=anisotropy,
        color=color,
    )

    # --- Step 3: Assign material ---
    _assign_material(atm_obj, material)

    return atm_obj


# ================================================================
# PRIVATE HELPERS
# ================================================================

def _create_sphere(name, center_obj_name, radius_bu, segments, ring_count):
    """Create a smooth UV sphere centered on the target body."""

    # Get center position from the body object
    center_obj = bpy.data.objects[center_obj_name]
    center_loc = center_obj.location.copy()

    # Create UV sphere at the body's location
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius_bu,
        segments=segments,
        ring_count=ring_count,
        location=center_loc,
    )

    # Rename the newly created object
    atm_obj = bpy.context.active_object
    atm_obj.name = name

    # Smooth shading for clean appearance
    bpy.ops.object.shade_smooth()

    # Parent to center object so it follows position/orientation
    atm_obj.parent = center_obj
    # Clear parent inverse so the sphere stays centered
    atm_obj.matrix_parent_inverse.identity()

    return atm_obj


def _create_volume_material(name, center_obj_name, R_body_bu, H_bu, beta0, anisotropy, color):
    """
    Create a volume-only material with exponential density profile.

    Node graph (13 nodes):
        Texture Coordinate (Object) → Length → Subtract(R_body)
        → Maximum(0) → Divide(H) → Multiply(-1) → Exponent
        → Multiply(beta0) → Principled Volume (Density)
        → Material Output (Volume)
    """

    # Create material
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    tree = mat.node_tree
    nodes = tree.nodes
    links = tree.links

    # Clear default nodes
    nodes.clear()

    # --- Node 0: Material Output ---
    n_output = nodes.new('ShaderNodeOutputMaterial')
    n_output.location = (800, 0)
    n_output.name = "Output"

    # --- Node 1: Principled Volume ---
    n_volume = nodes.new('ShaderNodeVolumePrincipled')
    n_volume.location = (500, 0)
    n_volume.name = "Principled Volume"
    n_volume.inputs['Anisotropy'].default_value = anisotropy
    # Set scattering color
    n_volume.inputs['Color'].default_value = (color[0], color[1], color[2], 1.0)

    # Link: Principled Volume → Material Output (Volume)
    links.new(n_volume.outputs['Volume'], n_output.inputs['Volume'])

    # === Geometry-based density computation ===

    # --- Node 2: Texture Coordinate ---
    n_texcoord = nodes.new('ShaderNodeTexCoord')
    n_texcoord.location = (-800, 0)
    n_texcoord.name = "Texture Coordinate"
    # Set object reference to the body (for object-space coordinates)
    n_texcoord.object = bpy.data.objects[center_obj_name]

    # --- Node 3: Vector Math (Length) → r = ||coord|| ---
    n_length = nodes.new('ShaderNodeVectorMath')
    n_length.location = (-600, 0)
    n_length.name = "Length"
    n_length.operation = 'LENGTH'
    links.new(n_texcoord.outputs['Object'], n_length.inputs[0])

    # --- Node 4: Value (R_body) ---
    n_R = nodes.new('ShaderNodeValue')
    n_R.location = (-600, -200)
    n_R.name = "R_body"
    n_R.outputs[0].default_value = R_body_bu

    # --- Node 5: Math (Subtract) → z_raw = r - R_body ---
    n_sub = nodes.new('ShaderNodeMath')
    n_sub.location = (-400, 0)
    n_sub.name = "Subtract"
    n_sub.operation = 'SUBTRACT'
    links.new(n_length.outputs['Value'], n_sub.inputs[0])
    links.new(n_R.outputs[0], n_sub.inputs[1])

    # --- Node 6: Math (Maximum) → z = max(z_raw, 0) ---
    n_max = nodes.new('ShaderNodeMath')
    n_max.location = (-200, 0)
    n_max.name = "Maximum"
    n_max.operation = 'MAXIMUM'
    n_max.inputs[1].default_value = 0.0
    links.new(n_sub.outputs[0], n_max.inputs[0])

    # --- Node 7: Value (H — scale height) ---
    n_H = nodes.new('ShaderNodeValue')
    n_H.location = (-200, -200)
    n_H.name = "H"
    n_H.outputs[0].default_value = H_bu

    # --- Node 8: Math (Divide) → u = z / H ---
    n_div = nodes.new('ShaderNodeMath')
    n_div.location = (0, 0)
    n_div.name = "Divide"
    n_div.operation = 'DIVIDE'
    links.new(n_max.outputs[0], n_div.inputs[0])
    links.new(n_H.outputs[0], n_div.inputs[1])

    # --- Node 9: Math (Multiply) → v = -(z/H) ---
    n_neg = nodes.new('ShaderNodeMath')
    n_neg.location = (150, 0)
    n_neg.name = "Negate"
    n_neg.operation = 'MULTIPLY'
    n_neg.inputs[1].default_value = -1.0
    links.new(n_div.outputs[0], n_neg.inputs[0])

    # --- Node 10: Math (Exponent) → w = exp(-z/H) ---
    n_exp = nodes.new('ShaderNodeMath')
    n_exp.location = (300, 0)
    n_exp.name = "Exponent"
    n_exp.operation = 'EXPONENT'
    links.new(n_neg.outputs[0], n_exp.inputs[0])

    # --- Node 11: Value (beta0) ---
    n_beta = nodes.new('ShaderNodeValue')
    n_beta.location = (150, -200)
    n_beta.name = "beta0"
    n_beta.outputs[0].default_value = beta0

    # --- Node 12: Math (Multiply) → density = beta0 * exp(-z/H) ---
    n_density = nodes.new('ShaderNodeMath')
    n_density.location = (400, -100)
    n_density.name = "Density"
    n_density.operation = 'MULTIPLY'
    links.new(n_exp.outputs[0], n_density.inputs[0])
    links.new(n_beta.outputs[0], n_density.inputs[1])

    # Link: density → Principled Volume: Density (gray socket)
    links.new(n_density.outputs[0], n_volume.inputs['Density'])

    return mat


def _assign_material(obj, material):
    """Assign a material to a Blender object (replacing any existing)."""
    if obj.data.materials:
        obj.data.materials[0] = material
    else:
        obj.data.materials.append(material)
