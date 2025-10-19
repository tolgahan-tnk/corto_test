#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_compositing.py
Compositing Branch Path Debugger

Blender içinden çalıştırın:
  blender --background --python debug_compositing.py

Bu script compositing node'larının path'lerini kontrol eder.
"""

import bpy
import sys
import os

sys.path.append(os.getcwd())

print("\n" + "="*80)
print("COMPOSITING PATH DEBUG SCRIPT")
print("="*80)

def check_compositor_nodes():
    """Check all compositor output nodes and their paths"""
    
    scene = bpy.context.scene
    
    if not scene.use_nodes:
        print("❌ Compositing not enabled!")
        return False
    
    tree = scene.node_tree
    
    if not tree:
        print("❌ No node tree found!")
        return False
    
    print(f"\n✅ Compositing enabled")
    print(f"Total nodes: {len(tree.nodes)}")
    
    # Find all output file nodes
    output_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
    
    if not output_nodes:
        print("❌ No CompositorNodeOutputFile nodes found!")
        return False
    
    print(f"\n✅ Found {len(output_nodes)} output file nodes:\n")
    
    all_valid = True
    
    for i, node in enumerate(output_nodes, 1):
        print(f"Node {i}: {node.name}")
        print(f"  Base Path: {node.base_path}")
        print(f"  File Slots: {len(node.file_slots)}")
        
        for j, slot in enumerate(node.file_slots):
            slot_path = slot.path
            print(f"    Slot {j}: '{slot_path}'")
            
            # Check for issues
            issues = []
            
            # Check for Windows backslashes
            if '\\' in slot_path:
                issues.append("Contains Windows backslashes (\\)")
            
            # Check for frame numbering
            if '#' not in slot_path:
                issues.append("Missing frame number placeholder (#)")
            
            # Check for proper directory structure
            if '/' not in slot_path and '#' in slot_path:
                issues.append("No directory structure (will save to root)")
            
            if issues:
                print(f"      ⚠️ ISSUES: {', '.join(issues)}")
                all_valid = False
            else:
                print(f"      ✅ Path looks good")
        
        print()
    
    return all_valid


def test_render_output_paths():
    """Test what the actual output paths would be"""
    print("\n" + "="*80)
    print("SIMULATED RENDER OUTPUT PATHS")
    print("="*80)
    
    scene = bpy.context.scene
    
    if not scene.use_nodes:
        print("❌ Compositing not enabled!")
        return
    
    tree = scene.node_tree
    output_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
    
    # Simulate frame 0
    frame = 0
    
    print(f"\nFrame {frame:06d} would create:\n")
    
    for node in output_nodes:
        base = node.base_path
        
        for slot in node.file_slots:
            slot_path = slot.path
            
            # Replace frame numbering
            frame_str = f"{frame:06d}"
            output_path = slot_path.replace("######", frame_str)
            
            # Combine with base path
            if base:
                full_path = os.path.join(base, output_path)
            else:
                full_path = output_path
            
            # Normalize path
            full_path = os.path.normpath(full_path)
            
            print(f"  {full_path}.png")
    
    print()


def fix_compositor_paths():
    """Fix all compositor paths for cross-platform compatibility"""
    print("\n" + "="*80)
    print("FIXING COMPOSITOR PATHS")
    print("="*80)
    
    scene = bpy.context.scene
    
    if not scene.use_nodes:
        print("❌ Compositing not enabled!")
        return
    
    tree = scene.node_tree
    output_nodes = [n for n in tree.nodes if n.bl_idname == "CompositorNodeOutputFile"]
    
    fixed_count = 0
    
    for node in output_nodes:
        print(f"\nFixing node: {node.name}")
        
        for i, slot in enumerate(node.file_slots):
            original_path = slot.path
            fixed_path = original_path
            
            # Fix Windows backslashes
            if '\\' in fixed_path:
                fixed_path = fixed_path.replace('\\', '/')
                print(f"  Slot {i}: Fixed backslashes")
            
            # Ensure frame numbering
            if '#' not in fixed_path:
                if '/' in fixed_path:
                    fixed_path = fixed_path.rstrip('/') + '/######'
                else:
                    fixed_path = '######'
                print(f"  Slot {i}: Added frame numbering")
            
            # Update if changed
            if original_path != fixed_path:
                slot.path = fixed_path
                fixed_count += 1
                print(f"  Slot {i}: '{original_path}' → '{fixed_path}'")
            else:
                print(f"  Slot {i}: No changes needed")
    
    print(f"\n✅ Fixed {fixed_count} path(s)")


def main():
    """Main debug routine"""
    
    print("\n1. Checking compositor node paths...")
    valid = check_compositor_nodes()
    
    print("\n2. Testing render output paths...")
    test_render_output_paths()
    
    if not valid:
        print("\n3. Fixing invalid paths...")
        fix_compositor_paths()
        
        print("\n4. Re-checking after fix...")
        check_compositor_nodes()
        
        print("\n5. Re-testing output paths...")
        test_render_output_paths()
    else:
        print("\n✅ All paths are valid!")
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()