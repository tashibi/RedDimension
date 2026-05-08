import os, glob, torch, trimesh, multiprocessing, gc
import numpy as np
from mesh_to_sdf import sample_sdf_near_surface

RAW_DIR = "E:/3D_gen_amd/dataset/raw"
SDF_DIR = "E:/3D_gen_amd/dataset/processed"
os.makedirs(SDF_DIR, exist_ok=True)

def process_mesh(path):
    try:
        name = os.path.basename(path).replace('.glb', '').replace('.obj', '')
        out_path = os.path.join(SDF_DIR, f"{name}.npz")
        if os.path.exists(out_path): return
        mesh = trimesh.load(path, force='mesh', process=True)
        mesh.vertices -= mesh.bounding_box.centroid
        mesh.apply_scale(0.9 / np.max(np.abs(mesh.vertices)))
        points, sdf = sample_sdf_near_surface(mesh, number_of_points=100000)
        np.savez_compressed(out_path, points=points.astype(np.float16), sdf=sdf.astype(np.float16))
        del mesh, points, sdf
        gc.collect()
        print(f"[OK] {name}")
    except Exception as e: print(f"[ERR] {path}: {e}")

if __name__ == "__main__":
    files = glob.glob(os.path.join(RAW_DIR, "**/*.glb"), recursive=True) + glob.glob(os.path.join(RAW_DIR, "**/*.obj"), recursive=True)
    with multiprocessing.Pool(processes=4, maxtasksperchild=1) as pool:
        pool.map(process_mesh, files)
