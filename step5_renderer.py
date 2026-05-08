import os, trimesh, glob, cv2, gc
import numpy as np

os.environ['PYGLET_HEADLESS'] = 'True'
RAW_DIR = "E:/3D_gen_amd/dataset/raw"
IMG_DIR = "E:/3D_gen_amd/dataset/images"
SDF_DIR = "E:/3D_gen_amd/dataset/processed"
os.makedirs(IMG_DIR, exist_ok=True)

def render_stable(path):
    name = os.path.basename(path).replace('.glb', '').replace('.obj', '')
    if os.path.exists(os.path.join(IMG_DIR, f"{name}_view_0.png")): return
    try:
        mesh = trimesh.load(path, force='mesh', skip_materials=True)
        mesh.vertices -= mesh.bounding_box.centroid
        mesh.apply_scale(1.0 / np.max(np.abs(mesh.vertices)))
        scene = mesh.scene()
        for i in range(10):
            angle = (2 * np.pi / 10) * i
            scene.set_camera(angles=(np.deg2rad(-20), angle, 0), distance=2.5)
            png = scene.save_image(resolution=(224, 224))
            with open(os.path.join(IMG_DIR, f"{name}_view_{i}.png"), 'wb') as f: f.write(png)
        del mesh; gc.collect()
        print(f"[OK] Rendered: {name}")
    except: pass

def cleanup():
    for s in glob.glob(os.path.join(SDF_DIR, "*.npz")):
        if not os.path.exists(os.path.join(IMG_DIR, f"{os.path.basename(s).replace('.npz', '')}_view_0.png")):
            os.remove(s)

if __name__ == "__main__":
    for f in glob.glob(os.path.join(RAW_DIR, "**/*.glb"), recursive=True): render_stable(f)
    cleanup()
