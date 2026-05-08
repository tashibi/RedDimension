import os
os.environ['HIP_VISIBLE_DEVICES'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '12.0.1'
os.environ['TORCH_ROCM_AOT_DISABLE'] = '1'
os.environ['HSA_DISABLE_P2P'] = '1'

import torch, cv2, numpy as np, trimesh, gradio as gr
from skimage import measure
from model import ImageEncoder, RedDimensionTransformer

DEVICE = torch.device("cuda")
enc = ImageEncoder().to(DEVICE)
dec = RedDimensionTransformer().to(DEVICE)
enc.load_state_dict(torch.load("./weights/encoder_final.pth", map_location=DEVICE))
dec.load_state_dict(torch.load("./weights/decoder_final.pth", map_location=DEVICE))
enc.eval(); dec.eval()

def generate(img):
    img_t = torch.from_numpy(cv2.resize(img, (224,224))).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.0
    with torch.no_grad():
        latent = enc(img_t); res = 64; g = np.linspace(-1.1, 1.1, res)
        grid = np.stack(np.meshgrid(g, g, g, indexing='ij'), -1).reshape(-1, 3)
        pts = torch.tensor(grid, dtype=torch.float32).to(DEVICE).unsqueeze(0)
        sdf = dec(pts, latent).cpu().numpy().reshape(res, res, res)
        v, f, _, _ = measure.marching_cubes(sdf, level=0.0)
        mesh = trimesh.Trimesh(vertices=v * (2.2/(res-1)) - 1.1, faces=f)
        mesh.export("out.obj")
    return "out.obj"

gr.Interface(fn=generate, inputs=gr.Image(), outputs=gr.Model3D(), title="RedDimension Gen").launch()
