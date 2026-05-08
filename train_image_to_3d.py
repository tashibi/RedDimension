import os
os.environ['HIP_VISIBLE_DEVICES'] = '1'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '12.0.1'
os.environ['TORCH_ROCM_AOT_DISABLE'] = '1'
os.environ['HSA_DISABLE_P2P'] = '1'

import torch, torch.nn.functional as F, bitsandbytes as bnb
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import cv2, glob, numpy as np
from model import ImageEncoder, RedDimensionTransformer

DEVICE = torch.device("cuda")
W_DIR = "E:/3D_gen_amd/weights"

class HeavyDataset(Dataset):
    def __init__(self):
        imgs = glob.glob("E:/3D_gen_amd/dataset/images/*.png")
        self.pairs = [(p, f"E:/3D_gen_amd/dataset/processed/{os.path.basename(p).split('_view')[0]}.npz") for p in imgs]
        self.pairs = [p for p in self.pairs if os.path.exists(p)]

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        img_p, sdf_p = self.pairs[idx]
        img = torch.from_numpy(cv2.resize(cv2.imread(img_p), (224, 224))).permute(2,0,1).float()/255.0
        data = np.load(sdf_p); idxs = np.random.choice(len(data['points']), 4096, replace=False)
        return img, torch.from_numpy(data['points'][idxs]), torch.from_numpy(data['sdf'][idxs]).unsqueeze(1)

def train():
    enc = ImageEncoder().to(DEVICE)
    dec = torch.compile(RedDimensionTransformer().to(DEVICE))
    opt = bnb.optim.Adam8bit(list(enc.parameters()) + list(dec.parameters()), lr=5e-5)
    scaler = GradScaler()
    loader = DataLoader(HeavyDataset(), batch_size=32, shuffle=True, num_workers=6)
    for epoch in range(1000):
        for imgs, pts, sdfs in loader:
            imgs, pts, sdfs = imgs.to(DEVICE), pts.to(DEVICE), sdfs.to(DEVICE)
            with autocast():
                loss = F.huber_loss(dec(pts, enc(imgs)), sdfs, delta=0.01)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        if epoch % 5 == 0:
            torch.save(enc.state_dict(), f"{W_DIR}/encoder_final.pth")
            torch.save(dec.state_dict(), f"{W_DIR}/decoder_final.pth")
        print(f"Epoch {epoch} | Loss: {loss.item():.6f}")

if __name__ == "__main__": train()
