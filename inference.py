import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader




class ChampionMLP(nn.Module):
    def __init__(self, input_size, num_classes, hidden_sizes, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers += [nn.Linear(prev, num_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




class QuickDrawTestDS(Dataset):
    def __init__(self, images):
        
        assert images.ndim == 2 and images.shape[1] == 784, "Expect flattened 28x28=784"
        self.x = images.astype(np.float32) / 255.0  

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])




def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    arch = ckpt.get("arch")
    dropout = ckpt.get("dropout", 0.2)
    state = ckpt["state_dict"]
    used_swa = ckpt.get("used_swa", False)
    return arch, dropout, state, used_swa, ckpt

def run_inference(ckpt_path, test_npz, out_dir, batch_size=512, device_str=None):
    device = torch.device(device_str) if device_str else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    
    npz = np.load(test_npz)
    assert "test_images" in npz.files, "NPZ must contain 'test_images'"
    X_test = npz["test_images"]  
    n_test, in_dim = X_test.shape
    assert in_dim == 784, "Expected flattened 28x28 vectors (784)"
    

    
    arch, dropout, state, used_swa, ckpt_meta = load_checkpoint(ckpt_path, device)
    
    if arch is None:
        raise ValueError("Checkpoint is missing 'arch' list of hidden sizes.")
    num_classes = ckpt_meta.get("num_classes", 15)  

    model = ChampionMLP(input_size=784, num_classes=num_classes, hidden_sizes=arch, dropout=dropout).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    
    ds = QuickDrawTestDS(X_test)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    
    all_preds = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
    all_preds = np.concatenate(all_preds)  

    
    os.makedirs(out_dir, exist_ok=True)
    npy_path = os.path.join(out_dir, "test_predictions.npy")
    csv_path = os.path.join(out_dir, "test_predictions.csv")  
    np.save(npy_path, all_preds)
    np.savetxt(csv_path, all_preds.reshape(1, -1), fmt="%d", delimiter=",")

    
    print(f"[OK] Inference complete.")
    print(f"    Checkpoint: {ckpt_path}")
    print(f"    Test samples: {n_test}")
    print(f"    Used SWA weights: {used_swa}")
    print(f"    Saved: {npy_path}")
    print(f"    Saved: {csv_path}")
    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Champion MLP Inference on QuickDraw Test Set")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to .pt checkpoint (e.g., checkpoints/champion_fulltrain_final.pt)")
    parser.add_argument("--test_npz", type=str, required=True,
                        help="Path to quickdraw_test.npz (contains 'test_images')")
    parser.add_argument("--out_dir", type=str, default="final_leaderboard_run",
                        help="Output directory for predictions")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None, help="e.g., 'cuda', 'cuda:0', or 'cpu'")
    args = parser.parse_args()

    run_inference(
        ckpt_path=args.ckpt_path,
        test_npz=args.test_npz,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        device_str=args.device
    )