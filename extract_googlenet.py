"""
Paper-faithful feature extraction for PRLVS.

Wang et al. §4.3: "We choose GoogleNet [40] as the feature extractor, which is
consistent with the comparisons. The size of the feature vector is 1024."

That is the 1024-d pool5 output (global average pool, before the classifier).
Videos are downsampled to 2 fps, the standard rate for TVSum/SumMe benchmarks.
"""
import os, sys, csv, time
import numpy as np, cv2, torch
from torchvision.models import googlenet, GoogLeNet_Weights

FPS_OUT = 2.0

def build():
    m = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Identity()          # -> 1024-d pool5
    m.eval()
    return m, GoogLeNet_Weights.IMAGENET1K_V1.transforms()

@torch.no_grad()
def features_for(path, model, tf, batch=64):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / FPS_OUT)))
    # sequential decode, keeping every `step`-th frame: far faster than seeking
    feats, buf, picks, n = [], [], [], 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        if n % step == 0:
            img = torch.from_numpy(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)
            buf.append(tf(img)); picks.append(n)
            if len(buf) == batch:
                feats.append(model(torch.stack(buf)).numpy()); buf = []
        n += 1
    if buf: feats.append(model(torch.stack(buf)).numpy())
    cap.release()
    return (np.concatenate(feats) if feats else np.zeros((0, 1024), np.float32)), picks, total, fps

def main(root, out="features_tvsum.npz"):
    vids = sorted({r[0] for r in csv.reader(
        open(os.path.join(root,"data_ex/data/ydata-tvsum50-anno.tsv")), delimiter="\t")})
    vdir = os.path.join(root, "video_ex/video")
    model, tf = build()
    store, t0 = {}, time.time()
    for i, v in enumerate(vids, 1):
        p = os.path.join(vdir, v + ".mp4")
        if not os.path.exists(p): continue
        f, picks, total, fps = features_for(p, model, tf)
        store[f"{v}__feat"]  = f.astype(np.float32)
        store[f"{v}__picks"] = np.asarray(picks, np.int32)
        store[f"{v}__meta"]  = np.asarray([total, fps], np.float32)
        print(f"  {i:2}/50 {v}  {f.shape[0]:5} frames @2fps  ({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(out, **store)
    print("wrote", out)

if __name__ == "__main__":
    main(sys.argv[1])
