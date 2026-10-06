#!/usr/bin/env python3
"""Fine-tuning harness for FAW severity — runs ONE (config, rep, fold) and writes
results_ft/<config>_rep<r>_fold<f>.json. Group-CV comes from folds.csv (no leakage).

Adaptation: frozen | lora | last4 | full. Ordinal head: frankhall | corn.
Loss = l_ce*CE(+effnum weights,label-smoothing) + l_ord*ordinal + l_dist*expected-MAE.
Prediction: argmax of the 9-way CE head when l_ce>0, else the ordinal rank head.
Optional two-branch: global image feats concatenated with a leaf/whorl crop's feats.
"""
import argparse, os, json, time, random, platform, numpy as np, pandas as pd, torch, torch.nn as nn
import torch.nn.functional as F, timm, cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score
import sys; sys.path.insert(0, os.path.dirname(__file__))
from faw_augment import get_train_transform, get_eval_transform, IMAGENET_MEAN, IMAGENET_STD

from pathlib import Path
DS = str(Path(os.environ.get("FAW_DATA_ROOT", "data")).resolve())
K = 9
BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"

CONFIGS = {
 "FULL_M3": ("full", "frankhall",0.0,1.0,0.0,0.0,False,"ord"),
 "CE_FH": ("lora", "frankhall",0.4,0.6,0.0,0.0,False,"ord"),
 "CE_FH_DIST": ("lora", "frankhall",0.4,0.6,0.1,0.0,False,"ord"),
 # name        adapt    ord        l_ce l_ord l_dist l_bin two   predict
 "baseline": ("frozen","frankhall",0.0, 1.0,  0.0,  0.0,  False,"ord"),
 "A":        ("lora",  "frankhall",0.0, 1.0,  0.0,  0.0,  False,"ord"),
 "B":        ("lora",  "corn",     0.0, 1.0,  0.0,  0.0,  False,"ord"),
 "C":        ("lora",  "corn",     0.6, 0.4,  0.0,  0.0,  False,"ce"),
 "D":        ("last4", "corn",     0.6, 0.4,  0.0,  0.0,  False,"ce"),
 "E":        ("last4", "corn",     0.5, 0.3,  0.2,  0.0,  False,"ce"),
 "F":        ("last4", "corn",     0.5, 0.3,  0.2,  0.0,  True, "ce"),
 # R: dual-head resistance model -> binary (1-4 vs 5-9) + CORN 9-class + CE
 "R":        ("lora",  "corn",     0.2, 0.3,  0.0,  0.5,  False,"ce"),
 # 9-class-MAX candidates (run at high res via --img_size); maximize EXACT 9-class
 "M1":       ("lora",  "corn",     1.0, 0.0,  0.0,  0.0,  False,"ce"),   # pure CE
 "M2":       ("lora",  "corn",     0.7, 0.3,  0.0,  0.0,  False,"ce"),   # CE+CORN
 "M3":       ("lora",  "frankhall",0.0, 1.0,  0.0,  0.0,  False,"ord"),  # ordinal-predict (A-style)
 "M4":       ("lora",  "corn",     0.6, 0.2,  0.2,  0.0,  False,"ce"),   # CE+CORN+distance
 # P: PRODUCT model -> trains ALL heads at once so the app can emit, with confidence:
 #   exact 1-9 severity (softmax cls) + within-1/2 bands (ordinal) + Resistant/Susceptible (binh).
 #   Frank-Hall ordinal drives the point prediction (best exact acc in the M-screen).
 "P":        ("lora",  "frankhall",0.4, 0.6,  0.1,  0.5,  False,"ord"),
}

# ---------------- leaf/whorl crop (fast ExG bounding box) --------------------
def leaf_crop(pil):
    rgb = np.array(pil.convert("RGB")); r,g,b = [rgb[...,i].astype(np.float32) for i in range(3)]
    exg = 2*g - r - b
    m = (exg > np.percentile(exg, 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7,7),np.uint8))
    ys, xs = np.where(m > 0)
    if len(xs) < 50: return pil
    x0,x1,y0,y1 = xs.min(), xs.max(), ys.min(), ys.max()
    pad = int(0.05*max(x1-x0, y1-y0))
    x0=max(0,x0-pad); y0=max(0,y0-pad); x1=min(rgb.shape[1],x1+pad); y1=min(rgb.shape[0],y1+pad)
    return Image.fromarray(rgb[y0:y1, x0:x1])

class FAWSet(Dataset):
    def __init__(self, df, tf, train, two):
        self.p=df["image_path"].tolist(); self.y=(df["injury_score"].values-1).astype(int)
        self.tf=tf; self.train=train; self.two=two
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        im=Image.open(f"{DS}/{self.p[i]}").convert("RGB")
        x=self.tf(im)
        if self.two:
            xc=self.tf(leaf_crop(im)); return x, xc, self.y[i]
        return x, self.y[i]

# ---------------- heads & losses -------------------------------------------
class Net(nn.Module):
    def __init__(self, adapt, two, img_size=224):
        super().__init__()
        self.bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0,
                                    img_size=img_size, dynamic_img_size=True)
        D = self.bb.num_features
        self.two = two; feat = D*2 if two else D
        if adapt == "lora":
            from peft import LoraConfig, get_peft_model
            for p in self.bb.parameters(): p.requires_grad_(False)
            self.bb = get_peft_model(self.bb, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                target_modules=["qkv","proj"]))
        elif adapt == "last4":
            for p in self.bb.parameters(): p.requires_grad_(False)
            for blk in self.bb.blocks[-4:]:
                for p in blk.parameters(): p.requires_grad_(True)
            for p in self.bb.norm.parameters(): p.requires_grad_(True)
        elif adapt == "frozen":
            for p in self.bb.parameters(): p.requires_grad_(False)
        # full: leave all trainable
        self.drop = nn.Dropout(0.2)
        self.cls  = nn.Linear(feat, K)          # 9-way (CE + distance + predict)
        self.rank = nn.Linear(feat, K-1)        # ordinal logits
        self.binh = nn.Linear(feat, 2)          # binary resistant(1-4)/suscept(5-9)
    def feats(self, x, xc):
        f = self.bb(x)
        if self.two: f = torch.cat([f, self.bb(xc)], 1)
        return self.drop(f)
    def forward(self, x, xc=None):
        f = self.feats(x, xc); return self.cls(f), self.rank(f), self.binh(f)

def corn_loss(logits, y):
    loss = torch.tensor(0., device=logits.device); n = 0
    for k in range(K-1):
        mask = torch.ones_like(y, dtype=torch.bool) if k == 0 else (y > k-1)
        if mask.sum() == 0: continue
        loss = loss + F.binary_cross_entropy_with_logits(
            logits[mask, k], (y[mask] > k).float(), reduction="sum")
        n += int(mask.sum())
    return loss / max(n, 1)

def frankhall_loss(logits, y):
    t = torch.stack([(y > k).float() for k in range(K-1)], 1)
    return F.binary_cross_entropy_with_logits(logits, t)


def rank_predict(logits, kind):
    p = torch.sigmoid(logits)
    if kind == "corn":
        cum = torch.cumprod(p, 1); return (cum > 0.5).sum(1)
    # frankhall: reconstruct class probs
    probs = torch.zeros(logits.size(0), K, device=logits.device)
    probs[:,0] = 1 - p[:,0]
    for i in range(1, K-1): probs[:,i] = p[:,i-1] - p[:,i]
    probs[:,-1] = p[:,-1]
    return probs.clamp(min=0).argmax(1)

def metrics(yt, yp):
    d = np.abs(yt - yp)
    return dict(accuracy=float(accuracy_score(yt,yp)),
                macro_f1=float(f1_score(yt,yp,average="macro")),
                qwk=float(cohen_kappa_score(yt,yp,labels=np.arange(K),weights="quadratic")),
                mae=float(mean_absolute_error(yt,yp)),
                within1=float(np.mean(d<=1)), within2=float(np.mean(d<=2)))

def main(a):
    started = time.perf_counter()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    adapt, ordk, l_ce, l_ord, l_dist, l_bin, two, predk = CONFIGS[a.config]
    folds = pd.read_csv(a.folds if a.folds else f"{DS}/folds.csv"); fc = folds[f"fold_rep{a.rep}"]
    tr_df, te_df = folds[fc != a.fold], folds[fc == a.fold]
    assert len(tr_df) and len(te_df), "empty training or test fold"
    assert not set(tr_df.image_path) & set(te_df.image_path), "image leakage"
    if "group" in folds:
        assert not set(tr_df.group) & set(te_df.group), "physical-plant leakage"
    cal_df = None
    if a.calibration:
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=a.seed)
        fit_idx, cal_idx = next(splitter.split(tr_df, groups=tr_df["group"]))
        cal_df = tr_df.iloc[cal_idx].copy()
        tr_df = tr_df.iloc[fit_idx].copy()
        assert not set(tr_df.group) & set(cal_df.group)
        assert not set(te_df.group) & set(cal_df.group)
    ntr = len(tr_df)
    # effective-number class weights + weighted sampler (on train)
    ytr = (tr_df["injury_score"].values - 1)
    cnt = np.bincount(ytr, minlength=K); beta = 0.999
    eff = (1 - beta) / (1 - beta**np.maximum(cnt,1)); eff = eff / eff.sum() * K
    cw = torch.tensor(eff, dtype=torch.float32, device=dev)
    # binary Resistant(0-3)/Susceptible(4-8) class weights (dataset is ~20/80)
    ybin_tr = (ytr >= 4).astype(int); bcnt = np.bincount(ybin_tr, minlength=2)
    bw = (len(ybin_tr) / (2 * np.maximum(bcnt, 1)))
    cw_bin = torch.tensor(bw, dtype=torch.float32, device=dev)
    samp_w = eff[ytr]; sampler = WeightedRandomSampler(samp_w, len(samp_w), replacement=True)

    etf = get_eval_transform(a.img_size)
    ttf = etf if a.no_aug else get_train_transform(a.img_size, bg_remove_p=0.0)
    tl = DataLoader(FAWSet(tr_df, ttf, True, two), batch_size=a.bs, sampler=sampler,
                    num_workers=8, pin_memory=True, drop_last=True)
    vl = DataLoader(FAWSet(te_df, etf, False, two), batch_size=a.bs, shuffle=False,
                    num_workers=8, pin_memory=True)

    if adapt == "frozen": a.lr = max(a.lr, 1e-3)   # head-only needs a higher LR
    net = Net(adapt, two, a.img_size).to(dev)
    params = [p for p in net.parameters() if p.requires_grad]
    ntrain = sum(p.numel() for p in params)
    ntotal = sum(p.numel() for p in net.parameters())
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_started = time.perf_counter()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.05)
    steps = a.epochs * len(tl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps,
                                                pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda")
    ordloss = {"corn":corn_loss, "frankhall":frankhall_loss}[ordk]

    for ep in range(a.epochs):
        net.train()
        for batch in tl:
            if two: x, xc, y = batch; xc = xc.to(dev)
            else:   x, y = batch; xc = None
            x, y = x.to(dev), y.to(dev)
            with torch.amp.autocast("cuda"):
                cl, rl, bl = net(x, xc); loss = 0.
                if l_ce:   loss = loss + l_ce*F.cross_entropy(cl, y, weight=cw, label_smoothing=0.05)
                if l_ord:  loss = loss + l_ord*ordloss(rl, y)
                if l_dist:
                    ev = (F.softmax(cl,1) * torch.arange(K,device=dev)).sum(1)
                    loss = loss + l_dist*F.mse_loss(ev, y.float())
                if l_bin:  loss = loss + l_bin*F.cross_entropy(bl, (y >= 4).long(), weight=cw_bin)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_started
    peak_train_bytes = torch.cuda.max_memory_allocated()
    net.eval()
    temperatures = {}
    cal_probabilities = None
    if cal_df is not None:
        from scipy.optimize import minimize_scalar
        cal_loader = DataLoader(FAWSet(cal_df, etf, False, two), batch_size=a.bs, shuffle=False, num_workers=8)
        records = []; targets = []
        with torch.no_grad(), torch.amp.autocast("cuda"):
            for batch in cal_loader:
                if two: x, xc, y = batch; xc = xc.to(dev)
                else: x, y = batch; xc = None
                cl, rl, bl = net(x.to(dev), xc)
                cumulative = torch.sigmoid(rl.float())
                if ordk == "corn": cumulative = torch.cumprod(cumulative, 1)
                op = torch.cat([1-cumulative[:,:1], cumulative[:,:-1]-cumulative[:,1:], cumulative[:,-1:]], 1).clamp(min=0)
                op = op/op.sum(1,keepdim=True).clamp(min=1e-12)
                records.append((F.softmax(cl.float(),1).cpu().numpy(),op.cpu().numpy(),F.softmax(bl.float(),1).cpu().numpy()))
                targets.append(y.numpy())
        cy = np.concatenate(targets)
        cp, op, bp = [np.concatenate([r[k] for r in records]) for k in range(3)]
        def scaled(prob,t):
            logp=np.log(np.clip(prob,1e-12,1))/t
            exp=np.exp(logp-logp.max(1,keepdims=True))
            return exp/exp.sum(1,keepdims=True)
        def fit_temperature(prob, labels):
            loss=lambda logt: -np.log(np.clip(scaled(prob,np.exp(logt))[np.arange(len(labels)),labels],1e-12,1)).mean()
            fitted=minimize_scalar(loss,bounds=(-3,3),method="bounded")
            return float(np.exp(fitted.x))
        ordinal_binary=np.column_stack([op[:,:4].sum(1),op[:,4:].sum(1)])
        temperatures={"ce":fit_temperature(cp,cy),"ordinal":fit_temperature(op,cy),"binary_head":fit_temperature(bp,(cy>=4).astype(int)),"ordinal_binary":fit_temperature(ordinal_binary,(cy>=4).astype(int))}
        cal_probabilities=pd.DataFrame(dict(image_path=cal_df.image_path.tolist(),group=cal_df.group.tolist(),y_true=cy+1))
        for k in range(9):
            cal_probabilities[f"ordinal_p{k+1}"]=op[:,k]
            cal_probabilities[f"ordinal_cal_p{k+1}"]=scaled(op,temperatures["ordinal"])[:,k]
        cal_probabilities["binary_raw_susceptible"]=bp[:,1]
        cal_probabilities["binary_cal_susceptible"]=scaled(bp,temperatures["binary_head"])[:,1]
        cal_probabilities["ordinal_binary_raw_susceptible"]=ordinal_binary[:,1]
        cal_probabilities["ordinal_binary_cal_susceptible"]=scaled(ordinal_binary,temperatures["ordinal_binary"])[:,1]
    infer_started = time.perf_counter()
    net.eval(); P=[]; Y=[]; BP=[]; CLP=[]; BINP=[]; ORDP=[]
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for batch in vl:
            if two: x, xc, y = batch; xc = xc.to(dev)
            else:   x, y = batch; xc = None
            cl, rl, bl = net(x.to(dev), xc)
            pred = cl.argmax(1) if predk == "ce" else rank_predict(rl, ordk)
            bp = bl.argmax(1) if l_bin else (pred >= 4).long()   # susceptible=1
            P.append(pred.cpu().numpy()); Y.append(y.numpy()); BP.append(bp.cpu().numpy())
            CLP.append(F.softmax(cl.float(),1).cpu().numpy())      # 9-way severity probs
            cumulative = torch.sigmoid(rl.float())
            if ordk == "corn": cumulative = torch.cumprod(cumulative, 1)
            ordinal_p = torch.cat([1-cumulative[:,:1], cumulative[:,:-1]-cumulative[:,1:], cumulative[:,-1:]], 1).clamp(min=0)
            ordinal_p = ordinal_p / ordinal_p.sum(1, keepdim=True).clamp(min=1e-12)
            ORDP.append(ordinal_p.cpu().numpy())
            BINP.append(F.softmax(bl.float(),1).cpu().numpy())     # R/S probs
    torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - infer_started
    ordprob = np.concatenate(ORDP)
    yp = np.concatenate(P); yt = np.concatenate(Y); bpred = np.concatenate(BP)
    clsprob = np.concatenate(CLP); binprob = np.concatenate(BINP)
    ybin = (yt >= 4).astype(int)
    res = dict(calibration_temperatures=temperatures,
               calibration_groups=int(cal_df.group.nunique()) if cal_df is not None else 0,
               calibration_images=int(len(cal_df)) if cal_df is not None else 0,
               seed=a.seed, total_params=int(ntotal), trainable_fraction=float(ntrain/ntotal),
               train_seconds=train_seconds, inference_seconds=infer_seconds,
               inference_images_per_second=float(len(yt)/infer_seconds),
               peak_train_vram_bytes=int(peak_train_bytes), gpu=torch.cuda.get_device_name(),
               torch_version=str(torch.__version__), timm_version=timm.__version__,
               python_version=platform.python_version(),
               fold_manifest=os.path.abspath(a.folds), n_train=int(ntr),
               total_seconds=float(time.perf_counter()-started),
               checkpoint_selection="fixed_final_epoch; no test-based selection",
               config=a.config, rep=a.rep, fold=a.fold, adapt=adapt, ord=ordk,
               l_ce=l_ce, l_ord=l_ord, l_dist=l_dist, l_bin=l_bin, two=two, predict=predk,
               img_size=a.img_size, epochs=a.epochs, no_aug=bool(a.no_aug),
               trainable_params=int(ntrain), n_test=int(len(yt)),
               binary_acc=float(accuracy_score(ybin, bpred)),
               binary_f1=float(f1_score(ybin, bpred, average="macro")),
               **metrics(yt, yp))
    os.makedirs(f"{DS}/results_ft", exist_ok=True)
    json.dump(res, open(f"{DS}/results_ft/{a.config}{a.tag}_rep{a.rep}_fold{a.fold}.json","w"), indent=2)
    # per-sample OOF predictions (1-9 scale, matching frozen oof_predictions.csv) so
    # stats_test can bootstrap QWK at the sample level for the fine-tuned models too.
    os.makedirs(f"{DS}/oof_ft", exist_ok=True)
    pd.DataFrame(dict(config=a.config, rep=a.rep, fold=a.fold,
                      image_path=te_df["image_path"].tolist(),
                      y_true=(yt + 1).astype(int), y_pred=(yp + 1).astype(int))
                 ).to_csv(f"{DS}/oof_ft/{a.config}{a.tag}_rep{a.rep}_fold{a.fold}.csv", index=False)
    if a.dump_probs:
        os.makedirs(f"{DS}/oof_ft_probs", exist_ok=True)
        pp = pd.DataFrame(dict(config=a.config, rep=a.rep, fold=a.fold,
                               image_path=te_df["image_path"].tolist(),
                               y_true=(yt + 1).astype(int), y_pred=(yp + 1).astype(int)))
        for i in range(K):
            pp[f"p{i+1}"] = clsprob[:, i]
            pp[f"ordinal_p{i+1}"] = ordprob[:, i]
        pp["p_resistant"] = binprob[:, 0]; pp["p_susceptible"] = binprob[:, 1]
        if cal_probabilities is not None:
            calibrated_ordinal=scaled(ordprob,temperatures["ordinal"])
            ordinal_binary=np.column_stack([ordprob[:,:4].sum(1),ordprob[:,4:].sum(1)])
            for k in range(9): pp[f"ordinal_cal_p{k+1}"]=calibrated_ordinal[:,k]
            pp["binary_cal_susceptible"]=scaled(binprob,temperatures["binary_head"])[:,1]
            pp["ordinal_binary_raw_susceptible"]=ordinal_binary[:,1]
            pp["ordinal_binary_cal_susceptible"]=scaled(ordinal_binary,temperatures["ordinal_binary"])[:,1]
            cal_probabilities.to_csv(f"{DS}/oof_ft_probs/{a.config}{a.tag}_rep{a.rep}_fold{a.fold}_calibration.csv",index=False)
        pp.to_csv(f"{DS}/oof_ft_probs/{a.config}{a.tag}_rep{a.rep}_fold{a.fold}.csv", index=False)
        print("DUMPED_PROBS", f"{a.config}{a.tag}_rep{a.rep}_fold{a.fold}", flush=True)
    print("RESULT", json.dumps(res), flush=True)
    if a.save_weights:
        save_ckpt(net, a.save_weights, adapt, ordk, two, predk, a.img_size, dict(res))
        print("SAVED_WEIGHTS", a.save_weights, flush=True)


def save_ckpt(net, path, adapt, ordk, two, predk, img_size, meta):
    """Save only the trainable sub-state (LoRA + 3 heads) + inference meta.
    Base ViT weights come from timm pretrained at load time (strict=False)."""
    trainable = {k: v.detach().cpu() for k, v in net.state_dict().items()
                 if k in {n for n, p in net.named_parameters() if p.requires_grad}}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state": trainable,
                "meta": {"backbone": BACKBONE, "adapt": adapt, "ord": ordk, "two": two,
                         "predict": predk, "img_size": img_size, "K": K,
                         "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                         "resistant_scores": [1, 2, 3, 4], "susceptible_scores": [5, 6, 7, 8, 9],
                         "cv_meta": meta}}, path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", required=True, choices=list(CONFIGS))
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no_aug", action="store_true")   # train transform = eval (no augmentation)
    ap.add_argument("--tag", default="")               # suffix for output filename
    ap.add_argument("--folds", default="")              # override folds.csv (cleaned set)
    ap.add_argument("--dump_probs", action="store_true")# write oof_ft_probs/<...>.csv
    ap.add_argument("--save_weights", default="")      # if set, save trainable ckpt here
    main(ap.parse_args())
