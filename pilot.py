"""
Pilot: does quantization hurt "knowing when it's wrong" more on field images
(PlantDoc) than on lab images (PlantVillage)?

One model (MobileNetV3-Large), one seed. FP32 vs INT8 static PTQ (real integer
kernels where they work, else exact simulation) vs INT4 weights / INT8
activations (simulated). Temperatures fitted on lab validation (source) and on a
small labeled PlantDoc split (field); the rest of PlantDoc is the field test set.
Outputs: metrics table, bootstrap CIs on the shift gap, flip rate by confidence
bin, risk-coverage plot, and a GO / NULL hint per variant.

Usage:
  python pilot.py --pv /data/plantvillage/color --pd /data/PlantDoc-Dataset --out runs/pilot

  --pv  folder with one sub-folder per PlantVillage class
  --pd  PlantDoc root containing train/ and test/ (one sub-folder per class)

Stages are cached in --out and reused only if the settings are unchanged.
"""
import argparse, copy, json, os, random, re, time, warnings
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.models import MobileNet_V3_Large_Weights
from torchvision.models.quantization import mobilenet_v3_large
import torch.ao.quantization as tq
from sklearn.metrics import f1_score, roc_auc_score

# torch.ao eager-mode quantization is deprecated (-> torchao) but still works;
# fine for the pilot. The paper's numbers come from the on-device LiteRT model.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------------------------------------------------------
# 1. Shared label set: PlantDoc folder -> PlantVillage folder.
#    REVIEW THIS before the real paper; it goes in the appendix.
#    Names are matched after normalisation (lowercase, alphanumerics only,
#    "maize"/"including sour" removed), so minor spelling variants across
#    Kaggle/GitHub copies still match.
# ----------------------------------------------------------------------------
PD_TO_PV = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato two spotted spider mites leaf": "Tomato___Spider_mites Two-spotted_spider_mite",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def norm(s):
    s = re.sub(r"[^a-z0-9]", "", s.lower())
    return s.replace("maize", "").replace("includingsour", "")


def list_images(d):
    return sorted(p for p in Path(d).rglob("*") if p.suffix.lower() in IMG_EXT)


def build_index(pv_root, pd_root, max_per_class, seed):
    pv_dirs = {norm(p.name): p for p in Path(pv_root).iterdir() if p.is_dir()}
    pd_dirs = {}
    for split in ("train", "test"):
        for p in (Path(pd_root) / split).iterdir():
            if p.is_dir():
                pd_dirs.setdefault(norm(p.name), []).append(p)

    classes, pv_items, pd_items, unmatched = [], [], [], []
    for pd_name, pv_name in PD_TO_PV.items():
        a, b = pd_dirs.get(norm(pd_name)), pv_dirs.get(norm(pv_name))
        if a is None or b is None:
            unmatched.append((pd_name, pv_name, a is not None, b is not None))
            continue
        y = len(classes)
        classes.append(pv_name)
        imgs = list_images(b)
        random.Random(seed).shuffle(imgs)
        pv_items += [(str(p), y) for p in imgs[:max_per_class]]
        for d in a:
            pd_items += [(str(p), y) for p in list_images(d)]

    if unmatched:
        print("\n[!] Mapping entries not found (pd_found, pv_found):")
        for u in unmatched:
            print("   ", u)
        print("    Your folders -- PlantVillage:", sorted(p.name for p in pv_dirs.values())[:50])
        print("    Your folders -- PlantDoc:", sorted({p.name for v in pd_dirs.values() for p in v}))
    if len(classes) < 10:
        raise SystemExit(f"Only {len(classes)} shared classes matched; fix PD_TO_PV or the paths.")

    # Drop unreadable PlantDoc files (the dataset has a few).
    ok = []
    for p, y in pd_items:
        try:
            with Image.open(p) as im:
                im.verify()
            ok.append((p, y))
        except Exception:
            pass
    print(f"Shared classes: {len(classes)} | PV images: {len(pv_items)} | "
          f"PD images: {len(ok)} (dropped {len(pd_items) - len(ok)} unreadable)")
    return classes, pv_items, ok


def stratified_split(items, fracs=(0.7, 0.15, 0.15), seed=0):
    # NOTE: pilot uses a random split. For the paper, split by perceptual-hash
    # groups so near-duplicate leaves never cross train/val/test.
    rng = random.Random(seed)
    by = {}
    for it in items:
        by.setdefault(it[1], []).append(it)
    out = ([], [], [])
    for v in by.values():
        rng.shuffle(v)
        n = len(v); a = int(fracs[0] * n); b = a + int(fracs[1] * n)
        out[0].extend(v[:a]); out[1].extend(v[a:b]); out[2].extend(v[b:])
    return out


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf
    def __len__(self):
        return len(self.items)
    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
TF_PLAIN = T.Compose([T.RandomResizedCrop(224, scale=(0.6, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), T.Normalize(MEAN, STD)])
# Robust: imitate field conditions (scale, framing, lighting, blur, occlusion) using lab images only.
TF_ROBUST = T.Compose([T.RandomResizedCrop(224, scale=(0.25, 1.0)), T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(), T.RandomRotation(20),
                       T.RandAugment(num_ops=2, magnitude=9),
                       T.ColorJitter(0.4, 0.4, 0.4, 0.1),
                       T.RandomApply([T.GaussianBlur(9, sigma=(0.1, 3.0))], p=0.3),
                       T.ToTensor(), T.Normalize(MEAN, STD), T.RandomErasing(p=0.25)])
TF_TRAIN = {"plain": TF_PLAIN, "robust": TF_ROBUST}
TF_EVAL = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def make_model(n_classes, pretrained):
    w = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
    m = mobilenet_v3_large(weights=w, quantize=False)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def train(model, train_items, val_items, args, device):
    dl = DataLoader(Files(train_items, TF_TRAIN[args.aug]), batch_size=args.bs, shuffle=True,
                    num_workers=args.workers, drop_last=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * len(dl))
    # Plain cross-entropy on purpose: label smoothing changes calibration.
    lossf = nn.CrossEntropyLoss()
    model.to(device)
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); tot = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            loss = lossf(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += loss.item()
        lg, yy = get_logits(model, val_items, args, device)
        print(f"epoch {ep+1}/{args.epochs} loss {tot/len(dl):.3f} "
              f"val acc {(lg.argmax(1)==yy).mean():.4f} ({time.time()-t0:.0f}s)")
    return model.cpu()


@torch.no_grad()
def get_logits(model, items, args, device="cpu"):
    model.eval().to(device)
    dl = DataLoader(Files(items, TF_EVAL), batch_size=args.bs, num_workers=args.workers)
    L, Y = [], []
    for x, y in dl:
        L.append(model(x.to(device)).float().cpu()); Y.append(y)
    return torch.cat(L).numpy(), torch.cat(Y).numpy()


def pick_quant_engine():
    """Use the first INT8 CPU engine this PyTorch build can actually run.
    x86/fbgemm on Intel/AMD CPUs (Windows, Linux), qnnpack on ARM (Mac M-series, phones)."""
    tried = []
    for e in ("x86", "fbgemm", "onednn", "qnnpack"):
        try:
            torch.backends.quantized.engine = e
            return e
        except RuntimeError as err:
            tried.append(f"{e}: {err}")
    raise SystemExit("No quantized engine works in this PyTorch build "
                     f"(supported_engines={torch.backends.quantized.supported_engines}).\n" + "\n".join(tried))


def quantize_int8_real(model, calib_items, args):
    """True INT8 model using PyTorch's CPU integer kernels (static PTQ)."""
    engine = pick_quant_engine()
    print(f"Quantizing to INT8 (real integer kernels) with engine: {engine}")
    m = copy.deepcopy(model).cpu().eval()
    m.fuse_model(is_qat=False)
    m.qconfig = tq.get_default_qconfig(engine)
    tq.prepare(m, inplace=True)
    get_logits(m, calib_items, args, "cpu")  # observers collect activation ranges
    tq.convert(m, inplace=True)
    return m


def quantize_sim(model, calib_items, args, device, w_bits=8):
    """Exact quantization simulation (fake quantization), independent of the CPU engine.
    Same scheme as mobile runtimes: weights signed per-channel symmetric (w_bits),
    activations uint8 per-tensor affine, ranges from the calibration images.
    BatchNorm is folded into the convolutions before weights are rounded.
    For w_bits < 8 the first convolution and the final classifier stay at 8 bits (standard practice)."""
    label = "INT8" if w_bits == 8 else f"INT{w_bits} weights / INT8 activations"
    print(f"Quantizing to {label} (simulated, per-channel weights, per-tensor uint8 activations)")
    act = tq.FakeQuantize.with_args(observer=tq.MinMaxObserver, quant_min=0, quant_max=255,
                                    dtype=torch.quint8, qscheme=torch.per_tensor_affine)
    def wq(bits):
        return tq.FakeQuantize.with_args(observer=tq.PerChannelMinMaxObserver,
                                         quant_min=-(2 ** (bits - 1)), quant_max=2 ** (bits - 1) - 1,
                                         dtype=torch.qint8, qscheme=torch.per_channel_symmetric, ch_axis=0)
    q8 = tq.QConfig(activation=act, weight=wq(8))
    m = copy.deepcopy(model).cpu().train()
    m.fuse_model(is_qat=True)
    m.qconfig = tq.QConfig(activation=act, weight=wq(w_bits))
    if w_bits < 8:
        m.features[0].qconfig = q8        # first conv
        m.classifier[3].qconfig = q8      # final classifier
    tq.prepare_qat(m, inplace=True)
    m.eval()                                       # BatchNorm uses its running statistics
    m.apply(tq.enable_observer); m.apply(tq.enable_fake_quant)
    get_logits(m, calib_items, args, device)       # collect activation ranges
    m.apply(tq.disable_observer)                   # freeze ranges, keep rounding on
    return m


def quantize_int8(model, calib_items, val_items, fp_val_logits, y_val, args, device):
    """Real INT8 when this machine's integer kernels work; otherwise exact simulation.
    Returns (model, logits_device, mode)."""
    if args.quant_mode in ("auto", "real"):
        try:
            qm = quantize_int8_real(model, calib_items, args)
            lg, _ = get_logits(qm, val_items, args, "cpu")
            fp_acc = (fp_val_logits.argmax(1) == y_val).mean()
            q_acc = (lg.argmax(1) == y_val).mean()
            if not np.isfinite(lg).all():
                raise RuntimeError("INT8 model produced NaN/inf outputs")
            if q_acc < fp_acc - 0.2:
                raise RuntimeError(f"INT8 accuracy collapsed ({q_acc:.3f} vs FP32 {fp_acc:.3f}); "
                                   "likely a kernel bug for this model on this engine")
            return qm, "cpu", "real"
        except (RuntimeError, SystemExit) as e:
            if args.quant_mode == "real":
                raise
            print(f"[!] Real INT8 failed on this machine: {e}\n    Falling back to exact INT8 simulation.")
    return quantize_sim(model, calib_items, args, device, 8), device, "simulated"


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def softmax(z, t=1.0):
    z = z / t; z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(logits, labels):
    """Fit on SOURCE validation logits only. Optimises log T so T stays > 0."""
    logT = torch.zeros(1, requires_grad=True)
    L, y = torch.tensor(logits, dtype=torch.float64), torch.tensor(labels)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=200)
    def step():
        opt.zero_grad(); loss = nn.functional.cross_entropy(L / logT.exp(), y)
        loss.backward(); return loss
    opt.step(step)
    return float(logT.detach().exp())


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1); e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return e


def aurc(conf, correct):
    c = correct[np.argsort(-conf)]
    risk = 1 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return risk.mean()


def risk_curve(conf, correct):
    c = correct[np.argsort(-conf)]
    k = np.arange(1, len(c) + 1)
    return k / len(c), 1 - np.cumsum(c) / k


def acc_at_cov(conf, correct, cov=0.8):
    k = max(1, int(round(cov * len(conf))))
    return correct[np.argsort(-conf)[:k]].mean()


def all_metrics(logits, y, t=1.0):
    p = softmax(logits, t); conf = p.max(1); pred = p.argmax(1); corr = (pred == y).astype(float)
    nll = -np.log(p[np.arange(len(y)), y] + 1e-12).mean()
    err_auroc = roc_auc_score(corr, conf) if 0 < corr.mean() < 1 else float("nan")
    d = dict(acc=corr.mean(), macro_f1=f1_score(y, pred, average="macro"),
                ece=ece(conf, corr), nll=nll, aurc=aurc(conf, corr),
                acc_at_80=acc_at_cov(conf, corr), err_auroc=err_auroc,
                conf_wrong=conf[corr == 0].mean() if (corr == 0).any() else float("nan"))
    return {k: float(v) for k, v in d.items()}


def bootstrap_gap(fp_pv, q_pv, y_pv, fp_pd, q_pd, y_pd, key, t_fp, t_q, n=1000, seed=0):
    """CI for: (INT8 - FP32 on PlantDoc) - (INT8 - FP32 on PlantVillage).
    Positive gap on ece/aurc = quantization hurts MORE under shift."""
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        i = rng.integers(0, len(y_pv), len(y_pv)); j = rng.integers(0, len(y_pd), len(y_pd))
        d_pv = all_metrics(q_pv[i], y_pv[i], t_q)[key] - all_metrics(fp_pv[i], y_pv[i], t_fp)[key]
        d_pd = all_metrics(q_pd[j], y_pd[j], t_q)[key] - all_metrics(fp_pd[j], y_pd[j], t_fp)[key]
        out.append(d_pd - d_pv)
    return float(np.mean(out)), float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def flip_by_conf(fp_logits, q_logits, bins=(0, .5, .7, .9, .99, 1.0001)):
    conf = softmax(fp_logits).max(1); flip = fp_logits.argmax(1) != q_logits.argmax(1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append((f"{lo:.2f}-{min(hi,1):.2f}", int(m.sum()), float(flip[m].mean()) if m.any() else float("nan")))
    return float(flip.mean()), rows


# ----------------------------------------------------------------------------
VARIANTS = {"q8": "INT8", "q4": "INT4-W"}
CACHE_KEYS = ("pv", "pd", "epochs", "lr", "max_per_class", "calib_n", "seed", "no_pretrained",
              "quant_mode", "aug", "field_cal_frac", "int4")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pv", required=True); ap.add_argument("--pd", required=True)
    ap.add_argument("--out", default="runs/pilot")
    ap.add_argument("--epochs", type=int, default=10); ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-per-class", type=int, default=1000, help="PlantVillage images per class (0 = all)")
    ap.add_argument("--aug", choices=["plain", "robust"], default="robust")
    ap.add_argument("--calib-n", type=int, default=256, help="PV train images for PTQ calibration")
    ap.add_argument("--field-cal-frac", type=float, default=0.2,
                    help="share of PlantDoc held out (labels used) to fit a field temperature; rest = field test")
    ap.add_argument("--no-int4", dest="int4", action="store_false", help="skip the INT4-weight variant")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--quant-mode", choices=["auto", "real", "sim"], default="auto",
                    help="INT8: real integer kernels if they work on this machine, else exact simulation")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    if args.max_per_class <= 0:
        args.max_per_class = 10 ** 9

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # Cached stages are reused only if the settings that produced them are unchanged.
    cfg = {k: getattr(args, k) for k in CACHE_KEYS}
    cfg_path = out / "config.json"
    cached = any((out / f).exists() for f in ("fp32.pt", "logits.npz"))
    stale = cached and (not cfg_path.exists() or json.load(open(cfg_path)) != json.loads(json.dumps(cfg)))
    if stale:
        print("[!] Settings differ from the cached run in this folder (or it is from an older version): "
              "retraining from scratch.")
        for f in ("fp32.pt", "int8.pt", "int4w.pt", "logits.npz"):
            (out / f).unlink(missing_ok=True)
    json.dump(cfg, open(cfg_path, "w"), indent=1)

    classes, pv, pd_all = build_index(args.pv, args.pd, args.max_per_class, args.seed)
    tr, va, te = stratified_split(pv, seed=args.seed)
    pd_cal, _, pd = stratified_split(pd_all, fracs=(args.field_cal_frac, 0.0), seed=args.seed)
    print(f"PV train/val/test: {len(tr)}/{len(va)}/{len(te)} | PD field-cal/field-test: {len(pd_cal)}/{len(pd)}")
    json.dump({"classes": classes}, open(out / "classes.json", "w"), indent=1)
    sets = (("pv_val", va), ("pv_test", te), ("pd_cal", pd_cal), ("pd", pd))

    # --- FP32 ---------------------------------------------------------------
    ck = out / "fp32.pt"
    model = make_model(len(classes), not args.no_pretrained)
    if ck.exists():
        model.load_state_dict(torch.load(ck)); print("Loaded", ck)
    else:
        model = train(model, tr, va, args, device); torch.save(model.state_dict(), ck)

    cache = out / "logits.npz"
    if cache.exists():
        z = dict(np.load(cache)); print("Loaded", cache)
    else:
        z = {}
        for name, items in sets:
            z[f"fp_{name}"], z[f"y_{name}"] = get_logits(model, items, args, device)
        calib = random.Random(args.seed).sample(tr, min(args.calib_n, len(tr)))
        # INT8: real integer kernels when they work here, else exact simulation
        qmodel, qdev, qmode = quantize_int8(model, calib, va, z["fp_pv_val"], z["y_pv_val"], args, device)
        torch.save(qmodel.state_dict(), out / "int8.pt")
        for name, items in sets:
            z[f"q8_{name}"], _ = get_logits(qmodel, items, args, qdev)
        z["quant_mode"] = np.array(qmode)
        # INT4 weights (INT8 activations): simulated; no CPU kernels exist for this
        if args.int4:
            q4 = quantize_sim(model, calib, args, device, w_bits=4)
            torch.save(q4.state_dict(), out / "int4w.pt")
            for name, items in sets:
                z[f"q4_{name}"], _ = get_logits(q4, items, args, device)
        for k in [k for k in z if k.startswith(("fp_", "q8_", "q4_"))]:
            if not np.isfinite(z[k]).all():
                raise SystemExit(f"Non-finite logits in {k}; please send this output to Claude.")
        np.savez(cache, **z)

    variants = [v for v in VARIANTS if f"{v}_pd" in z]
    qmode = str(z.get("quant_mode", "real"))
    print(f"\nINT8 mode: {qmode}" + (" | INT4-W: simulated" if "q4" in variants else ""))
    for v in variants:
        acc_fp = (z["fp_pv_val"].argmax(1) == z["y_pv_val"]).mean()
        acc_q = (z[f"{v}_pv_val"].argmax(1) == z["y_pv_val"]).mean()
        if acc_q < acc_fp - 0.2:
            print(f"[!] {VARIANTS[v]} lab accuracy collapsed ({acc_q:.3f} vs FP32 {acc_fp:.3f}).")

    # --- Temperatures --------------------------------------------------------
    # source: fitted on lab validation (no field data); field: fitted on the labeled field-cal split
    T = {"FP32": {"source": fit_temperature(z["fp_pv_val"], z["y_pv_val"]),
                  "field": fit_temperature(z["fp_pd_cal"], z["y_pd_cal"])}}
    for v in variants:
        T[VARIANTS[v]] = {"source_refit": fit_temperature(z[f"{v}_pv_val"], z["y_pv_val"]),
                          "field": fit_temperature(z[f"{v}_pd_cal"], z["y_pd_cal"])}

    res = {"quant_mode": qmode, "variants": [VARIANTS[v] for v in variants], "T": T,
           "n_classes": len(classes), "n_pv_train": len(tr), "n_pv_test": int(len(z["y_pv_test"])),
           "n_pd_cal": int(len(z["y_pd_cal"])), "n_pd": int(len(z["y_pd"])),
           "settings": {k: v for k, v in cfg.items() if k not in ("pv", "pd")}}
    rows = []
    for ds in ("pv_test", "pd"):
        y = z[f"y_{ds}"]
        conds = [("FP32 raw", z[f"fp_{ds}"], 1.0),
                 ("FP32 +T(source)", z[f"fp_{ds}"], T["FP32"]["source"]),
                 ("FP32 +T(field)", z[f"fp_{ds}"], T["FP32"]["field"])]
        for v in variants:
            n = VARIANTS[v]
            conds += [(f"{n} raw", z[f"{v}_{ds}"], 1.0),
                      (f"{n} +T(fp32 source)", z[f"{v}_{ds}"], T["FP32"]["source"]),
                      (f"{n} +T(refit source)", z[f"{v}_{ds}"], T[n]["source_refit"]),
                      (f"{n} +T(field)", z[f"{v}_{ds}"], T[n]["field"])]
        for label, lg, t in conds:
            m = all_metrics(lg, y, t); m.update(dataset=ds, model=label); rows.append(m)
    res["table"] = rows

    print("\nTemperatures:", json.dumps({k: {a: round(b, 3) for a, b in d.items()} for k, d in T.items()}))
    hdr = ["dataset", "model", "acc", "macro_f1", "ece", "nll", "aurc", "acc_at_80", "err_auroc", "conf_wrong"]
    print("\n" + " | ".join(f"{h:>20}" if h == "model" else f"{h:>9}" for h in hdr))
    for r in rows:
        print(" | ".join(f"{r[h]:>9.4f}" if isinstance(r[h], float) else
                         (f"{r[h]:>20}" if h == "model" else f"{r[h]:>9}") for h in hdr))

    res["flip"], res["gap"], res["hint"] = {}, {}, {}
    for v in variants:
        n = VARIANTS[v]
        print(f"\nFlip rate (FP32 vs {n} disagree), by FP32 confidence:")
        res["flip"][n] = {}
        for ds in ("pv_test", "pd"):
            tot, fr = flip_by_conf(z[f"fp_{ds}"], z[f"{v}_{ds}"]); res["flip"][n][ds] = {"overall": tot, "bins": fr}
            print(f"  {ds:>7}: overall {tot:.4f}  " + "  ".join(f"[{b}] n={c} {f:.3f}" for b, c, f in fr))
        print(f"Shift gap = ({n}-FP32 on PlantDoc) - ({n}-FP32 on PlantVillage), raw softmax, {args.boot} bootstraps:")
        res["gap"][n] = {}
        for key in ("acc", "ece", "aurc", "err_auroc"):
            g = bootstrap_gap(z["fp_pv_test"], z[f"{v}_pv_test"], z["y_pv_test"],
                              z["fp_pd"], z[f"{v}_pd"], z["y_pd"], key, 1.0, 1.0, n=args.boot)
            res["gap"][n][key] = g
            print(f"  {key:>10}: {g[0]:+.4f}  95% CI [{g[1]:+.4f}, {g[2]:+.4f}]")
        # Rough decision hint, not a statistical test of the paper's claims.
        e, a = res["gap"][n]["ece"], res["gap"][n]["aurc"]
        if e[1] > 0 or a[1] > 0:
            hint = f"GO: {n} measurably hurts reliability more under shift (CI excludes 0)."
        elif e[0] > 0.005 or a[0] > 0.005:
            hint = f"LEAN GO: {n} gap points the expected way but CI includes 0; add seeds."
        else:
            hint = f"NULL: {n} doesn't widen the lab-to-field reliability gap."
        res["hint"][n] = hint; print(hint)
    json.dump(res, open(out / "results.json", "w"), indent=1, default=float)

    # --- Plots ---------------------------------------------------------------
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    cols = {"fp": "#2563eb", "q8": "#dc2626", "q4": "#f59e0b"}
    names = {"fp": "FP32", **VARIANTS}
    fig, ax = plt.subplots(1, 1 + len(variants), figsize=(5 * (1 + len(variants)), 4))
    for ds, ls in (("pv_test", "-"), ("pd", "--")):
        y = z[f"y_{ds}"]
        for tag in ["fp"] + variants:
            p = softmax(z[f"{tag}_{ds}"]); cov, risk = risk_curve(p.max(1), (p.argmax(1) == y).astype(float))
            ax[0].plot(cov, risk, ls, color=cols[tag], label=f"{names[tag]} {'lab' if ds == 'pv_test' else 'field'}")
    ax[0].set(xlabel="coverage", ylabel="risk (error rate)", title="Risk-coverage"); ax[0].legend(fontsize=8)
    w = 0.38
    for i, v in enumerate(variants, 1):
        for k, (ds, col) in enumerate((("pv_test", "#64748b"), ("pd", "#ea580c"))):
            fr = res["flip"][VARIANTS[v]][ds]["bins"]
            ax[i].bar(np.arange(len(fr)) + (k - .5) * w, [f for _, _, f in fr], w, color=col,
                      label="lab" if ds == "pv_test" else "field")
        ax[i].set_xticks(range(len(fr)), [b for b, _, _ in fr], rotation=20)
        ax[i].set(xlabel="FP32 confidence", ylabel="flip rate", title=f"Where {VARIANTS[v]} changes the answer")
        ax[i].legend()
    fig.tight_layout(); fig.savefig(out / "pilot.png", dpi=150)
    print("Saved", out / "results.json", "and", out / "pilot.png")


if __name__ == "__main__":
    main()
