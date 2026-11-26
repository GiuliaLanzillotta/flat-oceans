"""Native full-network fine-tuning of torchvision ResNet-50 (IMAGENET1K_V2) on RESISC45, 45-way linear head.
Parallel to finetune_vit_resisc.py (SAME dataset splits, val protocol, eval, loss/aug/schedule/clip/AMP, best-val
checkpoint) for comparability; only the backbone and the ResNet-appropriate hyperparameters differ. Forward-ONLY
preprocessing = the weights' own eval transform (ResNet50_V2: resize->centercrop 224->ImageNet norm), on GPU. The full
backbone (conv + BatchNorm + residuals) is trained (BN in TRAIN mode during training; eval uses running stats). The
later atomic feature/downstream-family analysis will load this checkpoint in EVAL mode. Light GPU job.
Checkpoint format matches the ViT runs: {state_dict, epoch, val_acc, test_acc, args, val_idx, n_classes}."""
import argparse, json, os, sys, time, numpy as np, torch, torch.nn as nn
import torchvision.models as tvm
from torchvision.models import ResNet50_Weights

SP = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, SP)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(SP))), "runs", "ftvit_resisc45", "scripts"))
from ftvit_paths import data_path                    # SAME RESISC45 arrays as the ViT run


def load_split():
    Xtr = np.load(data_path("resisc45_train_imgs.npy"), mmap_mode="r")   # (18900,256,256,3) uint8
    ytr = np.load(data_path("resisc45_train_labels.npy"))
    Xte = np.load(data_path("resisc45_test_imgs.npy"), mmap_mode="r")
    yte = np.load(data_path("resisc45_test_labels.npy"))
    return Xtr, ytr, Xte, yte


def val_indices(ytr, per_class=50, seed=54321):        # IDENTICAL to the ViT split
    rng = np.random.RandomState(seed); val = []
    for c in np.unique(ytr):
        idx = np.where(ytr == c)[0]; val.append(rng.choice(idx, size=per_class, replace=False))
    val = np.sort(np.concatenate(val))
    mask = np.ones(len(ytr), bool); mask[val] = False
    return np.where(mask)[0], val


@torch.no_grad()
def to_batch(Xu, idx, device):
    b = np.asarray(Xu[idx]); x = torch.from_numpy(b).to(device).float().div_(255.0).permute(0, 3, 1, 2).contiguous()
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)            # ResNet-50 AdamW full fine-tune default
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--loss", choices=["ce", "mse"], default="ce")
    ap.add_argument("--aug", choices=["none", "flip"], default="flip")
    ap.add_argument("--warmup-epochs", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--tag", default="ft_resnet_resisc")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    ap.add_argument("--save-ckpt", default=None)
    args = ap.parse_args()
    dev = torch.device(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed)

    Xtr, ytr, Xte, yte = load_split(); tr_idx, va_idx = val_indices(ytr)
    if args.limit_train is not None: tr_idx = tr_idx[:args.limit_train]
    ytr_t = torch.from_numpy(ytr).long(); yte_t = torch.from_numpy(yte).long(); nc = int(ytr.max()) + 1
    print(f"train {len(tr_idx)} val {len(va_idx)} test {len(yte)} classes {nc}", flush=True)

    weights = ResNet50_Weights.IMAGENET1K_V2
    tfm = weights.transforms()                       # Resize->CenterCrop(224)->ImageNet Normalize
    net = tvm.resnet50(weights=weights)
    net.fc = nn.Linear(2048, nc); nn.init.zeros_(net.fc.bias); nn.init.trunc_normal_(net.fc.weight, std=0.02)
    net.to(dev)

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per = (len(tr_idx) + args.batch_size - 1) // args.batch_size
    total_steps = steps_per * args.epochs; warm = int(steps_per * args.warmup_epochs)
    def lr_at(step):
        if step < warm: return step / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm); return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    scaler = torch.cuda.amp.GradScaler(); ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(Xu, y_t, idx):
        net.eval(); correct = 0; loss_sum = 0.0; n = len(idx)
        for i in range(0, n, 128):
            j = idx[i:i + 128]; x = tfm(to_batch(Xu, j, dev))
            with torch.cuda.amp.autocast():
                logit = net(x)
            yb = y_t[j].to(dev); correct += (logit.argmax(1) == yb).sum().item(); loss_sum += ce(logit.float(), yb).item() * len(j)
        return correct / n, loss_sum / n

    curves = []; best = {"val_acc": -1}; all_tr = np.array(tr_idx)
    for ep in range(1, args.epochs + 1):
        net.train(); t0 = time.time(); rng = np.random.RandomState(1000 + args.seed * 97 + ep)
        order = rng.permutation(len(all_tr)); run_correct = 0; run_loss = 0.0; seen = 0
        for i in range(0, len(order), args.batch_size):
            j = all_tr[order[i:i + args.batch_size]]; x = tfm(to_batch(Xtr, j, dev))
            if args.aug == "flip":
                if rng.rand() < 0.5: x = torch.flip(x, dims=[3])
                if rng.rand() < 0.5: x = torch.flip(x, dims=[2])
            yb = ytr_t[j].to(dev); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                logit = net(x)
                if args.loss == "ce": loss = ce(logit, yb)
                else:
                    tgt = torch.zeros_like(logit); tgt[torch.arange(len(yb)), yb] = 1.0; loss = ((logit - tgt) ** 2).mean()
            scaler.scale(loss).backward()
            if args.clip: scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), args.clip)
            scaler.step(opt); scaler.update(); sched.step()
            run_correct += (logit.argmax(1) == yb).sum().item(); run_loss += loss.item() * len(yb); seen += len(yb)
        tr_acc = run_correct / seen
        va_acc, va_loss = evaluate(Xtr, ytr_t, va_idx); te_acc, te_loss = evaluate(Xte, yte_t, np.arange(len(yte)))
        rec = {"epoch": ep, "train_acc": tr_acc, "train_loss": run_loss / seen, "val_acc": va_acc, "val_loss": va_loss,
               "test_acc": te_acc, "test_loss": te_loss, "lr": sched.get_last_lr()[0], "sec": time.time() - t0}
        curves.append(rec)
        print(f"ep{ep:02d} lr{rec['lr']:.2e} train {tr_acc:.4f} val {va_acc:.4f} test {te_acc:.4f} ({rec['sec']:.0f}s)", flush=True)
        if va_acc > best["val_acc"]:
            best = {"epoch": ep, "val_acc": va_acc, "test_acc": te_acc, "train_acc": tr_acc}
            if args.save_ckpt:
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "epoch": ep,
                            "val_acc": va_acc, "test_acc": te_acc, "args": vars(args), "val_idx": va_idx, "n_classes": nc}, args.save_ckpt)
    summary = {"args": vars(args), "best_val": best, "final": curves[-1],
               "train_test_gap_at_best": best["train_acc"] - best["test_acc"], "curves": curves,
               "backbone": "torchvision resnet50 ResNet50_Weights.IMAGENET1K_V2 (full fine-tune)"}
    out = args.out or os.path.join(SP, f"{args.tag}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print("BEST", json.dumps(best), "GAP", summary["train_test_gap_at_best"], flush=True)
    print("wrote", out, "\nDONE", flush=True)


if __name__ == "__main__":
    main()
