"""
Vision.

ImageNet ViT-B/16 and CUB-200 ResNet-50: top-1 accuracy.
ADE20K: SETR-Naive ViT-L/16, mIoU.
Few-shot CUB and ImageNet-pretrained ViT → CUB-200 keep top-1 accuracy.

    python experiments/run_vision.py --benchmark imagenet --method flowra --data-dir /data/imagenet
    python experiments/run_vision.py --benchmark cub --method flowra --data-dir /data/CUB_200_2011
    python experiments/run_vision.py --benchmark cub --shots 4 --method flowra --data-dir /data/CUB_200_2011
    python experiments/run_vision.py --benchmark transfer --method flowra --data-dir /data/CUB_200_2011
    python experiments/run_vision.py --benchmark ade20k --method flowra --data-dir /data/ADEChallengeData2016
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from common import ROOT, device_str, load_config, resolve_config, set_seed, training_config
from methods import METHODS, attach, train_attached, trainable_count
from metrics import confusion_miou, miou_from_confusion
from flowra.utils.logging_utils import ExperimentLogger
from flowra.utils.batching import move_to_device


def _tfm(size):
    return transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def _folder(root, train: bool, size: int):
    split = "train" if train else "val"
    path = os.path.join(root, split)
    if not os.path.isdir(path):
        path = os.path.join(root, "train" if train else "test")
    return datasets.ImageFolder(path, transform=_tfm(size))


def _fewshot(ds, shots: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    by_c = {}
    for i, (_, y) in enumerate(ds.samples):
        by_c.setdefault(y, []).append(i)
    idx = []
    for y, ids in by_c.items():
        perm = torch.randperm(len(ids), generator=g).tolist()
        idx.extend(ids[j] for j in perm[:shots])
    return Subset(ds, idx)


def _timm(name: str, num_classes: int, pretrained: bool = True):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)


class ADE20KSeg(Dataset):
    """ADEChallengeData2016 layout: images/{training,validation}, annotations/{training,validation}."""

    def __init__(self, root: str, train: bool, size: int):
        split = "training" if train else "validation"
        self.img_dir = os.path.join(root, "images", split)
        self.ann_dir = os.path.join(root, "annotations", split)
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(self.img_dir)
        self.names = sorted(f for f in os.listdir(self.img_dir) if f.lower().endswith((".jpg", ".png")))
        self.img_tf = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.mask_tf = transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST)
        self.size = size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        stem = os.path.splitext(name)[0]
        mask_path = os.path.join(self.ann_dir, stem + ".png")
        mask = Image.open(mask_path)
        img_t = self.img_tf(img)
        mask_t = torch.from_numpy(np.array(self.mask_tf(mask), dtype=np.int64))
        # ADE annotations: 0 = background / unlabeled → ignore; 1..150 → classes 0..149
        labels = mask_t - 1
        labels[mask_t == 0] = 255
        return img_t, labels


def _vit_l_setr(backbone_name: str, image_size: int):
    """SETR-Naive encoder: ViT-L/16, ImageNet-21k, patch tokens at ``image_size``."""
    import timm
    kwargs = dict(pretrained=True, num_classes=0, global_pool="", img_size=image_size)
    for name in (backbone_name, "vit_large_patch16_224.augreg_in21k", "vit_large_patch16_224"):
        try:
            return timm.create_model(name, **kwargs)
        except Exception:
            continue
    return timm.create_model("vit_large_patch16_224", **kwargs)


class ViTSeg(nn.Module):
    """SETR-Naive (Zheng et al.): 1x1 conv on ViT-L/16 patch tokens, bilinear upsample.

    SETR-Naive ViT-L/16 on ADE20K (Zheng et al.).
    """

    def __init__(self, backbone_name: str, num_classes: int = 150, image_size: int = 512):
        super().__init__()
        self.backbone = _vit_l_setr(backbone_name, image_size)
        dim = self.backbone.num_features
        self.head = nn.Conv2d(dim, num_classes, kernel_size=1)
        self.num_classes = num_classes
        self.image_size = image_size

    def forward(self, pixel_values=None, labels=None, **kwargs):
        feat = self.backbone.forward_features(pixel_values)
        patches = feat[:, 1:] if feat.dim() == 3 and feat.shape[1] > 1 else feat
        if patches.dim() == 3:
            b, n, c = patches.shape
            h = w = int(n ** 0.5)
            logits = self.head(patches.reshape(b, h, w, c).permute(0, 3, 1, 2))
        else:
            pooled = patches.mean((-2, -1))[:, :, None, None]
            logits = self.head(pooled)
            logits = logits.expand(-1, -1, pixel_values.shape[-2] // 16, pixel_values.shape[-1] // 16)
        logits = F.interpolate(logits, size=pixel_values.shape[-2:], mode="bilinear", align_corners=False)
        loss = None
        if labels is not None:
            if labels.shape[-2:] != logits.shape[-2:]:
                up = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
            else:
                up = logits
            loss = F.cross_entropy(up, labels.long(), ignore_index=255)
        return type("O", (), {"logits": logits, "loss": loss})()


def collate_cls(batch):
    x = torch.stack([b[0] for b in batch])
    y0 = batch[0][1]
    if torch.is_tensor(y0) and y0.dim() >= 1:
        y = torch.stack([b[1] for b in batch])
    else:
        y = torch.tensor([b[1] for b in batch])
    return {"pixel_values": x, "labels": y}


def patch_forward(model):
    orig = model.forward

    def wrapped(*, pixel_values=None, labels=None, **kwargs):
        logits = orig(pixel_values)
        if labels is None:
            return type("O", (), {"logits": logits, "loss": None})()
        loss = F.cross_entropy(logits, labels)
        return type("O", (), {"logits": logits, "loss": loss})()
    model.forward = wrapped
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--benchmark", required=True, choices=("imagenet", "cub", "ade20k", "transfer"))
    ap.add_argument("--method", default="flowra", choices=METHODS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0, help="CUB few-shot; 0 = full")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "vision"))
    args = ap.parse_args()

    cfg = load_config(resolve_config(args.config, "vision.yaml"))
    data_dir = args.data_dir or cfg.get("data_dir") or ""
    if not data_dir:
        raise SystemExit("Pass --data-dir to the ImageNet / CUB / ADE20K root")
    device = device_str()
    set_seed(args.seed)
    size = int(cfg["image_size"])
    tag = f"{args.benchmark}_{args.method}_shot{args.shots}_seed{args.seed}"
    out = os.path.join(args.out, tag)
    logger = ExperimentLogger(out, name=args.benchmark)
    logger.log_environment()
    n_cls = int(cfg["num_ade20k_classes"])
    ignore = int(cfg.get("ade20k_ignore_index") or 255)

    if args.benchmark == "ade20k":
        seg_size = int(cfg.get("seg_image_size") or 512)
        model = ViTSeg(cfg["ade20k_model"], n_cls, image_size=seg_size).to(device)
        train_ds = ADE20KSeg(data_dir, True, seg_size)
        val_ds = ADE20KSeg(data_dir, False, seg_size)
        collate = collate_cls
    elif args.benchmark == "imagenet":
        model = patch_forward(_timm(cfg["imagenet_model"], cfg["num_imagenet_classes"]).to(device))
        train_ds, val_ds = _folder(data_dir, True, size), _folder(data_dir, False, size)
        collate = collate_cls
    else:
        name = cfg["transfer_model"] if args.benchmark == "transfer" else cfg["cub_model"]
        model = patch_forward(_timm(name, cfg["num_cub_classes"], pretrained=True).to(device))
        train_ds, val_ds = _folder(data_dir, True, size), _folder(data_dir, False, size)
        if args.shots:
            train_ds = _fewshot(train_ds, args.shots, args.seed)
        collate = collate_cls

    g = torch.Generator().manual_seed(args.seed)
    train = DataLoader(train_ds, batch_size=cfg["batch_size"] if args.benchmark != "ade20k" else min(4, cfg["batch_size"]),
                       shuffle=True, collate_fn=collate, generator=g)
    n_cal = min(int(cfg["flowra"]["calibration_samples"]), len(train_ds))
    calib = DataLoader(Subset(train_ds, range(n_cal)), batch_size=min(8, cfg["batch_size"]), collate_fn=collate)
    val = DataLoader(val_ds, batch_size=cfg["eval_batch_size"] if args.benchmark != "ade20k" else 2, collate_fn=collate)

    def evaluate(m):
        m.eval()
        if args.benchmark == "ade20k":
            conf = torch.zeros(n_cls, n_cls)
            with torch.no_grad():
                for batch in val:
                    batch = move_to_device(batch, device)
                    logits = m(**batch).logits
                    if logits.shape[-2:] != batch["labels"].shape[-2:]:
                        logits = F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear",
                                               align_corners=False)
                    pred = logits.argmax(1)
                    conf += confusion_miou(pred.cpu(), batch["labels"].cpu(), n_cls, ignore)
            iou = miou_from_confusion(conf)
            logger.info(f"ADE20K mIoU={100 * iou:.1f}")
            return {"miou": iou, "score": iou, "miou_pct": 100 * iou}
        correct = total = 0
        with torch.no_grad():
            for batch in val:
                batch = move_to_device(batch, device)
                pred = m(**batch).logits.argmax(-1)
                correct += int((pred == batch["labels"]).sum())
                total += batch["labels"].numel()
        acc = correct / max(total, 1)
        logger.info(f"{args.benchmark} top-1={100 * acc:.1f}")
        return {"accuracy": acc, "score": acc, "accuracy_pct": 100 * acc}

    model, fw, params_fn = attach(args.method, model, calib, cfg, args.seed, device, logger)
    epochs = cfg.get("ade20k_epochs") if args.benchmark == "ade20k" else cfg["training"].get("epochs")
    if args.benchmark == "ade20k":
        cfg = dict(cfg)
        cfg["training"] = dict(cfg.get("training") or {})
        cfg["training"]["lr_schedule"] = cfg["training"].get("ade20k_lr_schedule", "poly")
    tcfg = training_config(cfg, args.seed, epochs=epochs, eval_every=len(train))
    train_attached(model, fw, params_fn, train, tcfg, evaluate, logger)
    final = evaluate(model)
    result = {"benchmark": args.benchmark, "shots": args.shots,
              "method": args.method, "seed": args.seed, **final,
              "trainable_params": fw.parameter_counts()["adapter"] if fw else trainable_count(model)}
    logger.log("result", result, "RESULT " + json.dumps(result))
    if fw is not None:
        fw.save(os.path.join(out, "adapter.pt"))
    logger.close()


if __name__ == "__main__":
    main()
