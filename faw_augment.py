#!/usr/bin/env python3
"""Diverse augmentation for FAW injury (1-9) fine-tuning.

Designed for FINE-TUNING a pretrained backbone on the FAW dataset:
  * on-the-fly (applied in the DataLoader) -> fresh variety each epoch, no disk blow-up
  * TRAIN split only. val/test use get_eval_transform (clean).

Techniques (grouped):
  geometric   : RandomResizedCrop, H/V flip, small rotation, affine, perspective
  photometric : ColorJitter (mild), RandomAutocontrast, RandomAdjustSharpness, gamma
  degradation : GaussianBlur, Gaussian noise, JPEG-style compression
  occlusion   : RandomErasing (mild)
  background  : BackgroundRemover -> plant isolated on white/black/blur/random bg
                (domain randomization; classical ExG+Otsu+GrabCut, no model download)

ORDINAL CAUTION: injury score is a 1-9 severity scale. Keep color/erasing MILD so
augmentation doesn't hide or fake injury. MixUp/CutMix mix two severities -> only
use with a soft/ordinal target (helper provided), not hard labels.
"""
import numpy as np, cv2, torch
from PIL import Image
from torchvision.transforms import v2
from torchvision import tv_tensors

# --------------------------------------------------------------------------- #
#  Background removal (classical, no external weights)
# --------------------------------------------------------------------------- #
class BackgroundRemover:
    """Segment the plant via Excess-Green + Otsu, refine with GrabCut, then
    composite onto a chosen background. Callable on a PIL.Image -> PIL.Image."""
    def __init__(self, bg="white", p=1.0, grabcut=True):
        # bg in {"white","black","blur","random"}; p = prob of applying
        self.bg, self.p, self.grabcut = bg, p, grabcut

    def _mask(self, rgb):
        r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
        exg = 2 * g - r - b                                   # vegetation index
        exg = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        _, m = cv2.threshold(exg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        if self.grabcut and m.sum() > 0:
            gc = np.where(m > 0, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype("uint8")
            bgm, fgm = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
            try:
                cv2.grabCut(rgb, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
                m = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype("uint8")
            except cv2.error:
                pass
        return m

    def __call__(self, img):
        if self.p < 1.0 and torch.rand(1).item() > self.p:
            return img
        rgb = np.array(img.convert("RGB"))
        m = self._mask(rgb)
        m3 = (m[..., None] / 255.0)
        bg = self.bg
        if bg == "random":
            bg = ["white", "black", "blur"][int(torch.randint(0, 3, (1,)).item())]
        if bg == "white":
            back = np.full_like(rgb, 255)
        elif bg == "black":
            back = np.zeros_like(rgb)
        else:  # blur the original -> keeps context but suppresses it
            back = cv2.GaussianBlur(rgb, (0, 0), 25)
        out = (rgb * m3 + back * (1 - m3)).astype(np.uint8)
        return Image.fromarray(out)


class AddGaussianNoise:
    def __init__(self, std=0.03, p=0.3):
        self.std, self.p = std, p
    def __call__(self, t):                                    # t: float tensor CHW in [0,1]
        if torch.rand(1).item() > self.p:
            return t
        return torch.clamp(t + torch.randn_like(t) * self.std, 0, 1)


class RandomJPEG:
    def __init__(self, q=(40, 90), p=0.3):
        self.q, self.p = q, p
    def __call__(self, img):                                  # PIL -> PIL
        if torch.rand(1).item() > self.p:
            return img
        q = int(torch.randint(self.q[0], self.q[1] + 1, (1,)).item())
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(np.array(img.convert("RGB")),
                               cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        return Image.fromarray(cv2.cvtColor(cv2.imdecode(enc, 1), cv2.COLOR_BGR2RGB)) if ok else img


# --------------------------------------------------------------------------- #
#  Transform builders
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform(img_size=224, bg_remove_p=0.25, bg_mode="random"):
    """On-the-fly TRAIN augmentation. bg_remove_p=0 disables background removal."""
    pil_ops = []
    if bg_remove_p > 0:
        pil_ops.append(BackgroundRemover(bg=bg_mode, p=bg_remove_p, grabcut=False))
    pil_ops.append(RandomJPEG(p=0.25))
    return v2.Compose([
        *pil_ops,
        v2.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33), antialias=True),
        v2.RandomHorizontalFlip(0.5),
        v2.RandomVerticalFlip(0.2),
        v2.RandomApply([v2.RandomRotation(20)], p=0.5),
        v2.RandomApply([v2.RandomAffine(0, translate=(0.06, 0.06), shear=8)], p=0.3),
        v2.RandomPerspective(distortion_scale=0.2, p=0.2),
        v2.RandomApply([v2.ColorJitter(0.2, 0.2, 0.2, 0.03)], p=0.7),   # MILD (ordinal)
        v2.RandomAutocontrast(0.2),
        v2.RandomAdjustSharpness(1.5, 0.2),
        v2.RandomApply([v2.GaussianBlur(5, (0.1, 2.0))], p=0.2),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        AddGaussianNoise(0.03, 0.3),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        v2.RandomErasing(p=0.2, scale=(0.02, 0.12)),                    # MILD occlusion
    ])

def get_eval_transform(img_size=224, bg_remove=False, bg_mode="white"):
    """Clean val/test transform. If you fixed-preprocess with bg removal in TRAIN,
    set bg_remove=True here too (must be deterministic, p=1)."""
    ops = []
    if bg_remove:
        ops.append(BackgroundRemover(bg=bg_mode, p=1.0, grabcut=True))
    return v2.Compose([
        *ops,
        v2.Resize(int(img_size * 1.14), antialias=True),
        v2.CenterCrop(img_size),
        v2.ToImage(), v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def get_mixup_cutmix(num_classes=9, p=0.3):
    """Optional. NOTE: mixes two severities -> soft targets. Use with a loss that
    accepts soft labels (e.g. SoftTargetCrossEntropy). Disable if using hard CE."""
    return v2.RandomChoice([
        v2.MixUp(alpha=0.2, num_classes=num_classes),
        v2.CutMix(alpha=1.0, num_classes=num_classes),
    ], p=[0.5, 0.5])


if __name__ == "__main__":
    # tiny self-test
    im = Image.new("RGB", (256, 256), (120, 160, 90))
    t = get_train_transform(224)(im)
    print("train tensor:", tuple(t.shape), float(t.min()), float(t.max()))
    print("eval  tensor:", tuple(get_eval_transform(224)(im).shape))
    print("bg-removed PIL:", BackgroundRemover('white')(im).size)
