#!/usr/bin/env python3
"""Self-contained FAW model architecture (no cv2 / augmentation / dataset deps).
Kept byte-identical to train_ft.py's Net so saved state_dict keys load cleanly.
Used by infer.py for inference + Grad-CAM (cluster figures and the HF Space app)."""
import torch, torch.nn as nn, timm

K = 9
BACKBONE = "vit_base_patch14_reg4_dinov2.lvd142m"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Net(nn.Module):
    def __init__(self, adapt, two, img_size=224):
        super().__init__()
        self.bb = timm.create_model(BACKBONE, pretrained=True, num_classes=0,
                                    img_size=img_size, dynamic_img_size=True)
        D = self.bb.num_features
        self.two = two; feat = D * 2 if two else D
        if adapt == "lora":
            from peft import LoraConfig, get_peft_model
            for p in self.bb.parameters(): p.requires_grad_(False)
            self.bb = get_peft_model(self.bb, LoraConfig(
                r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                target_modules=["qkv", "proj"]))
        elif adapt == "last4":
            for p in self.bb.parameters(): p.requires_grad_(False)
            for blk in self.bb.blocks[-4:]:
                for p in blk.parameters(): p.requires_grad_(True)
            for p in self.bb.norm.parameters(): p.requires_grad_(True)
        elif adapt == "frozen":
            for p in self.bb.parameters(): p.requires_grad_(False)
        self.drop = nn.Dropout(0.2)
        self.cls = nn.Linear(feat, K)
        self.rank = nn.Linear(feat, K - 1)
        self.binh = nn.Linear(feat, 2)

    def feats(self, x, xc):
        f = self.bb(x)
        if self.two: f = torch.cat([f, self.bb(xc)], 1)
        return self.drop(f)

    def forward(self, x, xc=None):
        f = self.feats(x, xc); return self.cls(f), self.rank(f), self.binh(f)
