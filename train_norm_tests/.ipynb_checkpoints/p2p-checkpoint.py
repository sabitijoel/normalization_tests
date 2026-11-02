#!/usr/bin/env python
# coding: utf-8

import os
import math
import cv2
import random
import shutil
import numpy as np
from glob import glob
from typing import Optional, Tuple
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.checkpoint as checkpoint
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import spectral_norm

import torchvision.transforms as transforms
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torchvision.utils import save_image
from torchvision.transforms import Resize, InterpolationMode

import albumentations as A
from albumentations.pytorch import ToTensorV2


from einops import rearrange
from easydict import EasyDict as edict
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============================
# Reproducibility
# ============================

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================
# Normalization factory
# =============================================================
# All four options carry per-channel affine parameters, so the cells of the
# study differ only in how the statistics are pooled -- not in capacity.
#
#   bn -> stats over (N, H, W) per channel
#   in -> stats over (H, W) per sample per channel
#   ln -> stats over (C, H, W) per sample          == GroupNorm(1, C)
#   gn -> stats over (C/g, H, W) per sample        == GroupNorm(g, C)

GN_GROUPS = 32


def make_norm(kind, channels):
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "in":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "ln":
        return nn.GroupNorm(1, channels)
    if kind == "gn":
        return nn.GroupNorm(min(GN_GROUPS, channels), channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


# ============================
# BasicBlock  (discriminator)
# ============================
class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, norm_kind="bn",
                 kernel_size=4, stride=2, padding=1, norm=True, sn=False):
        super().__init__()
        conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.conv = spectral_norm(conv) if sn else conv
        self.isn = make_norm(norm_kind, outplanes) if norm else None
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)

        if self.isn is not None:
            fx = self.isn(fx)

        fx = self.lrelu(fx)
        return fx



# ============================
# Discriminator
# ============================
class Discriminator(nn.Module):
    """Basic Discriminator"""
    def __init__(self, norm_kind="bn", sn=False):
        super().__init__()
        self.block1 = BasicBlock(4, 64, norm_kind, norm=False, sn=sn)
        self.block2 = BasicBlock(64, 128, norm_kind, sn=sn)
        self.block3 = BasicBlock(128, 256, norm_kind, sn=sn)
        self.block4 = BasicBlock(256, 512, norm_kind, sn=sn)
        final = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        self.block5 = spectral_norm(final) if sn else final

    def forward(self, x, cond):
        # blocks forward
        x = torch.cat([x, cond], dim=1)
        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)

        return fx
# =============================================================
# Encoder Block
# =============================================================
class EncoderBlock(nn.Module):
    """Encoder block"""
    def __init__(self, inplanes, outplanes, norm_kind="bn",
                 kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)

        self.bn = make_norm(norm_kind, outplanes) if norm else None

    def forward(self, x):
        x = self.lrelu(x)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


# =============================================================
# Decoder Block
# =============================================================
class DecoderBlock(nn.Module):
    """Decoder block"""
    def __init__(self, inplanes, outplanes, norm_kind="bn",
                 kernel_size=4, stride=2, padding=1, dropout=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.deconv = nn.ConvTranspose2d(inplanes, outplanes, kernel_size, stride, padding)
        self.bn = make_norm(norm_kind, outplanes)
        self.dropout = nn.Dropout2d(0.5, inplace=True) if dropout else None

    def forward(self, x):
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


# ===========
# Generator
# ===========
class Generator(nn.Module):
    def __init__(self, norm_kind="bn"):
        super().__init__()

        self.encoder1 = nn.Conv2d(1, 64, 4, 2, 1)
        self.encoder2 = EncoderBlock(64, 128, norm_kind)
        self.encoder3 = EncoderBlock(128, 256, norm_kind)
        self.encoder4 = EncoderBlock(256, 512, norm_kind)
        self.encoder5 = EncoderBlock(512, 512, norm_kind)
        self.encoder6 = EncoderBlock(512, 512, norm_kind)
        self.encoder7 = EncoderBlock(512, 512, norm_kind)
        self.encoder8 = EncoderBlock(512, 512, norm_kind, norm=False)  # 1x1 bottleneck

        self.decoder8 = DecoderBlock(512, 512, norm_kind, dropout=True)
        self.decoder7 = DecoderBlock(1024, 512, norm_kind, dropout=True)
        self.decoder6 = DecoderBlock(1024, 512, norm_kind, dropout=True)
        self.decoder5 = DecoderBlock(1024, 512, norm_kind)
        self.decoder4 = DecoderBlock(1024, 256, norm_kind)
        self.decoder3 = DecoderBlock(512, 128, norm_kind)
        self.decoder2 = DecoderBlock(256, 64, norm_kind)
        self.decoder1 = nn.ConvTranspose2d(128, 3, 4, 2, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)
        e8 = self.encoder8(e7)

        d8 = self.decoder8(e8)
        d8 = torch.cat([d8, e7], dim=1)

        d7 = self.decoder7(d8)
        d7 = torch.cat([d7, e6], dim=1)

        d6 = self.decoder6(d7)
        d6 = torch.cat([d6, e5], dim=1)

        d5 = self.decoder5(d6)
        d5 = torch.cat([d5, e4], dim=1)

        d4 = self.decoder4(d5)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.decoder3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.decoder2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        out = self.decoder1(d2)
        return torch.tanh(out)


        
# ============================
# Dataset
# ============================

class MapDataset(Dataset):
    def __init__(self, root_gray_dir, root_jpg_dir, paired_transform, input_transform, mask_transform):
        self.root_gray_dir = root_gray_dir
        self.root_jpg_dir = root_jpg_dir
        self.paired_transform = paired_transform
        self.input_transform = input_transform
        self.mask_transform = mask_transform

        self.grays = [f for f in sorted(os.listdir(root_gray_dir))
                      if os.path.isfile(os.path.join(root_gray_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.jpgs = [f for f in sorted(os.listdir(root_jpg_dir))
                     if os.path.isfile(os.path.join(root_jpg_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        self.grays = self.grays[:min(len(self.grays), len(self.jpgs))]
        self.jpgs = self.jpgs[:min(len(self.grays), len(self.jpgs))]

    def __len__(self):
        return len(self.grays)

    def __getitem__(self, index):
        gray_file = self.grays[index]
        jpg_file = self.jpgs[index]
        gray_path = os.path.join(self.root_gray_dir, gray_file)
        jpg_path = os.path.join(self.root_jpg_dir, jpg_file)
        gray_image_pil = Image.open(gray_path).convert('L')
        gray_image_np = np.array(gray_image_pil)
        jpg_image_pil = Image.open(jpg_path).convert('RGB')
        jpg_image_np = np.array(jpg_image_pil) 
        augmentations = self.paired_transform(image=jpg_image_np, image0=gray_image_np)
        jpg_image_aug = augmentations["image"]
        gray_image_aug = augmentations["image0"]

        gray_tensor = self.input_transform(image=gray_image_aug)["image"]
        jpg_tensor = self.mask_transform(image=jpg_image_aug)["image"]

        return gray_tensor, jpg_tensor


# ============================
# Utility Functions
# ============================


def save_checkpoint(model, optimizer, epoch, checkpoint_folder, filename="checkpoint"):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    os.makedirs(checkpoint_folder, exist_ok=True)
    full_filename = os.path.join(checkpoint_folder, f"{filename}_epoch_{epoch}.pth")
    torch.save(checkpoint, full_filename)
    print(f"Checkpoint saved to {full_filename}")

# ============================
# Training & Validation Functions
# ============================

def backpropagation(model, scaler, loss, optimizer):
    model.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()


#Declare lists to store the losses
train_losses_G, train_losses_D, val_losses_G, val_losses_D = [], [], [], []
# Per-epoch instrumentation, index-aligned with the loss lists above.
epoch_records = []


# ----------------------------
# Complexity / provenance
# ----------------------------

def count_params(model):
    """Trainable parameters, plus non-trainable buffers (BN running stats)."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    buffers = sum(b.numel() for b in model.buffers())
    return {"total": total, "trainable": trainable, "buffers": buffers}


def measure_flops(model, x):
    """FLOPs for one forward pass. mul-add counts as 2; divide by 2 for MACs."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
        was_training = model.training
        model.eval()
        counter = FlopCounterMode(display=False)
        with counter, torch.no_grad():
            model(x)
        if was_training:
            model.train()
        return counter.get_total_flops()
    except Exception as exc:
        print(f"FLOP count unavailable: {exc}", flush=True)
        return None


def measure_latency(model, x, warmup=20, runs=50):
    """Mean/std forward latency in ms at the given batch size."""
    if not torch.cuda.is_available():
        return None
    was_training = model.training
    model.eval()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        torch.cuda.synchronize()
        for _ in range(runs):
            starter.record()
            model(x)
            ender.record()
            torch.cuda.synchronize()
            times.append(starter.elapsed_time(ender))
    if was_training:
        model.train()
    mean = sum(times) / len(times)
    var = sum((t - mean) ** 2 for t in times) / len(times)
    return {"mean_ms": mean, "std_ms": var ** 0.5, "batch": x.shape[0]}


def source_fingerprint():
    """SHA of this file, so a checkpoint can be tied to the exact code."""
    try:
        import hashlib
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except Exception:
        return None


def validate_fn(gen, disc, val_loader, l1_loss, bce_loss, DEVICE, L1_LAMBDA):
    gen.eval()
    disc.eval()
    total_D_loss, total_G_loss = 0, 0
    total_l1, psnr_sum, n_images = 0.0, 0.0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            y_fake = gen(x)
            D_real = disc(x, y)
            D_fake = disc(x, y_fake)

            # BCE loss calculations
            real_labels = torch.ones_like(D_real)
            fake_labels = torch.zeros_like(D_fake)

            D_real_loss = bce_loss(D_real, real_labels)
            D_fake_loss = bce_loss(D_fake, fake_labels)
            D_loss = (D_real_loss + D_fake_loss) / 2

            # Generator loss with BCE
            G_fake_loss = bce_loss(D_fake, real_labels)  # Trick discriminator to think fake is real
            L1 = l1_loss(y_fake, y) * L1_LAMBDA
            G_loss = G_fake_loss + L1

            total_D_loss += D_loss.item()
            total_G_loss += G_loss.item()
            total_l1 += (L1 / L1_LAMBDA).item()

            # PSNR on [0,1] RGB, per image. Reported as a cheap training-time
            # signal only -- the paper's SSIM / Delta E come from the existing
            # post-hoc harness so the conventions stay consistent.
            pred = (y_fake.clamp(-1, 1) + 1) / 2
            targ = (y + 1) / 2
            mse = ((pred - targ) ** 2).mean(dim=[1, 2, 3])
            psnr_sum += (-10 * torch.log10(mse + 1e-12)).sum().item()
            n_images += x.shape[0]

    gen.train()
    disc.train()
    n_batches = len(val_loader)
    return {
        "val_G": total_G_loss / n_batches,
        "val_D": total_D_loss / n_batches,
        "val_L1": total_l1 / n_batches,
        "val_PSNR": psnr_sum / max(n_images, 1),
    }


def train_fn(disc, gen, loader, val_loader, opt_disc, opt_gen, l1_loss, bce_loss,
             g_scaler, d_scaler, DEVICE, L1_LAMBDA):
    global train_losses_G, train_losses_D, val_losses_G, val_losses_D, epoch_records
    loop = tqdm(loader, leave=True)
    total_D_loss, total_G_loss = 0, 0
    D_ITERS = 1
    G_ITERS = 1

    n_seen = 0
    t_train_start = time.perf_counter()

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        n_seen += x.shape[0]

        # ========================
        # Train Discriminator (D_ITERS times)
        # ========================
        for _ in range(D_ITERS):
            with torch.amp.autocast('cuda'):
                y_fake = gen(x)

                D_real = disc(x, y)
                D_fake = disc(x, y_fake.detach())

                real_labels = torch.ones_like(D_real)
                fake_labels = torch.zeros_like(D_fake)

                D_real_loss = bce_loss(D_real, real_labels)
                D_fake_loss = bce_loss(D_fake, fake_labels)
                D_loss = (D_real_loss + D_fake_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            total_D_loss += D_loss.item()

        # ========================
        # Train Generator (G_ITERS times)
        # ========================
        for _ in range(G_ITERS):
            with torch.amp.autocast('cuda'):
                y_fake = gen(x)
                D_fake = disc(x, y_fake)

                real_labels = torch.ones_like(D_fake)
                G_fake_loss = bce_loss(D_fake, real_labels)
                L1 = l1_loss(y_fake, y) * L1_LAMBDA

                G_loss = G_fake_loss + L1

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            total_G_loss += G_loss.item()

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    train_time = time.perf_counter() - t_train_start

    avg_train_G_loss = total_G_loss / (len(loader) * G_ITERS)
    avg_train_D_loss = total_D_loss / (len(loader) * D_ITERS)
    train_losses_G.append(avg_train_G_loss)
    train_losses_D.append(avg_train_D_loss)

    t_val_start = time.perf_counter()
    val = validate_fn(gen, disc, val_loader, l1_loss, bce_loss, DEVICE, L1_LAMBDA)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    val_time = time.perf_counter() - t_val_start

    val_losses_G.append(val["val_G"])
    val_losses_D.append(val["val_D"])

    record = {
        "train_G": avg_train_G_loss,
        "train_D": avg_train_D_loss,
        "val_G": val["val_G"],
        "val_D": val["val_D"],
        "val_L1": val["val_L1"],
        "val_PSNR": val["val_PSNR"],
        "train_time_s": train_time,
        "val_time_s": val_time,
        "epoch_time_s": train_time + val_time,
        "train_throughput_img_s": n_seen / train_time if train_time > 0 else None,
    }
    if torch.cuda.is_available():
        record["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
        record["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
    epoch_records.append(record)

    print(f"Train G Loss: {avg_train_G_loss:.4f}, Train D Loss: {avg_train_D_loss:.4f}")
    print(f"Val G Loss: {val['val_G']:.4f}, Val D Loss: {val['val_D']:.4f}, "
          f"Val PSNR: {val['val_PSNR']:.3f} dB")
    print(f"epoch {train_time + val_time:.1f}s "
          f"(train {train_time:.1f}s, val {val_time:.1f}s, "
          f"{record['train_throughput_img_s']:.1f} img/s)", flush=True)

    return avg_train_D_loss, avg_train_G_loss, val["val_G"], val["val_D"]


# ============================
# Main Function
# ============================

def main(SEED, GEN_NORM, DISC_NORM, NUM_EPOCHS=200, SAVE_EVERY=5, BATCH_SIZE=8, DISC_SN=False):
    global train_losses_G, train_losses_D, val_losses_G, val_losses_D, epoch_records
    train_losses_G, train_losses_D, val_losses_G, val_losses_D = [], [], [], []
    epoch_records = []

    # Run name: g<gen norm>_d<disc norm>[_sn][_bs<N>] -- each variant gets its own folder
    CELL = f"g{GEN_NORM}_d{DISC_NORM}"
    if DISC_SN:
        CELL += "_sn"
    if BATCH_SIZE != 8:
        CELL += f"_bs{BATCH_SIZE}"
    print(f"=== cell {CELL} | seed {SEED} | {NUM_EPOCHS} epochs | batch {BATCH_SIZE} "
          f"| disc spectral norm {'on' if DISC_SN else 'off'} ===", flush=True)

    seed_everything(SEED)
    g = torch.Generator()
    g.manual_seed(SEED)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", DEVICE)
    IMAGE_SIZE = 256
    L1_LAMBDA = 100

    # Directories
    DATA_ROOT = "/cbio/projects/037/jsabiti/datasets/ISIC"
    train_gray_dir     = f"{DATA_ROOT}/ISIC2018_Task3_Training_Input/gray"
    train_color_dir    = f"{DATA_ROOT}/ISIC2018_Task3_Training_Input/rgb"
    validate_gray_dir  = f"{DATA_ROOT}/ISIC2018_Task3_Validation_Input/gray"
    validate_color_dir = f"{DATA_ROOT}/ISIC2018_Task3_Validation_Input/rgb"

    checkpoint_folder = f"{DATA_ROOT}/checkpoints/norm_study/{CELL}/seed_{SEED}"
    save_dir = checkpoint_folder
    os.makedirs(save_dir, exist_ok=True)

    # Hyperparameters -- identical across every cell of the grid
    LEARNING_RATE_GEN = 1e-4
    LEARNING_RATE_DISC = 2e-4
    NUM_WORKERS = 4
    LOAD_MODEL = False
    SAVE_MODEL = True
    CHECKPOINT_GEN = f"{CELL}_seed{SEED}"

    GRAY_MEAN = [0.5]
    GRAY_STD = [0.5]
    RGB_STD = [0.5, 0.5, 0.5]
    RGB_MEAN = [0.5, 0.5, 0.5]

    transform_paired = A.Compose(
        [
            A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
        ],
        additional_targets={"image0": "image"}
    )

    transform_input_final = A.Compose([
        A.ColorJitter(brightness=0.1, contrast=0.1, p=0.3),
        A.Normalize(mean=GRAY_MEAN, std=GRAY_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

    transform_mask_final = A.Compose([
        A.Normalize(mean=RGB_MEAN, std=RGB_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

    # Deterministic validation: resize + normalize only, no random augmentation.
    val_transform_paired = A.Compose(
        [A.Resize(width=IMAGE_SIZE, height=IMAGE_SIZE)],
        additional_targets={"image0": "image"}
    )
    val_transform_input = A.Compose([
        A.Normalize(mean=GRAY_MEAN, std=GRAY_STD, max_pixel_value=255.0),
        ToTensorV2()
    ])

    train_dataset = MapDataset(
        train_gray_dir, train_color_dir, paired_transform=transform_paired,
        input_transform=transform_input_final, mask_transform=transform_mask_final
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g,
        persistent_workers=NUM_WORKERS > 0,
    )

    val_dataset = MapDataset(
        validate_gray_dir, validate_color_dir,
        paired_transform=val_transform_paired,
        input_transform=val_transform_input,
        mask_transform=transform_mask_final
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, worker_init_fn=seed_worker, generator=g,
        persistent_workers=NUM_WORKERS > 0,
    )

    print(f"train images: {len(train_dataset)}, val images: {len(val_dataset)}", flush=True)

    # Models
    gen = Generator(GEN_NORM).to(DEVICE)
    disc = Discriminator(DISC_NORM, sn=DISC_SN).to(DEVICE)

    # ---- one-off complexity measurements, before training perturbs the cache ----
    probe = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    gen_flops = measure_flops(gen, probe)
    gen_latency = measure_latency(gen, probe)
    del probe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    meta = {
        "cell": CELL, "gen_norm": GEN_NORM, "disc_norm": DISC_NORM, "seed": SEED,
        "disc_spectral_norm": DISC_SN,
        "epochs_requested": NUM_EPOCHS, "save_every": SAVE_EVERY,
        "batch_size": BATCH_SIZE, "image_size": IMAGE_SIZE,
        "lr_gen": LEARNING_RATE_GEN, "lr_disc": LEARNING_RATE_DISC,
        "l1_lambda": L1_LAMBDA, "num_workers": NUM_WORKERS,
        "gn_groups": GN_GROUPS,
        "train_images": len(train_dataset), "val_images": len(val_dataset),
        "gen_params": count_params(gen), "disc_params": count_params(disc),
        "gen_flops_fwd_b1": gen_flops,
        "gen_macs_fwd_b1": gen_flops / 2 if gen_flops else None,
        "gen_latency_b1": gen_latency,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "hostname": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "source_sha256_16": source_fingerprint(),
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(save_dir, f"meta_{CELL}_seed{SEED}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    flops_str = f"{gen_flops/1e9:.2f}G" if gen_flops else "n/a"
    print(f"gen params {meta['gen_params']['total']/1e6:.2f}M, "
          f"disc params {meta['disc_params']['total']/1e6:.2f}M, "
          f"gen FLOPs {flops_str}", flush=True)

    # Optimizers & Losses
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE_GEN, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE_DISC, betas=(0.5, 0.999))

    L1_LOSS = nn.L1Loss()
    BCE_LOSS = nn.BCEWithLogitsLoss()

    g_scaler = torch.amp.GradScaler()
    d_scaler = torch.amp.GradScaler()

    history_path = os.path.join(save_dir, f"history_{CELL}_seed{SEED}.json")
    t_run_start = time.perf_counter()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        train_fn(disc, gen, train_loader, val_loader, opt_disc, opt_gen,
                 L1_LOSS, BCE_LOSS, g_scaler, d_scaler, DEVICE, L1_LAMBDA)

        if SAVE_MODEL and ((epoch + 1) % SAVE_EVERY == 0 or (epoch + 1) == NUM_EPOCHS):
            save_checkpoint(gen, opt_gen, epoch + 1, checkpoint_folder, filename=CHECKPOINT_GEN)

        # Rewrite each epoch so a killed job still leaves usable curves.
        with open(history_path, "w") as f:
            json.dump({
                "cell": CELL, "gen_norm": GEN_NORM, "disc_norm": DISC_NORM,
                "disc_spectral_norm": DISC_SN,
                "seed": SEED, "epochs_done": epoch + 1,
                "elapsed_s": time.perf_counter() - t_run_start,
                "epochs": epoch_records,
                "train_G": train_losses_G, "train_D": train_losses_D,
                "val_G": val_losses_G, "val_D": val_losses_D,
            }, f, indent=2)

    # ---- run summary ----
    total_s = time.perf_counter() - t_run_start
    ep_times = [r["epoch_time_s"] for r in epoch_records]
    meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["epochs_completed"] = len(epoch_records)
    meta["total_train_time_s"] = total_s
    meta["total_train_time_h"] = total_s / 3600
    meta["mean_epoch_time_s"] = sum(ep_times) / len(ep_times) if ep_times else None
    if torch.cuda.is_available():
        meta["peak_alloc_GB"] = torch.cuda.max_memory_allocated() / 1e9
        meta["peak_reserved_GB"] = torch.cuda.max_memory_reserved() / 1e9
    meta["best_val_PSNR"] = max((r["val_PSNR"] for r in epoch_records), default=None)
    with open(os.path.join(save_dir, f"meta_{CELL}_seed{SEED}.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # Plot losses
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_D, label="Train Loss (D)")
    plt.plot(val_losses_D, label="Validation Loss (D)")
    plt.legend()
    plt.title(f"Discriminator Loss -- {CELL} seed {SEED}")

    plt.subplot(1, 2, 2)
    plt.plot(train_losses_G, label="Train Loss (G)")
    plt.plot(val_losses_G, label="Validation Loss (G)")
    plt.legend()
    plt.title(f"Generator Loss -- {CELL} seed {SEED}")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{CELL}_seed{SEED}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"done: {len(epoch_records)} epochs in {total_s/3600:.2f} h", flush=True)


if __name__ == "__main__":
    NORM_CHOICES = ["bn", "in", "ln", "gn", "none"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gen-norm", choices=NORM_CHOICES, required=True)
    parser.add_argument("--disc-norm", choices=NORM_CHOICES, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--disc-sn", action="store_true",
                        help="apply spectral normalization to every discriminator conv")

    args = parser.parse_args()
    main(args.seed, args.gen_norm, args.disc_norm, args.epochs, args.save_every,
         args.batch_size, args.disc_sn)