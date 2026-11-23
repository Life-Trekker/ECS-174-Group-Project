"""
Image colorization (ECS 174 project) -- implements the proposal's Method section.
Expected dataset layout:   mirflickr/images/0/0.jpg, 1.jpg, ...   mirflickr/images/1/..., etc.
Every image is converted to one STANDARD size (--img-size, default 64) before it reaches the network.

Pipeline
  1. `filter`   : drop grayscale / sepia / too-small / unreadable images
  2. data       : resize+crop to the standard size, RGB -> CIELAB, feed L, predict (a, b)
  3. model      : U-Net, ResNet34 (ImageNet-pretrained) 1-channel encoder, 2-channel decoder
  4. `train`    : L1 + Huber loss; add --gan to train the U-Net against a PatchGAN discriminator
  5. `test`     : metrics on the held-out split
  6. `colorize` : colorize any image(s) with a trained checkpoint
"""
# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------
import argparse  # parses command-line arguments (filter / train / test / colorize + flags)
import json  # saves and loads the train/val/test split as a JSON file
import random  # shuffles the image list reproducibly
from functools import partial  # lets us pass an extra fixed argument to a multiprocessing worker
from multiprocessing import Pool  # runs the image filter on several CPU cores at once
from pathlib import Path  # object-oriented, cross-platform file paths

import numpy as np  # array math; images are handled as numpy arrays
import torch  # the deep learning framework
import torch.nn as nn  # neural network layers (Conv2d, BatchNorm2d, ...)
import torch.nn.functional as F  # stateless ops (interpolate, losses, ...)
from PIL import Image  # loads/saves images
from skimage import color  # rgb2lab / lab2rgb color-space conversion
from torch.utils.data import DataLoader, Dataset  # batching + custom dataset base class
from torchvision import models, transforms  # pretrained ResNet + image transforms
from tqdm import tqdm  # progress bars

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DEFAULT_IMG_SIZE = 256  # the standard size every image is forced to; override with --img-size 128 / 256
EXTS = {".jpg", ".jpeg", ".png"}  # file extensions we treat as images
AB_SCALE = 110.0  # a,b channels span roughly [-110, 110]; dividing by this maps them to ~[-1, 1]


# ----------------------------------------------------------------------------
# 1. Filtering (remove grayscale / sepia / tiny images)
# ----------------------------------------------------------------------------
def is_valid_image(path, min_side, gray_chroma=4.0, sepia_conc=0.97, sepia_chroma=30.0):
    """Return True if the image is full color and at least min_side pixels on its shorter side.

    grayscale: mean chroma (sqrt(a^2+b^2)) is ~0
    sepia    : chroma exists but all pixels share (almost) the same hue,
               i.e. the chroma-weighted hue concentration R is ~1.
    Thresholds are heuristics -- eyeball a few rejects and tune if needed.
    """
    try:  # wrap everything: corrupt/unreadable files should simply be rejected

        img = Image.open(path).convert("RGB")  # open the file and force 3-channel RGB

        if min(img.size) < min_side:  # img.size is (width, height); check the shorter side
            return False  # smaller than the standard size -> would need upscaling -> reject
        
        img.thumbnail((96, 96))  # shrink in place to <=96px; makes the color stats much faster
        lab = color.rgb2lab(np.asarray(img))  # convert to CIELAB -> array of shape (H, W, 3)
        a, b = lab[..., 1], lab[..., 2]  # split out the two color channels
        chroma = np.hypot(a, b)  # per-pixel colorfulness = sqrt(a^2 + b^2)
        if chroma.mean() < gray_chroma:  # almost no color anywhere -> it's a grayscale image
            return False  # reject grayscale
        hue = np.arctan2(b, a)  # per-pixel hue angle in radians

        # R = length of the chroma-weighted average hue vector (1.0 means every pixel has the same hue).
        # Numerator: magnitude of the sum of chroma-weighted unit hue vectors.
        # Denominator: total chroma, so R is normalized to [0, 1].
        R = np.hypot((chroma * np.cos(hue)).sum(), (chroma * np.sin(hue)).sum()) / (chroma.sum() + 1e-8)
        if R > sepia_conc and chroma.mean() < sepia_chroma:  # single hue + mild saturation -> sepia tone
            return False  # reject sepia
        return True  # passed every check: it's a real color image

    
    except Exception:  # any error while opening/processing the file
        return False  # treat unreadable images as invalid


def _check(p, min_side):
    """Pool worker: returns the path as a string if valid, else None."""
    return str(p) if is_valid_image(p, min_side) else None  # None entries get discarded later


def cmd_filter(args):
    """`filter` subcommand: scan the images folder and write the list of valid images to a text file."""
    root = Path(args.data)  # the images folder, e.g. mirflickr/images (contains subfolders 0, 1, 2, ...)
    if not root.exists():  # catch typos in --data before doing any work
        raise SystemExit(f"Folder not found: {root}")  # stop with a clear error message
    # If --parts was given (e.g. 0 1 2) search only those numbered subfolders; otherwise search everything.
    roots = [root / s for s in args.parts] if args.parts else [root]
    # Recursively find every image file under each search root; sorted first so the shuffle below is reproducible.
    paths = sorted(p for r in roots for p in r.rglob("*") if p.suffix.lower() in EXTS)
    random.Random(args.seed).shuffle(paths)  # shuffle so the sample is drawn from ALL subfolders
    print(f"Found {len(paths)} images in {len(roots)} location(s), filtering (min side {args.img_size}px)...")
    keep = []  # will hold the paths of images that pass the filter
    worker = partial(_check, min_side=args.img_size)  # fix the min_side argument for every worker call
    with Pool(args.workers) as pool:  # start a pool of worker processes
        # imap streams results in order; chunksize batches work to reduce overhead; tqdm shows progress.
        for r in tqdm(pool.imap(worker, paths, chunksize=64), total=len(paths)):
            if r:  # r is a path string if valid, None otherwise
                keep.append(r)  # remember the valid image
            if args.limit and len(keep) >= args.limit:  # have we collected enough valid images?
                break  # stop early; leaving the `with` block shuts the pool down
    Path(args.out).write_text("\n".join(keep))  # save one valid path per line
    print(f"Kept {len(keep)} valid images -> {args.out}")  # report how many were saved


# ----------------------------------------------------------------------------
# 2. Data: RGB -> CIELAB, L is the input, (a, b) is the target
# ----------------------------------------------------------------------------
class ColorDataset(Dataset):
    """Loads an image and returns (L, ab) tensors ready for the network."""

    def __init__(self, paths, train, img_size):
        self.paths = paths  # list of image file paths for this split
        self.tf = transforms.Compose([  # chain of image transforms applied in order
            transforms.Resize(img_size),  # scale so the SHORTER side equals the standard size (keeps aspect)
            # Training: random square crop (data augmentation). Eval: deterministic center crop.
            transforms.RandomCrop(img_size) if train else transforms.CenterCrop(img_size),
            # Training: random horizontal flip. Eval: identity (nn.Identity can be pickled on Windows, a lambda can't).
            transforms.RandomHorizontalFlip() if train else nn.Identity(),
        ])

    def __len__(self):
        return len(self.paths)  # DataLoader asks how many samples exist

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))  # load image i, force RGB, apply transforms
        lab = color.rgb2lab(np.asarray(img)).astype(np.float32)  # RGB -> Lab, shape (S, S, 3), float32
        # L is in [0, 100]; dividing by 50 and subtracting 1 maps it to [-1, 1].
        # [..., :1] keeps a trailing channel dim of size 1; permute moves it to (1, H, W) for PyTorch.
        L = torch.from_numpy(lab[..., :1] / 50.0 - 1.0).permute(2, 0, 1)
        # a and b divided by 110 -> approximately [-1, 1]; result shape (2, H, W).
        ab = torch.from_numpy(lab[..., 1:] / AB_SCALE).permute(2, 0, 1)
        return L, ab  # input (grayscale info) and target (color info)


def lab_to_rgb(L, ab):
    """L: (B,1,H,W) in [-1,1], ab: (B,2,H,W) in [-1,1]  ->  (B,H,W,3) float RGB in [0,1]."""
    L = (L.detach().cpu().numpy() + 1.0) * 50.0  # undo the normalization: back to [0, 100], as numpy
    ab = ab.detach().cpu().numpy() * AB_SCALE  # undo the normalization: back to ~[-110, 110]
    # Stack channels into (B, 3, H, W), then move channels last -> (B, H, W, 3) as skimage expects.
    lab = np.concatenate([L, ab], axis=1).transpose(0, 2, 3, 1)
    # Convert each image Lab -> RGB and clip to [0, 1] (some Lab colors fall outside the RGB gamut).
    return np.stack([np.clip(color.lab2rgb(x), 0, 1) for x in lab])


def make_splits(list_file, n_train, n_val, n_test, seed):
    """Shuffle the valid image list and cut it into train/val/test lists."""
    paths = [p for p in Path(list_file).read_text().splitlines() if p]  # read paths, skip blank lines
    random.Random(seed).shuffle(paths)  # seeded shuffle so splits are reproducible
    need = n_train + n_val + n_test  # total images requested
    if len(paths) < need:  # fewer valid images than requested?
        f = len(paths) / need  # fraction of the request we can actually satisfy
        # Shrink each split by the same fraction so the proportions stay the same.
        n_train, n_val, n_test = int(n_train * f), int(n_val * f), int(n_test * f)
        print(f"Only {len(paths)} valid images; using {n_train}/{n_val}/{n_test}")
    return (paths[:n_train],  # first chunk -> training
            paths[n_train:n_train + n_val],  # next chunk -> validation
            paths[n_train + n_val:n_train + n_val + n_test])  # next chunk -> test (no overlap)


# ----------------------------------------------------------------------------
# 3. Model: U-Net with a pretrained ResNet34 encoder (1 channel in, 2 channels out)
# ----------------------------------------------------------------------------
class ConvBlock(nn.Sequential):
    """Two (3x3 conv -> batch norm -> ReLU) layers; the basic decoder building block."""

    def __init__(self, i, o):
        super().__init__(  # nn.Sequential runs the layers below in order
            nn.Conv2d(i, o, 3, padding=1, bias=False),  # 3x3 conv, i -> o channels, same spatial size
            nn.BatchNorm2d(o),  # normalize activations (bias unnecessary since BN has its own shift)
            nn.ReLU(inplace=True),  # nonlinearity
            nn.Conv2d(o, o, 3, padding=1, bias=False),  # second 3x3 conv, o -> o channels
            nn.BatchNorm2d(o),  # normalize again
            nn.ReLU(inplace=True),  # nonlinearity again
        )


class Up(nn.Module):
    """One decoder step: upsample, concatenate the encoder skip connection, then convolve."""

    def __init__(self, in_c, skip_c, out_c):
        super().__init__()  # initialize the nn.Module machinery
        # After concatenation the channel count is in_c + skip_c; the block reduces it to out_c.
        self.conv = ConvBlock(in_c + skip_c, out_c)

    def forward(self, x, skip):
        # Enlarge x to the exact spatial size of the skip feature map (bilinear interpolation).
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        # Glue decoder features and encoder features together along the channel axis, then convolve.
        return self.conv(torch.cat([x, skip], dim=1))


class ResNetUNet(nn.Module):
    """Input: (B,1,H,W) L channel.  Output: (B,2,H,W) a,b channels in [-1,1]."""

    def __init__(self, pretrained=True):
        super().__init__()  # initialize nn.Module
        # Load ResNet34; use ImageNet weights if pretrained=True, otherwise random init (used at test time,
        # because the checkpoint will overwrite the weights anyway).
        r = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
        # Original conv1 takes 3 channels (RGB). We need 1 channel (L), so make a new 1-channel conv...
        self.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)  # 7x7 kernel, stride 2, padding 3, like ResNet
        # ...and initialize it with the RGB filters summed into one channel (keeps pretrained knowledge).
        self.conv1.weight.data.copy_(r.conv1.weight.data.sum(dim=1, keepdim=True))
        # Reuse the pretrained batch norm, ReLU and max-pool layers from ResNet unchanged.
        self.bn1, self.relu, self.maxpool = r.bn1, r.relu, r.maxpool
        # Reuse the four pretrained residual stages: these form the encoder.
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

        self.up4 = Up(512, 256, 256)  # decoder: 512ch @1/32 + skip x3 (256ch @1/16) -> 256ch
        self.up3 = Up(256, 128, 128)  # decoder: 256ch + skip x2 (128ch @1/8) -> 128ch
        self.up2 = Up(128, 64, 64)  # decoder: 128ch + skip x1 (64ch @1/4) -> 64ch
        self.up1 = Up(64, 64, 64)  # decoder: 64ch + skip x0 (64ch @1/2) -> 64ch
        # Head: also sees the original L (64+1 channels) at full resolution, refines to 32 channels,
        # 1x1 conv gives the 2 output channels (a, b), tanh squashes to [-1, 1] to match the targets.
        self.head = nn.Sequential(ConvBlock(64 + 1, 32), nn.Conv2d(32, 2, 1), nn.Tanh())

    def forward(self, L):
        x0 = self.relu(self.bn1(self.conv1(L)))  # stem: 64 channels at 1/2 resolution
        x1 = self.layer1(self.maxpool(x0))  # max-pool then stage 1: 64 channels at 1/4
        x2 = self.layer2(x1)  # stage 2: 128 channels at 1/8
        x3 = self.layer3(x2)  # stage 3: 256 channels at 1/16
        x4 = self.layer4(x3)  # stage 4 (bottleneck): 512 channels at 1/32
        d = self.up4(x4, x3)  # upsample bottleneck, merge with x3
        d = self.up3(d, x2)  # upsample, merge with x2
        d = self.up2(d, x1)  # upsample, merge with x1
        d = self.up1(d, x0)  # upsample, merge with x0 (back to 1/2 resolution)
        # Upsample the last 1/2-resolution features to the full input size.
        d = F.interpolate(d, size=L.shape[-2:], mode="bilinear", align_corners=False)
        # Append the raw L channel (sharp detail) and predict the final 2 color channels.
        return self.head(torch.cat([d, L], dim=1))


class PatchDiscriminator(nn.Module):
    """Small CNN that scores overlapping patches of an (L, a, b) image as real/fake."""

    def __init__(self, in_c=3, nf=64):  # in_c=3 because we feed L + a + b together
        super().__init__()  # initialize nn.Module

        def blk(i, o, stride, norm=True):
            # One discriminator block: 4x4 conv (+ optional batch norm) + LeakyReLU.
            layers = [nn.Conv2d(i, o, 4, stride, 1, bias=not norm)]  # skip bias when BN follows
            if norm:  # the very first block conventionally has no normalization
                layers.append(nn.BatchNorm2d(o))
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # LeakyReLU is standard for GAN discriminators
            return layers  # list of layers, unpacked with * below

        self.net = nn.Sequential(
            *blk(in_c, nf, 2, norm=False),  # 3 -> 64 channels, halves resolution, no norm
            *blk(nf, nf * 2, 2),  # 64 -> 128, halves resolution
            *blk(nf * 2, nf * 4, 2),  # 128 -> 256, halves resolution
            *blk(nf * 4, nf * 8, 1),  # 256 -> 512, keeps resolution (stride 1)
            nn.Conv2d(nf * 8, 1, 4, 1, 1),  # final conv -> 1 logit per patch (no sigmoid; BCEWithLogits does it)
        )

    def forward(self, L, ab):
        return self.net(torch.cat([L, ab], dim=1))  # join L and ab into 3 channels -> (B,1,h,w) patch logits


# ----------------------------------------------------------------------------
# 4. Training
# ----------------------------------------------------------------------------
def pixel_losses(pred, ab):
    l1 = F.l1_loss(pred, ab)  # mean absolute error between predicted and true a,b
    huber = F.huber_loss(pred, ab, delta=0.1)  # quadratic for small errors, linear for large ones (robust)
    return l1, huber  # both are tracked; training minimizes their sum


@torch.no_grad()  # no gradients needed during validation -> saves memory and time
def validate(G, loader, dev):
    G.eval()  # eval mode: batch norm uses running stats, no training behavior
    n, l1_sum, hub_sum = 0, 0.0, 0.0  # running totals: sample count and summed losses
    for L, ab in loader:  # iterate over validation batches
        L, ab = L.to(dev), ab.to(dev)  # move the batch to the GPU/CPU device
        l1, hub = pixel_losses(G(L), ab)  # predict colors and compute both losses
        l1_sum += l1.item() * len(L)  # weight by batch size (last batch may be smaller)
        hub_sum += hub.item() * len(L)  # same for Huber
        n += len(L)  # count samples seen
    return l1_sum / n, hub_sum / n  # dataset-wide average of each loss


@torch.no_grad()  # no gradients needed when just making pictures
def save_samples(G, loader, dev, path, n=8):
    G.eval()  # eval mode for the generator
    L, ab = next(iter(loader))  # grab the first validation batch
    L, ab = L[:n].to(dev), ab[:n].to(dev)  # keep only n images and move them to the device
    pred = G(L)  # model's predicted colors
    gray = lab_to_rgb(L, torch.zeros_like(ab))  # a=b=0 gives the plain grayscale version
    fake = lab_to_rgb(L, pred)  # L + predicted ab -> colorized image
    real = lab_to_rgb(L, ab)  # L + true ab -> original image
    # For each of the three groups, concatenate its n images side by side along the width axis.
    rows = [np.concatenate(list(x), axis=1) for x in (gray, fake, real)]
    # Stack the three rows vertically, scale to 0-255 bytes, and save as a PNG grid.
    Image.fromarray((np.concatenate(rows, axis=0) * 255).astype(np.uint8)).save(path)


def cmd_train(args):
    if args.img_size % 32 != 0:  # the ResNet halves the resolution 5 times (2^5 = 32)
        raise SystemExit("--img-size must be a multiple of 32 (e.g. 64, 128, 256)")  # stop with a clear message
    torch.manual_seed(args.seed)  # make weight init / shuffling reproducible
    # Pick the best available device: NVIDIA GPU, then Apple GPU, then CPU.
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {dev}, image size: {args.img_size}x{args.img_size}")  # confirm the setup to the user
    out = Path(args.out)  # output directory for checkpoints and sample images
    out.mkdir(parents=True, exist_ok=True)  # create it (and parents) if missing

    tr, va, te = make_splits(args.list, args.n_train, args.n_val, args.n_test, args.seed)  # split the data
    (out / "splits.json").write_text(json.dumps({"train": tr, "val": va, "test": te}))  # save split for `test`
    # Helper that builds a DataLoader: shuffle and drop the last partial batch only when training;
    # pin_memory speeds up CPU->GPU copies on CUDA.
    mk = lambda p, train: DataLoader(ColorDataset(p, train, args.img_size), batch_size=args.batch_size,
                                     shuffle=train, num_workers=args.workers,
                                     pin_memory=dev.type == "cuda", drop_last=train)
    train_dl, val_dl = mk(tr, True), mk(va, False)  # training loader (augmented) and validation loader

    G = ResNetUNet(pretrained=True).to(dev)  # the generator / colorizer, with pretrained encoder
    # Adam optimizer; beta1=0.5 is the usual GAN setting and works fine for plain training too.
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    if args.gan:  # only build GAN components if requested
        D = PatchDiscriminator().to(dev)  # the discriminator
        optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))  # its own optimizer
        bce = nn.BCEWithLogitsLoss()  # real/fake classification loss (sigmoid built in)

    best = float("inf")  # lowest validation L1 seen so far
    for epoch in range(1, args.epochs + 1):  # one pass over the training set per epoch
        G.train()  # training mode (batch norm updates its statistics)
        bar = tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}")  # progress bar wrapping the loader
        for L, ab in bar:  # iterate over training batches
            L, ab = L.to(dev), ab.to(dev)  # move batch to the device
            pred = G(L)  # forward pass: predict (a, b) from L

            if args.gan:  # --- discriminator step ---
                real_logits = D(L, ab)  # D's opinion of the real color image
                fake_logits = D(L, pred.detach())  # D's opinion of the fake; detach so G isn't updated here
                # D wants real -> 1 and fake -> 0; average the two losses.
                lossD = 0.5 * (bce(real_logits, torch.ones_like(real_logits)) +
                               bce(fake_logits, torch.zeros_like(fake_logits)))
                optD.zero_grad(set_to_none=True)  # clear old gradients
                lossD.backward()  # compute gradients for D
                optD.step()  # update D's weights

            # --- generator (U-Net) step ---
            l1, hub = pixel_losses(pred, ab)  # pixel-wise losses against ground truth
            lossG = l1 + hub  # basic objective: L1 + Huber
            msg = {"l1": f"{l1.item():.4f}", "huber": f"{hub.item():.4f}"}  # text shown in the progress bar
            if args.gan:
                fake_logits = D(L, pred)  # D's opinion of the fake, this time WITH gradients flowing to G
                adv = bce(fake_logits, torch.ones_like(fake_logits))  # G wants D to say "real" (label 1)
                lossG = args.lambda_pix * lossG + adv  # pix2pix-style: heavy pixel loss + adversarial loss
                msg.update(adv=f"{adv.item():.3f}", D=f"{lossD.item():.3f}")  # also show GAN losses
            optG.zero_grad(set_to_none=True)  # clear old generator gradients
            lossG.backward()  # backpropagate through G (and through D's input path if GAN)
            optG.step()  # update the generator's weights
            bar.set_postfix(msg)  # display current losses on the progress bar

        val_l1, val_hub = validate(G, val_dl, dev)  # evaluate on the validation set after each epoch
        print(f"  val L1 {val_l1:.4f} (~{val_l1 * AB_SCALE:.2f} Lab units)  val Huber {val_hub:.4f}")
        save_samples(G, val_dl, dev, out / f"samples_epoch{epoch:02d}.png")  # visual check of quality
        ckpt = {"model": G.state_dict(), "epoch": epoch, "img_size": args.img_size}  # weights + size they expect
        torch.save(ckpt, out / "last.pt")  # always save the latest weights
        if val_l1 < best:  # new best validation score?
            best = val_l1  # remember it
            torch.save(ckpt, out / "best.pt")  # keep the best weights
    print(f"Done. Best val L1 = {best:.4f}")  # final summary


# ----------------------------------------------------------------------------
# 5. Test-set evaluation
# ----------------------------------------------------------------------------
def load_model(ckpt, dev):
    data = torch.load(ckpt, map_location=dev)  # load the checkpoint dictionary from disk
    G = ResNetUNet(pretrained=False).to(dev)  # build the architecture; no need to download ImageNet weights
    G.load_state_dict(data["model"])  # copy the trained weights into the model
    img_size = data.get("img_size", DEFAULT_IMG_SIZE)  # the standard size this model was trained at
    return G.eval(), img_size  # eval mode model plus its training image size


@torch.no_grad()  # no gradients during testing
def cmd_test(args):
    from skimage.metrics import peak_signal_noise_ratio as psnr  # image-quality metric (higher is better)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available
    G, img_size = load_model(args.ckpt, dev)  # load the trained colorizer and the size it expects
    splits = Path(args.ckpt).parent / "splits.json"  # the split file saved next to the checkpoint
    if splits.exists():  # reuse the exact split used during training (avoids test-set leakage)
        test_paths = json.loads(splits.read_text())["test"]
    else:  # otherwise rebuild the split from the same seed
        test_paths = make_splits(args.list, args.n_train, args.n_val, args.n_test, args.seed)[2]
    dl = DataLoader(ColorDataset(test_paths, False, img_size), batch_size=args.batch_size,
                    num_workers=args.workers)  # no augmentation for testing

    l1s, hubs, psnrs = [], [], []  # per-batch (or per-image for PSNR) metric lists
    for L, ab in tqdm(dl, desc="test"):  # iterate over test batches
        L, ab = L.to(dev), ab.to(dev)  # move to device
        pred = G(L)  # predict colors
        l1, hub = pixel_losses(pred, ab)  # pixel losses on the batch
        l1s.append(l1.item()); hubs.append(hub.item())  # record them
        # Convert predicted and true images to RGB and compare each pair with PSNR.
        for f, r in zip(lab_to_rgb(L, pred), lab_to_rgb(L, ab)):
            psnrs.append(psnr(r, f, data_range=1.0))  # reference first, then estimate; pixels span [0, 1]
    print(f"Test images: {len(test_paths)} at {img_size}x{img_size}")  # how many images were evaluated
    print(f"L1 (normalized): {np.mean(l1s):.4f}  (~{np.mean(l1s) * AB_SCALE:.2f} Lab a/b units)")  # avg L1
    print(f"Huber:           {np.mean(hubs):.4f}")  # avg Huber
    print(f"RGB PSNR:        {np.mean(psnrs):.2f} dB")  # avg PSNR


# ----------------------------------------------------------------------------
# 6. Colorize arbitrary images (keeps the original L at full resolution)
# ----------------------------------------------------------------------------
@torch.no_grad()  # inference only
def cmd_colorize(args):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU if available
    G, img_size = load_model(args.ckpt, dev)  # load the trained model and its standard size
    out_dir = Path(args.out_dir)  # where results are written
    out_dir.mkdir(parents=True, exist_ok=True)  # create the folder if needed
    for p in args.inputs:  # process each input image
        img = Image.open(p).convert("RGB")  # open and force RGB (works for grayscale files too)
        w, h = img.size  # remember the original width and height
        L_full = color.rgb2lab(np.asarray(img))[..., 0]  # full-resolution L channel, shape (H, W), in [0, 100]
        # Make a standard-size grayscale copy (the network's input size); back to 3-channel RGB for rgb2lab.
        small = np.asarray(img.convert("L").resize((img_size, img_size), Image.BICUBIC).convert("RGB"))
        # Take its L channel, normalize to [-1, 1], add batch and channel dims -> (1,1,S,S), send to device.
        L_small = torch.from_numpy(color.rgb2lab(small)[..., 0] / 50.0 - 1.0).float()[None, None].to(dev)
        ab = G(L_small)  # predict the two color channels at the standard size
        # Upsample predicted colors to the original size, drop the batch dim, move to numpy, undo normalization.
        ab = F.interpolate(ab, size=(h, w), mode="bilinear", align_corners=False)[0].cpu().numpy() * AB_SCALE
        # Combine the ORIGINAL sharp L with the predicted ab -> (3, H, W), then channels-last -> (H, W, 3).
        lab = np.concatenate([L_full[None], ab], axis=0).transpose(1, 2, 0)
        # Convert Lab -> RGB, clip to valid range, scale to 0-255 bytes.
        rgb = (np.clip(color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8)
        dest = out_dir / f"{Path(p).stem}_colorized.png"  # output filename based on the input name
        Image.fromarray(rgb).save(dest)  # write the colorized image to disk
        print("saved", dest)  # confirm to the user


# ----------------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()  # top-level argument parser
    sub = ap.add_subparsers(dest="cmd", required=True)  # one subcommand is mandatory: filter/train/test/colorize

    f = sub.add_parser("filter")  # `python colorize.py filter ...`
    f.add_argument("--data", default="mirflickr/images")  # folder that contains the numbered subfolders 0, 1, 2...
    f.add_argument("--out", default="valid_images.txt")  # where to write the list of valid image paths
    f.add_argument("--workers", type=int, default=4)  # number of parallel processes
    # Optional list of numbered subfolders of --data to search, e.g. --parts 0 1 2 (default: search everything).
    f.add_argument("--parts", nargs="*", default=None, help="only these subfolders of --data, e.g. --parts 0 1 2")
    # Stop once this many valid images are found (70k covers 50k/5k/5k plus a margin); 0 means no limit.
    f.add_argument("--limit", type=int, default=70_000, help="stop after this many valid images (0 = no limit)")
    f.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)  # images smaller than this are rejected
    f.add_argument("--seed", type=int, default=0)  # seed for the shuffle so the sample is reproducible
    f.set_defaults(fn=cmd_filter)  # remember which function handles this subcommand

    def add_data_args(p):
        """Arguments shared by `train` and `test`."""
        p.add_argument("--list", default="valid_images.txt")  # list of valid images from `filter`
        p.add_argument("--n-train", type=int, default=50_000)  # training set size
        p.add_argument("--n-val", type=int, default=5_000)  # validation set size
        p.add_argument("--n-test", type=int, default=5_000)  # test set size
        p.add_argument("--batch-size", type=int, default=32)  # images per batch (lower if you run out of GPU memory)
        p.add_argument("--workers", type=int, default=4)  # DataLoader worker processes (use 0 if Windows errors)
        p.add_argument("--seed", type=int, default=0)  # random seed for shuffling and init

    t = sub.add_parser("train")  # `python colorize.py train ...`
    add_data_args(t)  # attach the shared data arguments
    t.add_argument("--out", default="runs/unet")  # directory for checkpoints and sample images
    t.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)  # standard size; multiple of 32
    t.add_argument("--epochs", type=int, default=20)  # number of passes over the training set
    t.add_argument("--lr", type=float, default=2e-4)  # Adam learning rate
    t.add_argument("--gan", action="store_true", help="train U-Net as GAN generator (adds adversarial loss)")  # flag
    t.add_argument("--lambda-pix", type=float, default=100.0, help="weight of L1+Huber when --gan")  # GAN balance
    t.set_defaults(fn=cmd_train)  # handler for `train`

    e = sub.add_parser("test")  # `python colorize.py test ...`
    add_data_args(e)  # attach the shared data arguments
    e.add_argument("--ckpt", required=True)  # path to the trained checkpoint (.pt)
    e.set_defaults(fn=cmd_test)  # handler for `test`

    c = sub.add_parser("colorize")  # `python colorize.py colorize ...`
    c.add_argument("--ckpt", required=True)  # path to the trained checkpoint (.pt)
    c.add_argument("--inputs", nargs="+", required=True)  # one or more image files to colorize
    c.add_argument("--out-dir", default="results")  # folder for the colorized outputs
    c.set_defaults(fn=cmd_colorize)  # handler for `colorize`

    args = ap.parse_args()  # parse sys.argv
    args.fn(args)  # call the chosen subcommand's function with the parsed arguments


if __name__ == "__main__":  # only run main() when executed as a script, not when imported
    main()