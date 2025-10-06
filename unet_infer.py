import argparse
import os
from typing import List, Optional

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel

from diff_unet import UNetModel
from style_encoder.model import MobileNetV3Style
from feature_extractor import ImageEncoder


def device_from_str(dev: Optional[str]) -> torch.device:
	if dev is None:
		return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if dev.startswith("cuda") and not torch.cuda.is_available():
		return torch.device("cpu")
	return torch.device(dev)


def white_pad_to_width(img: Image.Image, target_h: int = 64, target_w: int = 256) -> Image.Image:
	# keep aspect by resizing height to target_h, then pad/crop width to target_w
	img = img.convert("RGB")
	w, h = img.size
	if h != target_h:
		new_w = int(round(w * (target_h / h)))
		img = img.resize((max(1, new_w), target_h), Image.BICUBIC)
		w, h = img.size
	# pad or center-crop to target_w
	if w == target_w:
		return img
	if w < target_w:
		# white background
		canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
		# left align for handwriting-style consistency
		canvas.paste(img, (0, 0))
		return canvas
	# if wider, shrink a bit rather than hard crop
	while w > target_w:
		w = max(target_w, w - 20)
		img = img.resize((w, target_h), Image.BICUBIC)
	if img.size[0] < target_w:
		canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
		canvas.paste(img, (0, 0))
		return canvas
	return img

def load_state_safely(module: nn.Module, state_dict_path: str, device: torch.device):
    sd = torch.load(state_dict_path, map_location=device)

    # handle checkpoints saved with wrappers
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    # strip possible prefixes like 'module.' or 'model.'
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            new_k = k[len("module."):]
        elif k.startswith("model."):
            new_k = k[len("model."):]
        else:
            new_k = k

        # ✅ Move tensors to the correct device
        if isinstance(v, torch.Tensor):
            new_sd[new_k] = v.to(device)
        else:
            new_sd[new_k] = v

    missing, unexpected = module.load_state_dict(new_sd, strict=False)

    # ✅ PRINT actual names to inspect
    if missing:
        print(f"\n[warn] Missing keys when loading {state_dict_path}: {len(missing)}")
        print("  → First 15 missing keys:", missing[:15])
    if unexpected:
        print(f"[warn] Unexpected keys when loading {state_dict_path}: {len(unexpected)}")
        print("  → First 15 unexpected keys:", unexpected[:15])

    return missing, unexpected

def build_transforms():
	return transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
	])


def prepare_style_batch(style_paths: List[str], img_h: int, img_w: int) -> torch.Tensor:
	tfm = build_transforms()
	imgs = []
	for p in style_paths:
		img = Image.open(p).convert("RGB")
		img = white_pad_to_width(img, target_h=img_h, target_w=img_w)
		imgs.append(tfm(img))
	batch = torch.stack(imgs, dim=0)
	return batch


class SimpleArgs:
	# minimal shim to satisfy UNetModel's expectations
	def __init__(self, interpolation: bool = False, mix_rate: Optional[float] = None):
		self.interpolation = interpolation
		self.mix_rate = mix_rate


@torch.inference_mode()
def sample_image(
	unet: nn.Module,
	tokenizer: CanineTokenizer,
	text: str,
	style_feats: Optional[torch.Tensor],
	vae: Optional[AutoencoderKL],
	scheduler: DDIMScheduler,
	device: torch.device,
	img_h: int,
	img_w: int,
	channels: int,
	steps: int,
	latent: bool,
):
	# text conditioning
	tok = tokenizer([text], padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(device)

	# init noise
	if latent:
		x = torch.randn((1, 4, img_h // 8, img_w // 8), device=device)
	else:
		x = torch.randn((1, channels, img_h, img_w), device=device)

	scheduler.set_timesteps(steps)
	for t in scheduler.timesteps:
		t_b = torch.full((x.size(0),), t.item(), device=device, dtype=torch.long)
		noise_pred = unet(x, timesteps=t_b, context=tok, y=None, style_extractor=style_feats)
		step_out = scheduler.step(noise_pred, t, x)
		x = step_out.prev_sample

	if latent:
		latents = (1 / 0.18215) * x
		image = vae.decode(latents).sample
		image = (image / 2 + 0.5).clamp(0, 1)  # [0,1]
		image = (image * 255).round().byte()
		return image
	else:
		image = (x.clamp(-1, 1) + 1) / 2
		image = (image * 255).round().byte()
		return image


def main():
	parser = argparse.ArgumentParser(description="UNetModel inference: text + style")
	parser.add_argument("--text", type=str, required=True, help="Text to render")
	parser.add_argument("--unet_ckpt", type=str, required=True, help="Path to UNet checkpoint (ckpt.pt or ema_ckpt.pt)")
	parser.add_argument("--style_encoder_ckpt", type=str, required=True, help="Path to style encoder weights (see style_encoder/model.py)")
	parser.add_argument("--style_refs", type=str, nargs="+", required=True, help="Paths to style reference images (recommend 5; will repeat to 5 if fewer)")
	parser.add_argument("--out", type=str, default="out.png", help="Output image path")
	parser.add_argument("--stable_dif_path", type=str, default="./stable-diffusion-v1-5", help="Path to Stable Diffusion v1-5 folder (for VAE & scheduler)")
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--img_h", type=int, default=64)
	parser.add_argument("--img_w", type=int, default=256)
	parser.add_argument("--channels", type=int, default=4)
	parser.add_argument("--emb_dim", type=int, default=320)
	parser.add_argument("--num_heads", type=int, default=4)
	parser.add_argument("--num_res_blocks", type=int, default=1)
	parser.add_argument("--steps", type=int, default=50)
	parser.add_argument("--latent", action="store_true", help="Use latent-space UNet (with SD VAE)")
	parser.add_argument("--no-latent", dest="latent", action="store_false", help="Use pixel-space UNet")
	parser.set_defaults(latent=True)
	args = parser.parse_args()

	device = device_from_str(args.device)

	# tokenizer and text encoder
	tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
	text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
	text_encoder.eval()

	# UNet config (aligned to train.py defaults)
	unet_cfg = dict(
		image_size=(args.img_h, args.img_w),
		in_channels=args.channels,
		model_channels=args.emb_dim,
		out_channels=args.channels,
		num_res_blocks=args.num_res_blocks,
		attention_resolutions=(1, 1),
		channel_mult=(1, 1),
		num_heads=args.num_heads,
		num_classes=None,            # not needed when passing style features
		context_dim=args.emb_dim,
		vocab_size=95,               # unused internally; placeholder
		text_encoder=text_encoder,
		args=SimpleArgs(interpolation=False, mix_rate=None),
	)

	unet = UNetModel(**unet_cfg).to(device)
	unet.eval()
	# load_state_safely(unet, args.unet_ckpt, device)
	unet.load_state_dict(torch.load(args.unet_ckpt, weights_only=True,map_location=device)["state_dict"])  # verify file exists

	# VAE + scheduler (only when latent)
	if args.latent:
		vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(device)
		vae.eval()
	else:
		vae = None
	scheduler = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

	# Style encoder: output must be 1280-dim to match UNet.style_lin
	style_encoder = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=False)#MobileNetV3Style(embedding_dim=1280).to(device)
	style_encoder.eval()
	# load_state_safely(style_encoder, args.style_encoder_ckpt, device)
	style_encoder.load_state_dict(torch.load(args.style_encoder_ckpt, weights_only=True,map_location=device)["state_dict"])  # verify file exists
	# Build style feature batch (5 refs expected by UNet forward; repeat if fewer)
	refs = args.style_refs
	if len(refs) < 5:
		refs = refs + [refs[-1]] * (5 - len(refs))
	elif len(refs) > 5:
		refs = refs[:5]
	style_batch = prepare_style_batch(refs, args.img_h, args.img_w).to(device)
	# ensure style encoder weights and input are on same device & dtype
	style_encoder = style_encoder.to(device)
	style_batch = style_batch.to(device=device, dtype=next(style_encoder.parameters()).dtype)
	with torch.inference_mode():
		style_feats = style_encoder(style_batch)  # [5, 1280]

	# Sample
	with torch.inference_mode():
		image_tensor = sample_image(
			unet=unet,
			tokenizer=tokenizer,
			text=args.text,
			style_feats=style_feats,
			vae=vae,
			scheduler=scheduler,
			device=device,
			img_h=args.img_h,
			img_w=args.img_w,
			channels=args.channels,
			steps=args.steps,
			latent=args.latent,
		)

	# To PIL and save
	# image_tensor: [1, C, H, W], uint8, range 0..255
	image = image_tensor[0].cpu()
	if image.size(0) == 4:  # latent decoded returns 3 channels, but keep guard
		image = image[:3]
	pil = transforms.ToPILImage()(image)
	pil.save(args.out)
	print(f"Saved image to {args.out}")


if __name__ == "__main__":
	main()

