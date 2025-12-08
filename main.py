from typing import List, Optional, OrderedDict

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel

from diff_unet import UNetModel
from style_encoder.model import MobileNetV3Style

from postpocessing.utils import form_line

from repo.upload_main import upload,generate_node_code
from repo.client import BlobClient

class ModelPipeline:
	def __init__(self,
			  unet_path,
			  vae_folder_path,
			  style_encoder_path,
			  device: torch.device
	):
		self.text_tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
		self.text_encoder = CanineModel.from_pretrained("google/canine-c").to(device)
		
		unet_cfg = dict(
		image_size=(64, 256),
		in_channels=4,
		model_channels=320,
		out_channels=4,
		num_res_blocks=1,
		attention_resolutions=(1, 1),
		channel_mult=(1, 1),
		num_heads=4,
		num_classes=None,            # not needed when passing style features
		context_dim=320,
		vocab_size=95,               # unused internally; placeholder
		text_encoder=self.text_encoder,
		args=SimpleArgs(interpolation=False, mix_rate=None),
		)

		self.unet = UNetModel(**unet_cfg).to(device)
		state_dict = torch.load(unet_path, map_location=device)

		new_state_dict = OrderedDict()
		for k, v in state_dict.items():
			if "module." in k:
				k = k.replace("module.", "")
			if k == "label_emb.weight":  # skip this one
				continue
			new_state_dict[k] = v

		# Now load
		self.unet.load_state_dict(new_state_dict)

		self.vae = AutoencoderKL.from_pretrained(vae_folder_path, subfolder="vae").to(device)

		self.scheduler = DDIMScheduler.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="scheduler")

		self.style_encoder = MobileNetV3Style(embedding_dim=1280)
		state = torch.load(style_encoder_path, map_location=device)
		self.style_encoder.load_state_dict(state)


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

def load_state_safely(module: nn.Module, state_dict_path: str, map_location: torch.device):
	sd = torch.load(state_dict_path, map_location=map_location)
	# handle checkpoints saved with wrappers
	if isinstance(sd, dict) and "state_dict" in sd:
		sd = sd["state_dict"]
	# strip possible prefixes like 'module.'
	new_sd = {}
	for k, v in sd.items():
		if k.startswith("module."):
			new_k = k[len("module."):]
		elif k.startswith("model."):
			new_k = k[len("model."):]
		else:
			new_k = k
		new_sd[new_k] = v
	missing, unexpected = module.load_state_dict(new_sd, strict=False)
	if missing:
		print(f"[warn] Missing keys when loading {state_dict_path}: {len(missing)}")
	if unexpected:
		print(f"[warn] Unexpected keys when loading {state_dict_path}: {len(unexpected)}")

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
	steps: int,
):
	# text conditioning
	tok = tokenizer([text], padding="max_length", truncation=True, return_tensors="pt", max_length=40).to(device)
	
	x = torch.randn((1, 4, img_h // 8, img_w // 8), device=device)
	
	scheduler.set_timesteps(steps)
	
	for t in scheduler.timesteps:
		t_b = torch.full((x.size(0),), t.item(), device=device, dtype=torch.long)
		noise_pred = unet(x, timesteps=t_b, context=tok, y=None, style_extractor=style_feats)
		step_out = scheduler.step(noise_pred, t, x)
		x = step_out.prev_sample


	latents = (1 / 0.18215) * x
	image = vae.decode(latents).sample
	image = (image / 2 + 0.5).clamp(0, 1)  # [0,1]
	image = (image * 255).round().byte()
	return image


def main_sample(
		model_pipeline: ModelPipeline,
		text_list : List[str],
		style_refs : List[str],
		uname : str,
):	
	device = 'cuda'
	if not torch.cuda.is_available():
		print("CUDA is needed.")
		return
    

	# tokenizer and text encoder
	tokenizer = model_pipeline.text_tokenizer#CanineTokenizer.from_pretrained("google/canine-c")
	text_encoder = model_pipeline.text_encoder#CanineModel.from_pretrained("google/canine-c").to(device)
	text_encoder.eval()

	unet = model_pipeline.unet
	unet.eval()
	
	# VAE + scheduler (only when latent)
	vae = model_pipeline.vae
	vae.eval()

	scheduler = model_pipeline.scheduler

	# Style encoder: output must be 1280-dim to match UNet.style_lin
	style_encoder = model_pipeline.style_encoder
	style_encoder.eval()

	# Build style feature batch (5 refs expected by UNet forward; repeat if fewer)
	refs = style_refs
	if len(refs) < 5:
		refs = refs + [refs[-1]] * (5 - len(refs))
	elif len(refs) > 5:
		refs = refs[:5]
	style_batch = prepare_style_batch(refs, 64, 256).to(device)
	# ensure style encoder weights and input are on same device & dtype
	style_encoder = style_encoder.to(device)
	style_batch = style_batch.to(device=device, dtype=next(style_encoder.parameters()).dtype)
	with torch.inference_mode():
		style_feats = style_encoder(style_batch)  # [5, 1280]
	word_count = 0

	file_paths = []
	# Sample
	for text in text_list:
		word_count+=1
		print(f"Generating image for text {word_count}: {text}")
		with torch.inference_mode():
			image_tensor = sample_image(
                unet=unet,
                tokenizer=tokenizer,
                text=text,
                style_feats=style_feats,
                vae=vae,
                scheduler=scheduler,
                device=device,
                img_h=64,
                img_w=256,
                steps=100,
            )
		image = image_tensor[0].cpu()
		if image.size(0) == 4:  # latent decoded returns 3 channels
			image = image[:3]
		pil = transforms.ToPILImage()(image)
		file_path = f"{uname}/generated_{word_count}.png"
		pil.save(file_path)
		file_paths.append(file_path)
	
	img_tensor_list = form_line(file_paths, text_list)

	node_code = generate_node_code(uname)
	blob_client = BlobClient()
	if upload(img_tensor_list, node_code, blob_client):
		return (node_code, len(img_tensor_list))
	else:
		return None

	