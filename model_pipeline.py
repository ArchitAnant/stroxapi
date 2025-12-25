import torch
from diffusion_pen.diff_unet import UNetModel
from typing import List, Optional, OrderedDict
from style_encoder.model import MobileNetV3Style
from diffusers import AutoencoderKL, DDIMScheduler
from transformers import CanineTokenizer, CanineModel

class SimpleArgs:
	# minimal shim to satisfy UNetModel's expectations
	def __init__(self, interpolation: bool = False, mix_rate: Optional[float] = None):
		self.interpolation = interpolation
		self.mix_rate = mix_rate

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
		use_fp16 = True if device=='cpu' else False
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
