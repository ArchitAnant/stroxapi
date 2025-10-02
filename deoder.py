from diffusers import AutoencoderKL
import torch

class LatentToImageDecoder:
    def __init__(self, model_name="runwayml/stable-diffusion-v1-5"):
        self.vae = AutoencoderKL.from_pretrained(
            model_name,
            subfolder="vae"
        ).cuda().eval()

    def decode(self, latent):
        """
        Decode latent tensor to image.

        Args:
            latent (torch.Tensor): Latent tensor of shape [B, 4, H/8, W/8].

        Returns:
            torch.Tensor: Decoded image tensor of shape [B, 3, H, W].
        """
        with torch.no_grad():
            decoded = self.vae.decode(latent).sample
        return decoded
    
LatentToImageDecoder()