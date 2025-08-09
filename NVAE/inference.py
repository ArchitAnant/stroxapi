import torch
from torch.utils.data import DataLoader
import argparse
import os
from model import AutoEncoder
import utils

class NVAEInference:
    def __init__(self, model_path, device='cuda'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', type=str, default='handwriting')
        parser.add_argument('--use_se', action='store_true')
        parser.add_argument('--res_dist', action='store_true')
        parser.add_argument('--num_x_bits', type=int, default=8)
        parser.add_argument('--num_latent_scales', type=int, default=4)
        parser.add_argument('--num_groups_per_scale', type=int, default=4)
        parser.add_argument('--num_latent_per_group', type=int, default=4)
        parser.add_argument('--min_groups_per_scale', type=int, default=1)
        parser.add_argument('--num_channels_enc', type=int, default=64)
        parser.add_argument('--num_channels_dec', type=int, default=64)
        parser.add_argument('--num_preprocess_blocks', type=int, default=2)
        parser.add_argument('--num_preprocess_cells', type=int, default=2)
        parser.add_argument('--num_cell_per_cond_enc', type=int, default=2)
        parser.add_argument('--num_postprocess_blocks', type=int, default=2)
        parser.add_argument('--num_postprocess_cells', type=int, default=2)
        parser.add_argument('--num_cell_per_cond_dec', type=int, default=2)
        parser.add_argument('--arch_instance', type=str, default='default')
        parser.add_argument('--global_rank', type=int, default=0)
        parser.add_argument('--save', type=str, default='./save')

        args = parser.parse_args([])
        arch_instance = utils.get_arch_cells(args.arch_instance)
        writer = utils.Writer(args.global_rank, args.save)

        self.model = AutoEncoder(args, writer, arch_instance)
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(device)
        self.device = device
        self.model.eval()

    def run_encoder(self, data_batch):
        with torch.inference_mode():
            data_batch = data_batch.to(self.device)
            latent = self.model.encoder(data_batch)
        return latent

    def run_decoder(self, latent_batch):
        with torch.inference_mode():
            latent_batch = latent_batch.to(self.device)
            recon = self.model.decoder(latent_batch)
        return recon


# if __name__ == '__main__':
#     model_path = './checkpoint.pt'  # replace with your actual checkpoint path
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     inference_engine = NVAEInference(model_path, device)

#     dummy_input = torch.randn(2, 1, 128, 128)  # Adjust shape based on dataset
#     latents = inference_engine.run_encoder(dummy_input)
#     outputs = inference_engine.run_decoder(latents)

#     print("Latents shape:", latents.shape)
#     print("Reconstructed output shape:", outputs.shape)
