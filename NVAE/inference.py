import torch
from torch.utils.data import DataLoader
import argparse
import os
from model import AutoEncoder
import utils

class NVAEInference:
    def __init__(self, model_path, device='cuda'):
        parser = argparse.ArgumentParser()
        parser.add_argument('--root', type=str, default='model/',
                        help='location of the results')
        parser.add_argument('--save', type=str, default='eval/',
                            help='id used for storing intermediate results')
        # data
        parser.add_argument('--dataset', type=str, default='handwriting',
                            choices=['cifar10', 'mnist', 'omniglot', 'celeba_64', 'celeba_256',
                                    'imagenet_32', 'ffhq', 'lsun_bedroom_128', 'stacked_mnist',
                                    'lsun_church_128', 'lsun_church_64'],
                            help='which dataset to use')
        parser.add_argument('--data', type=str, default='nasvae/data/',
                            help='location of the data corpus')
        # optimization
        parser.add_argument('--batch_size', type=int, default=16,
                            help='batch size per GPU')
        parser.add_argument('--learning_rate', type=float, default=1e-2,
                            help='init learning rate')
        parser.add_argument('--learning_rate_min', type=float, default=1e-4,
                            help='min learning rate')
        parser.add_argument('--weight_decay', type=float, default=3e-4,
                            help='weight decay')
        parser.add_argument('--weight_decay_norm', type=float, default=0.,
                            help='The lambda parameter for spectral regularization.')
        parser.add_argument('--weight_decay_norm_init', type=float, default=10.,
                            help='The initial lambda parameter')
        parser.add_argument('--weight_decay_norm_anneal', action='store_true', default=False,
                            help='This flag enables annealing the lambda coefficient from '
                                '--weight_decay_norm_init to --weight_decay_norm.')
        parser.add_argument('--epochs', type=int, default=10,
                            help='num of training epochs')
        parser.add_argument('--warmup_epochs', type=int, default=5,
                            help='num of training epochs in which lr is warmed up')
        parser.add_argument('--fast_adamax', action='store_true', default=False,
                            help='This flag enables using our optimized adamax.')
        parser.add_argument('--arch_instance', type=str, default='res_mbconv',
                            help='path to the architecture instance')
        # KL annealing
        parser.add_argument('--kl_anneal_portion', type=float, default=0.3,
                            help='The portions epochs that KL is annealed')
        parser.add_argument('--kl_const_portion', type=float, default=0.0001,
                            help='The portions epochs that KL is constant at kl_const_coeff')
        parser.add_argument('--kl_const_coeff', type=float, default=0.0001,
                            help='The constant value used for min KL coeff')
        # Flow params
        parser.add_argument('--num_nf', type=int, default=0,
                            help='The number of normalizing flow cells per groups. Set this to zero to disable flows.')
        parser.add_argument('--num_x_bits', type=int, default=8,
                            help='The number of bits used for representing data for colored images.')
        # latent variables
        parser.add_argument('--num_latent_scales', type=int, default=1,
                            help='the number of latent scales')
        parser.add_argument('--num_groups_per_scale', type=int, default=10,
                            help='number of groups of latent variables per scale')
        parser.add_argument('--num_latent_per_group', type=int, default=20,
                            help='number of channels in latent variables per group')
        parser.add_argument('--ada_groups', action='store_true', default=False,
                            help='Settings this to true will set different number of groups per scale.')
        parser.add_argument('--min_groups_per_scale', type=int, default=1,
                            help='the minimum number of groups per scale.')
        # encoder parameters
        parser.add_argument('--num_channels_enc', type=int, default=32,
                            help='number of channels in encoder')
        parser.add_argument('--num_preprocess_blocks', type=int, default=2,
                            help='number of preprocessing blocks')
        parser.add_argument('--num_preprocess_cells', type=int, default=3,
                            help='number of cells per block')
        parser.add_argument('--num_cell_per_cond_enc', type=int, default=1,
                            help='number of cell for each conditional in encoder')
        # decoder parameters
        parser.add_argument('--num_channels_dec', type=int, default=32,
                            help='number of channels in decoder')
        parser.add_argument('--num_postprocess_blocks', type=int, default=2,
                            help='number of postprocessing blocks')
        parser.add_argument('--num_postprocess_cells', type=int, default=3,
                            help='number of cells per block')
        parser.add_argument('--num_cell_per_cond_dec', type=int, default=1,
                            help='number of cell for each conditional in decoder')
        parser.add_argument('--num_mixture_dec', type=int, default=10,
                            help='number of mixture components in decoder. set to 1 for Normal decoder.')
        # NAS
        parser.add_argument('--use_se', action='store_true', default=False,
                            help='This flag enables squeeze and excitation.')
        parser.add_argument('--res_dist', action='store_true', default=False,
                            help='This flag enables squeeze and excitation.')
        parser.add_argument('--cont_training', action='store_true', default=False,
                            help='This flag enables training from an existing checkpoint.')
        # DDP.
        parser.add_argument('--num_proc_node', type=int, default=1,
                            help='The number of nodes in multi node env.')
        parser.add_argument('--node_rank', type=int, default=0,
                            help='The index of node.')
        parser.add_argument('--local_rank', type=int, default=0,
                            help='rank of process in the node')
        parser.add_argument('--global_rank', type=int, default=0,
                            help='rank of process among all the processes')
        parser.add_argument('--num_process_per_node', type=int, default=1,
                            help='number of gpus')
        parser.add_argument('--master_address', type=str, default='0.0.0.0',
                            help='address for master')
        parser.add_argument('--seed', type=int, default=1,
                            help='seed used for initialization')

        args = parser.parse_args([])
        arch_instance = utils.get_arch_cells(args.arch_instance)
        writer = utils.Writer(args.global_rank, args.save)

        self.model = AutoEncoder(args, writer, arch_instance)
        checkpoint = torch.load(model_path, map_location=device,weights_only=True)
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
