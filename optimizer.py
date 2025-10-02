import torch
import argparse
from pathlib import Path
from diff_unet import UNetModel

# Example: replace with your actual model class
# from your_model_file import UNetModel  


def load_checkpoint(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    # Usually weights are in 'state_dict', else load full dict
    state_dict = ckpt.get("state_dict", ckpt)

    # Strip out training-only entries
    state_dict = {k: v for k, v in state_dict.items() if "optimizer" not in k and "step" not in k}

    # Convert to FP16 if float32
    for k in state_dict:
        if state_dict[k].dtype == torch.float32:
            state_dict[k] = state_dict[k].half()

    return state_dict

def export_torchscript(model, output_path: str, input_shape=(1, 3, 256, 256)):
    model.eval()
    example_input = torch.randn(*input_shape).half().cuda()  # FP16
    traced_model = torch.jit.trace(model.cuda(), example_input)
    traced_model.save(output_path)
    print(f"TorchScript inference model saved at: {output_path}")
    
def main():
    parser = argparse.ArgumentParser(description="Export CKPT/EMA to FP16 TorchScript for inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to your ckpt or ema_ckpt file")
    parser.add_argument("--output", type=str, default="model_inference.pt", help="Output TorchScript file")
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4) 
    parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))  
    parser.add_argument('--dataset', type=str, default='iam', help='iam, gnhk') 
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path') 
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False) 
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')
    
    args = parser.parse_args()

    # Load weights
    state_dict = load_checkpoint(args.ckpt)
    print("Loaded checkpoint from:", args.ckpt)

    # --- Initialize your model with the proper arguments ---
    model = UNetModel(
        image_size=(64, 256),        # your training image size
        in_channels=4,               # channels from training
        model_channels=320,          # emb_dim from training
        out_channels=4,              # same as in_channels
        num_res_blocks=1,            # from training
        attention_resolutions=(1, 1),# example value, adjust if needed
        channel_mult=(1, 1),         # same as training
        num_heads=4,                 # from training
        num_classes=None,            # set if your model is class-conditional
        # You can also pass context_dim, vocab_size, text_encoder if used
        args=args
    )

    # Load checkpoint
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded.")

    # Export TorchScript
    export_torchscript(model, args.output)
    print("Export completed.")


if __name__ == "__main__":
    main()
