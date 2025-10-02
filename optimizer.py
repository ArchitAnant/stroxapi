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

def export_torchscript(model, output_path: str, args):
    model.eval()
    
    # Create dummy inputs matching the diffusion model's expected inputs
    batch_size = 1
    
    # For latent diffusion: (batch, 4, height//8, width//8)
    if args.latent:
        img_height, img_width = args.img_size if isinstance(args.img_size, tuple) else (64, 256)
        input_shape = (batch_size, 4, img_height // 8, img_width // 8)
    else:
        input_shape = (batch_size, 3, 64, 256)
    
    # Create dummy inputs matching the forward signature: (x, timesteps, context, y, style_extractor)
    x = torch.randn(*input_shape).half().cuda()
    timesteps = torch.randint(0, 1000, (batch_size,)).cuda()
    
    # Create dummy text features based on model type
    if args.model_name == 'diffusionpen':
        # CANINE tokenizer output format (max_length=40 from train.py)
        context = {
            'input_ids': torch.randint(0, 1000, (batch_size, 40)).cuda(),
            'attention_mask': torch.ones(batch_size, 40).cuda()
        }
    else:
        # Wordstylist uses simple tensor (OUTPUT_MAX_LEN=95 from train.py)
        context = torch.randint(0, 100, (batch_size, 95)).cuda()
    
    # Style class labels (y = s_id from training)
    if args.dataset == 'iam':
        num_classes = 339
    elif args.dataset == 'gnhk':
        num_classes = 515
    else:
        num_classes = 339
    y = torch.randint(0, num_classes, (batch_size,)).cuda()
    
    # Dummy style features (MobileNetV2 output size)
    style_extractor = torch.randn(batch_size, 1280).half().cuda()
    
    # Trace the model with the correct forward signature
    model = model.cuda().half()
    with torch.no_grad():
        traced_model = torch.jit.trace(
            model, 
            (x, timesteps, context, y, style_extractor),
            strict=False
        )
    
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

    # --- Initialize your model with the proper arguments (matching train.py) ---
    
    # Set style_classes based on dataset (from train.py)
    if args.dataset == 'iam':
        style_classes = 339
    elif args.dataset == 'gnhk':
        style_classes = 515
    else:
        style_classes = 339  # default to IAM
    
    # Character classes and vocab_size calculation (from train.py)
    character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']
    
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
    else:
        vocab_size = len(character_classes)
    
    # Initialize text encoder if needed (from train.py)
    text_encoder = None
    if args.model_name == 'diffusionpen':
        from transformers import CanineModel, CanineTokenizer
        import torch.nn as nn
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        # Note: For inference, we don't need DataParallel, just move to device
        text_encoder = text_encoder.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model with exact same parameters as train.py
    model = UNetModel(
        image_size=args.img_size,            # from args
        in_channels=args.channels,           # from args
        model_channels=args.emb_dim,         # from args
        out_channels=args.channels,          # same as in_channels
        num_res_blocks=args.num_res_blocks,  # from args
        attention_resolutions=(1, 1),        # from train.py
        channel_mult=(1, 1),                 # from train.py
        num_heads=args.num_heads,            # from args
        num_classes=style_classes,           # calculated above
        context_dim=args.emb_dim,            # from train.py
        vocab_size=vocab_size,               # calculated above
        text_encoder=text_encoder,           # initialized above
        args=args                            # pass args
    )

    # Load checkpoint
    model.load_state_dict(state_dict, strict=False)
    print("Model weights loaded.")

    # Export TorchScript
    export_torchscript(model, args.output, args)
    print("Export completed.")


if __name__ == "__main__":
    main()
