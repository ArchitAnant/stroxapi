def main():
    '''Sampling-only version'''

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='diffusionpen')
    parser.add_argument('--level', type=str, default='word')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--dataset', type=str, default='iam')
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--unet', type=str, default='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--sampling_mode', type=str, default='single_sampling')
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')

    args = parser.parse_args()
    
    print('torch version', torch.__version__)

    ######################### LOAD MODEL #########################
    idx = int(''.join(filter(str.isdigit, args.device)))
    device_ids = [idx]

    tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
    text_encoder = CanineModel.from_pretrained("google/canine-c")
    text_encoder = nn.DataParallel(text_encoder, device_ids=device_ids).to(args.device)

    unet = UNetModel(image_size=args.img_size, in_channels=args.channels, model_channels=args.emb_dim,
                     out_channels=args.channels, num_res_blocks=args.num_res_blocks,
                     attention_resolutions=(1,1), channel_mult=(1, 1),
                     num_heads=args.num_heads, num_classes=339,
                     context_dim=args.emb_dim, vocab_size=71,
                     text_encoder=text_encoder, args=args)

    unet = nn.DataParallel(unet, device_ids=device_ids).to(args.device)
    unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
    unet.eval()

    ema = EMA(0.995)
    ema_model = copy.deepcopy(unet).eval().requires_grad_(False)
    ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt'))

    ######################### LOAD VAE/SCHEDULER #########################
    vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae").to(args.device)
    vae.requires_grad_(False)

    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    ######################### LOAD STYLE ENCODER #########################
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True)
    state_dict = torch.load(args.style_path, map_location=args.device)
    feature_extractor.load_state_dict({k: v for k, v in state_dict.items() if k in feature_extractor.state_dict()})
    feature_extractor = nn.DataParallel(feature_extractor, device_ids=device_ids).to(args.device)
    feature_extractor.eval()

    ######################### SAMPLING #########################
    print('Sampling started....')

    if args.sampling_mode == 'single_sampling':
        for word in ['text', 'word']:
            print('Word:', word)
            s = random.randint(0, 339)
            print('style', s)
            labels = torch.tensor([s]).long().to(args.device)
            ema_sampled_images = diffusion.sampling(ema_model, vae, n=1, x_text=word, labels=labels,
                                                    args=args, style_extractor=feature_extractor,
                                                    noise_scheduler=ddim, transform=None,
                                                    tokenizer=tokenizer, text_encoder=text_encoder)
            save_single_images(ema_sampled_images, f'./image_samples/{word}_style_{s}.png', args)

    else:
        print('Sampling mode not implemented in this minimal version')
