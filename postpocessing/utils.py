import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
import random

class Line:
    def __init__(self,top_height, small_word_size, baseline_dist):
        self.top_height = top_height
        self.small_word_size = small_word_size
        self.baseline_dist = baseline_dist

    def __str__(self):
        return f"Line(top_height={self.top_height}, small_word_size={self.small_word_size}, baseline_dist={self.baseline_dist})"
    

def clip_image(img: torch.Tensor, threshold=0.95) -> torch.Tensor:
    """
    Clip the handwriting image tensor to the tightest region containing ink,
    then pad 2 pixels of white margin on all sides.

    Args:
        img (torch.Tensor): Tensor of shape (3, H, W), values in [0, 1]
        threshold (float): Pixels >= threshold are treated as background (white)

    Returns:
        torch.Tensor: Cropped and padded image (3, newH, newW)
    """
    # Work on one channel (grayscale duplicated across 3)
    gray = img[0]  # (H, W)
    H, W = gray.shape

    # Mask where handwriting (ink) exists
    mask = gray < threshold

    # If no ink found, return original
    if not mask.any():
        return img

    # Bounding box of ink
    ys, xs = torch.where(mask)
    top, bottom = ys.min().item(), ys.max().item()
    left, right = xs.min().item(), xs.max().item()

    # Crop
    cropped = img[:, top:bottom + 1, left:right + 1]

    # Determine vertical padding
    pad_top = 2 if top > 0 else 0
    pad_bottom = 2 if bottom < H - 1 else 0

    # Always keep horizontal padding (left/right)
    padded = F.pad(cropped, pad=(2, 2, pad_top, pad_bottom), mode="constant", value=1.0)

    return padded

def analyse_word(image_tensor: torch.Tensor,
                 threshold=0.7) -> Line:
  gray = image_tensor[0]
  H,W = gray.shape
  mask = gray<threshold

  row_stroke = torch.sum(mask,dim=1).to(torch.float16)
  lower = torch.quantile(row_stroke.float(),0.25)
  bottom = torch.quantile(row_stroke.float(),0.8)
  mid = (lower+bottom)/2

  indices = torch.where(row_stroke>=mid)[0]

  return Line(
      top_height=H,
      small_word_size=indices.shape[0],
      baseline_dist=indices[-1]
  )


def resize_img(curr_img: torch.Tensor,
               sample_line: Line,
               curr_line: Line) -> torch.Tensor:
    """
    Resize and align handwriting image to match target small-letter height
    and baseline (measured from top).

    Final image height is padded/cropped to 64px; width can vary.

    Args:
        curr_img: (3, H, W) handwriting tensor [0,1]
        sample_line: target style Line (baseline_dist from top)
        curr_line: current image Line (baseline_dist from top)
    """
    _, H, W = curr_img.shape

    # Convert to floats (avoid Tensor round issues)
    sample_small = float(sample_line.small_word_size)
    sample_base = float(sample_line.baseline_dist)
    curr_small = float(curr_line.small_word_size)
    curr_base = float(curr_line.baseline_dist)

    scale = sample_small / max(curr_small, 1.0)
    new_H = int(round(H * scale))
    new_W = int(round(W * scale))

    resized = F.interpolate(
        curr_img.unsqueeze(0),
        size=(new_H, new_W),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

    # revised_line = analyse_word(resized)
    curr_scaled_baseline = curr_base * scale
    delta_baseline = int(round(sample_base - curr_scaled_baseline))

    if delta_baseline > 0:
        # curr baseline is ABOVE target → pad TOP
        aligned = F.pad(resized, (0, 0, delta_baseline, 0), value=1.0)
    elif delta_baseline < 0:
        # curr baseline is BELOW target → pad BOTTOM
        aligned = F.pad(resized, (0, 0, 0, -delta_baseline), value=1.0)
    else:
        aligned = resized

    _, h_aligned, _ = aligned.shape
    if h_aligned < 64:
        pad_top = (64 - h_aligned) // 2
        pad_bottom = 64 - h_aligned - pad_top
        aligned = F.pad(aligned, (0, 0, pad_top, pad_bottom), value=1.0)
    elif h_aligned > 64:
        start = (h_aligned - 64) // 2
        aligned = aligned[:, start:start + 64, :]

    return aligned


def stitch_images_side_by_side(image_tensors):
    """
    Stitches a list of image tensors side by side with random padding between them.

    Args:
        image_tensors (list): A list of torch.Tensor images (assuming same height and channels).

    Returns:
        torch.Tensor: A single tensor with images stitched side by side with padding.
    """
    if not image_tensors:
        return None

    stitched_image = image_tensors[0]

    for i in range(1, len(image_tensors)):
        # Determine random padding width (e.g., between 5 and 20 pixels)
        padding_width = random.randint(15, 20)
        # For PyTorch-style tensors (C, H, W)
        C, H, W = stitched_image.shape
        padding = torch.ones((C, H, padding_width), dtype=stitched_image.dtype, device=stitched_image.device)
        stitched_image = torch.cat((stitched_image, padding, image_tensors[i]), dim=2)

    return stitched_image

def form_line(image_path_list,text):
  base_image = Image.open(image_path_list[0]).convert('L')
  base_image_tensor = clip_image(transforms.ToTensor()(base_image))
  base_image_line = analyse_word(base_image_tensor)
  text = text[1:]

  img_tensors = [base_image_tensor]

  cenders = ['q','t','y','p','d','f','g','h','j','k','l','b']

  for i,img_path in enumerate(image_path_list[1:]):
    curr_img = Image.open(img_path).convert('L')
    curr_img_tensor = clip_image(transforms.ToTensor()(curr_img))
    curr_line = analyse_word(curr_img_tensor)
    if text[i] not in cenders and text[i].lower() and len(text[i])==1:
      curr_line.small_word_size=64
      curr_line.baseline_dist+=64
    print(f"{text[i]} : {curr_line}")
    curr_re_tensor = resize_img(curr_img_tensor,base_image_line,curr_line)
    img_tensors.append(curr_re_tensor)
  print()
  
  return stitch_images_side_by_side(img_tensors)