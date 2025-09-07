import os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class IAMDataset(Dataset):
  def __init__(self, root_dir , transform = None):
    self.samples = []
    self.transform = transform
    classes = sorted(os.listdir(root_dir))
    self.class_to_idx = {cls_name : i for i, cls_name in enumerate(classes)}


    for cls in classes:
      cls_folder = os.path.join(root_dir, cls)
      for fname in os.listdir(cls_folder):
        if fname.endswith(".png"):
          self.samples.append((os.path.join(cls_folder,fname), self.class_to_idx[cls]))

  def __len__(self):
    return len(self.samples)

  def __getitem__(self,idx):
    from PIL import Image
    path, label = self.samples[idx]
    image = Image.open(path).convert("L")

    if self.transform:
      image = self.transform(image)
    return image, label
  
def get_train_loader(data_path: str,batch_size=64,num_workers=4):
  transform = transforms.Compose([
        transforms.Resize((32,128)),
        transforms.Grayscale(num_output_channels = 3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3,std=[0.5]*3)
    ])
  
  train_dataset = IAMDataset(data_path, transform=transform)
  train_loader = DataLoader(train_dataset,batch_size,shuffle=True,num_workers=num_workers)
  return train_loader
