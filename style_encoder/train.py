import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from model import MobileNetV3Style
from dataset import get_train_loader
from utils import sample_triplets

device = torch.device("cuda" if torch.cuda.is_available() else "cpu" )

# Hyperparameters
DATA_PATH = "data/"
NUM_EPOCHS = 20
LR = 1e-3
MARGIN=1.0
P=2
EMBEDDINGS_DIM=512


model = MobileNetV3Style(embedding_dim=EMBEDDINGS_DIM).to(device)
criterion_triplet = nn.TripletMarginLoss(margin=MARGIN, p=P)
optimizer = optim.Adam(model.parameters(), lr=LR)
train_loader = get_train_loader(data_path=DATA_PATH)


for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    total_pos_dist = 0
    total_neg_dist = 0
    total_triplets = 0

    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(images)

        anc, pos, neg = sample_triplets(embeddings, labels)
        if anc is None:
            continue

        # Compute distances
        pos_dist = torch.norm(anc - pos, p=2, dim=1)
        neg_dist = torch.norm(anc - neg, p=2, dim=1)

        loss = criterion_triplet(anc, pos, neg)
        loss.backward()
        optimizer.step()

        batch_triplets = anc.size(0)
        total_loss += loss.item() * batch_triplets
        total_pos_dist += pos_dist.sum().item()
        total_neg_dist += neg_dist.sum().item()
        total_triplets += batch_triplets

        # Update tqdm dynamically
        loop.set_postfix({
            "Loss": total_loss/total_triplets if total_triplets>0 else 0,
            "PosDist": total_pos_dist/total_triplets if total_triplets>0 else 0,
            "NegDist": total_neg_dist/total_triplets if total_triplets>0 else 0
        })

    avg_loss = total_loss / total_triplets if total_triplets>0 else 0
    avg_pos_dist = total_pos_dist / total_triplets if total_triplets>0 else 0
    avg_neg_dist = total_neg_dist / total_triplets if total_triplets>0 else 0
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Triplet Loss: {avg_loss:.4f} | Avg Pos Dist: {avg_pos_dist:.4f} | Avg Neg Dist: {avg_neg_dist:.4f}")

