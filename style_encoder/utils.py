import torch
from torch.nn import functional as F


def sample_triplets(embeddings, labels):
  labels= labels.cpu().numpy()
  anchor, positive, negative = [],[],[]
  for i in range(len(labels)):
    anc = embeddings[i]
    pos_idx = i
    same_label = [j for j in range(len(labels)) if labels[j] == labels[i] and j !=i]
    if len(same_label) == 0:
      continue
    pos_idx = same_label[torch.randint(len(same_label), (1,)).item()]
    diff_label = [j for j in range(len(labels)) if labels[j]!=labels[i]]
    neg_idx = diff_label[torch.randint(len(diff_label), (1,)).item()]
    anchor.append(anc)
    positive.append(embeddings[pos_idx])
    negative.append(embeddings[neg_idx])
  if len(anchor) ==0:
    return None, None , None
  return torch.stack(anchor), torch.stack(positive), torch.stack(negative)