import torch
import torch.nn.functional as F
import numpy as np


class NTXentLoss_poly(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss_poly, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye(2 * self.batch_size, 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        # Combine representations and print shape
        representations = torch.cat([zjs, zis], dim=0)
        # print(f"representations shape: {representations.shape}")

        # Compute similarity matrix and print its shape
        similarity_matrix = self.similarity_function(representations, representations)
        # print(f"similarity_matrix shape: {similarity_matrix.shape}")

        # Extract positive pairs from the diagonal offsets
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        # print(f"l_pos shape: {l_pos.shape}")
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # print(f"r_pos shape: {r_pos.shape}")
        
        # Concatenate positive scores and print shape before and after reshape
        positives = torch.cat([l_pos, r_pos])
        # print(f"positives before view: {positives.shape}")
        try:
            positives = positives.view(2 * self.batch_size, 1)
            # print(f"positives after view: {positives.shape}")
        except Exception as e:
            print(f"Error reshaping positives: {e}")
            raise

        # Extract negatives using the mask and print shape
        negatives = similarity_matrix[self.mask_samples_from_same_repr]
        # print(f"negatives raw shape: {negatives.shape}")
        try:
            negatives = negatives.view(2 * self.batch_size, -1)
            # print(f"negatives after view: {negatives.shape}")
        except Exception as e:
            print(f"Error reshaping negatives: {e}")
            raise

        # Combine positives and negatives into logits and scale
        logits = torch.cat((positives, negatives), dim=1)
        # print(f"logits shape: {logits.shape}")
        logits /= self.temperature

        # Create labels (all zeros, because positive score is at index 0)
        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        CE = self.criterion(logits, labels)
        # print(f"CrossEntropy Loss: {CE.item():.4f}")

        # Compute additional poly loss component
        onehot_label = torch.cat((torch.ones(2 * self.batch_size, 1),
                                  torch.zeros(2 * self.batch_size, negatives.shape[-1])),
                                 dim=-1).to(self.device).long()
        pt = torch.mean(onehot_label * F.softmax(logits, dim=-1))
        # print(f"Poly loss term (pt): {pt.item():.4f}")

        epsilon = self.batch_size
        loss = CE / (2 * self.batch_size) + epsilon * (1 / self.batch_size - pt)
        # print(f"Final NTXentLoss_poly: {loss.item():.4f}")
        return loss

def cosine_byol(p, z):
    """BYOL‑style cosine loss (returns 0 when p == z)."""
    z = z.detach()
    p = F.normalize(p, dim=-1)
    z = F.normalize(z, dim=-1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()
