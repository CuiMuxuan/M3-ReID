import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemoryLoss(nn.Module):
    """
    EMA class-prototype loss for small identity-balanced ReID batches.

    The memory is a non-parametric buffer: it is updated from detached batch
    features and supplies more class-level negatives than the current mini-batch.
    Gradients flow only through the current embeddings.
    """

    def __init__(self, num_classes, feat_dim, temperature=0.07, momentum=0.2):
        super().__init__()
        if num_classes <= 0:
            raise ValueError('num_classes must be positive.')
        if feat_dim <= 0:
            raise ValueError('feat_dim must be positive.')
        if temperature <= 0:
            raise ValueError('temperature must be positive.')
        if not 0.0 < momentum <= 1.0:
            raise ValueError('momentum must be in (0, 1].')
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.temperature = temperature
        self.momentum = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim))
        self.register_buffer('valid', torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def _update_memory(self, embeddings, labels):
        embeddings = F.normalize(embeddings.detach().float(), p=2, dim=1)
        labels = labels.detach().long()
        unique_labels = torch.unique(labels)
        for label in unique_labels:
            label_idx = int(label.item())
            if label_idx < 0 or label_idx >= self.num_classes:
                continue
            class_feat = embeddings[labels == label].mean(dim=0)
            class_feat = F.normalize(class_feat, p=2, dim=0)
            if self.valid[label_idx]:
                updated = (1.0 - self.momentum) * self.prototypes[label_idx] + self.momentum * class_feat
                self.prototypes[label_idx] = F.normalize(updated, p=2, dim=0)
            else:
                self.prototypes[label_idx] = class_feat
                self.valid[label_idx] = True

    def forward(self, embeddings, labels):
        if embeddings.size(0) == 0:
            return embeddings.new_zeros(())
        if embeddings.size(1) != self.feat_dim:
            raise ValueError(f'Expected feature dim {self.feat_dim}, got {embeddings.size(1)}.')

        labels = labels.long()
        in_range = (labels >= 0) & (labels < self.num_classes)
        if not in_range.any():
            return embeddings.new_zeros(())

        embeddings = embeddings[in_range]
        labels = labels[in_range]
        self._update_memory(embeddings, labels)

        valid_class_indices = torch.nonzero(self.valid, as_tuple=False).flatten()
        if valid_class_indices.numel() <= 1:
            return embeddings.new_zeros(())

        features = F.normalize(embeddings.float(), p=2, dim=1)
        proto = self.prototypes[valid_class_indices].to(features.device)
        logits = torch.matmul(features, proto.t()) / self.temperature

        class_to_logit = embeddings.new_full((self.num_classes,), -1, dtype=torch.long)
        class_to_logit[valid_class_indices] = torch.arange(
            valid_class_indices.numel(), device=features.device, dtype=torch.long
        )
        targets = class_to_logit[labels]
        valid_targets = targets >= 0
        if not valid_targets.any():
            return embeddings.new_zeros(())
        return F.cross_entropy(logits[valid_targets], targets[valid_targets])
