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


class CrossModalityPrototypeTripletLoss(nn.Module):
    """
    EMA modality-specific prototype triplet loss for VI-ReID retrieval.

    For each anchor, the positive is the same identity prototype from the other
    modality; negatives are different identity prototypes from that other
    modality. This targets i2v/v2i ranking more directly than class CE memory.
    """

    def __init__(self, num_classes, feat_dim, margin=0.1, momentum=0.2, modality_values=(1, 2)):
        super().__init__()
        if num_classes <= 0:
            raise ValueError('num_classes must be positive.')
        if feat_dim <= 0:
            raise ValueError('feat_dim must be positive.')
        if margin < 0:
            raise ValueError('margin must be non-negative.')
        if not 0.0 < momentum <= 1.0:
            raise ValueError('momentum must be in (0, 1].')
        if len(modality_values) != 2:
            raise ValueError('CrossModalityPrototypeTripletLoss expects exactly two modalities.')
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.margin = margin
        self.momentum = momentum
        self.modality_values = tuple(int(item) for item in modality_values)
        self.register_buffer('prototypes', torch.zeros(num_classes, 2, feat_dim))
        self.register_buffer('valid', torch.zeros(num_classes, 2, dtype=torch.bool))

    @torch.no_grad()
    def _update_memory(self, embeddings, labels, m_labels):
        embeddings = F.normalize(embeddings.detach().float(), p=2, dim=1)
        labels = labels.detach().long()
        m_labels = m_labels.detach().long()
        for mod_idx, mod_value in enumerate(self.modality_values):
            mod_mask = m_labels == mod_value
            if not mod_mask.any():
                continue
            unique_labels = torch.unique(labels[mod_mask])
            for label in unique_labels:
                label_idx = int(label.item())
                if label_idx < 0 or label_idx >= self.num_classes:
                    continue
                class_mask = mod_mask & (labels == label)
                class_feat = embeddings[class_mask].mean(dim=0)
                class_feat = F.normalize(class_feat, p=2, dim=0)
                if self.valid[label_idx, mod_idx]:
                    updated = ((1.0 - self.momentum) * self.prototypes[label_idx, mod_idx]
                               + self.momentum * class_feat)
                    self.prototypes[label_idx, mod_idx] = F.normalize(updated, p=2, dim=0)
                else:
                    self.prototypes[label_idx, mod_idx] = class_feat
                    self.valid[label_idx, mod_idx] = True

    def forward(self, embeddings, labels, m_labels):
        if embeddings.size(0) == 0:
            return embeddings.new_zeros(())
        if embeddings.size(1) != self.feat_dim:
            raise ValueError(f'Expected feature dim {self.feat_dim}, got {embeddings.size(1)}.')

        labels = labels.long()
        m_labels = m_labels.long()
        in_range = (labels >= 0) & (labels < self.num_classes)
        in_modality = torch.zeros_like(in_range)
        for mod_value in self.modality_values:
            in_modality = in_modality | (m_labels == mod_value)
        keep = in_range & in_modality
        if not keep.any():
            return embeddings.new_zeros(())

        embeddings = embeddings[keep]
        labels = labels[keep]
        m_labels = m_labels[keep]
        features = F.normalize(embeddings.float(), p=2, dim=1)

        losses = []
        for mod_idx, mod_value in enumerate(self.modality_values):
            anchor_mask = m_labels == mod_value
            if not anchor_mask.any():
                continue
            opp_idx = 1 - mod_idx
            valid_class_indices = torch.nonzero(self.valid[:, opp_idx], as_tuple=False).flatten()
            if valid_class_indices.numel() <= 1:
                continue

            anchor_features = features[anchor_mask]
            anchor_labels = labels[anchor_mask]
            proto = self.prototypes[valid_class_indices, opp_idx].to(anchor_features.device)
            sim = torch.matmul(anchor_features, proto.t())
            positive_mask = valid_class_indices.unsqueeze(0).eq(anchor_labels.unsqueeze(1))
            valid_anchor = positive_mask.any(dim=1)
            if not valid_anchor.any():
                continue

            sim = sim[valid_anchor]
            positive_mask = positive_mask[valid_anchor]
            pos_sim = sim.masked_fill(~positive_mask, -2.0).max(dim=1)[0]
            neg_sim = sim.masked_fill(positive_mask, -2.0).max(dim=1)[0]
            losses.append(F.softplus(neg_sim - pos_sim + self.margin))

        self._update_memory(embeddings, labels, m_labels)
        if not losses:
            return embeddings.new_zeros(())
        return torch.cat(losses).mean().to(embeddings.dtype)
