import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalityBatchHardTripletLoss(nn.Module):
    """
    Batch-hard triplet loss tailored to visible-infrared retrieval.
    For each anchor, positives are same-identity samples from other modalities and
    negatives are different-identity samples from other modalities.
    """

    def __init__(self, margin=0.3, soft_margin=True):
        super().__init__()
        self.margin = margin
        self.soft_margin = soft_margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, embeddings, id_labels, m_labels):
        if embeddings.size(0) <= 1:
            return embeddings.new_zeros(())

        embeddings = F.normalize(embeddings, p=2, dim=1)
        dist_mat = 1.0 - torch.matmul(embeddings, embeddings.t())

        same_id = id_labels.unsqueeze(0).eq(id_labels.unsqueeze(1))
        diff_id = ~same_id
        diff_modal = m_labels.unsqueeze(0).ne(m_labels.unsqueeze(1))

        positive_mask = same_id & diff_modal
        negative_mask = diff_id & diff_modal
        valid_anchor = positive_mask.any(dim=1) & negative_mask.any(dim=1)
        if not valid_anchor.any():
            return embeddings.new_zeros(())

        pos_dist = dist_mat.masked_fill(~positive_mask, -1.0)
        neg_dist = dist_mat.masked_fill(~negative_mask, 2.0)

        hard_pos = pos_dist[valid_anchor].max(dim=1)[0]
        hard_neg = neg_dist[valid_anchor].min(dim=1)[0]

        if self.soft_margin:
            return F.softplus(hard_pos - hard_neg).mean()

        target = torch.ones_like(hard_pos)
        return self.ranking_loss(hard_neg, hard_pos, target)


class CosFaceProxyLoss(nn.Module):
    """
    CosFace-style proxy classification loss using the model classifier weights.

    Unlike mini-batch triplet losses, this compares each embedding against all
    training identities through the classifier weight matrix, which gives a
    stronger all-class separation signal for Rank-1 retrieval fine-tuning.
    """

    def __init__(self, scale=32.0, margin=0.2):
        super().__init__()
        if scale <= 0:
            raise ValueError('scale must be positive.')
        if margin < 0:
            raise ValueError('margin must be non-negative.')
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, classifier_weight, labels):
        if embeddings.size(0) == 0:
            return embeddings.new_zeros(())

        features = F.normalize(embeddings.float(), p=2, dim=1)
        proxies = F.normalize(classifier_weight.float(), p=2, dim=1)
        logits = torch.matmul(features, proxies.t())

        labels = labels.long()
        valid = (labels >= 0) & (labels < logits.size(1))
        if not valid.any():
            return embeddings.new_zeros(())

        logits = logits[valid]
        labels = labels[valid]
        target_logits = logits.gather(1, labels.view(-1, 1))
        logits = logits.scatter(1, labels.view(-1, 1), target_logits - self.margin)
        return F.cross_entropy(logits * self.scale, labels).to(embeddings.dtype)
