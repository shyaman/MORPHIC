"""The selected frozen UNI2-h encoder and gated LR head."""
from __future__ import annotations

import math
import hashlib
from pathlib import Path

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F


def verify_sha256(path, expected):
    digest = hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            digest.update(chunk)
    if digest.hexdigest()!=expected:
        raise ValueError(f'Unexpected SHA256 for {path}')


class RoleBranch(nn.Module):
    def __init__(self, features=1536, rank=128):
        super().__init__()
        self.cell_projection = nn.Linear(features, rank)
        self.gene_gate = nn.Linear(rank, rank)
        self.gene_bias = nn.Linear(rank, rank)
        self.output = nn.Sequential(nn.GELU(), nn.Linear(rank, rank))

    def read_projected(self, projected, gene):
        return self.output(torch.sigmoid(self.gene_gate(gene)) * projected + self.gene_bias(gene))


class GatedLRHead(nn.Module):
    def __init__(self, pairs, features=1536, rank=128):
        super().__init__()
        genes = sorted({gene for row in pairs.itertuples() for gene in
                        (row.ligand_subunits + ';' + row.receptor_subunits).split(';')})
        lookup = {gene: i + 1 for i, gene in enumerate(genes)}
        ligands = [row.ligand_subunits.split(';') for row in pairs.itertuples()]
        receptors = [row.receptor_subunits.split(';') for row in pairs.itertuples()]
        width = max(max(map(len, ligands)), max(map(len, receptors)))
        def encode(groups):
            return torch.tensor([[lookup[g] for g in group] + [0] * (width - len(group))
                                 for group in groups], dtype=torch.long)
        self.register_buffer('ligand_ids', encode(ligands))
        self.register_buffer('receptor_ids', encode(receptors))
        self.gene_embeddings = nn.Embedding(len(genes) + 1, rank, padding_idx=0)
        self.rank = rank
        self.genes = genes
        self.ligand_branch = RoleBranch(features, rank)
        self.receptor_branch = RoleBranch(features, rank)
        self.scorer = nn.Sequential(nn.Linear(3 * rank + 1, rank), nn.GELU(), nn.Linear(rank, 1))

    def _complex(self, ids):
        return self.gene_embeddings(ids).sum(1) / (ids != 0).sum(1).clamp_min(1)[:, None]

    def encode_context(self, sender, receiver, distance_um):
        return torch.cat((self.ligand_branch.cell_projection(sender.float()),
                          self.receptor_branch.cell_projection(receiver.float()),
                          (distance_um.float() / 40)[:, None]), dim=1)

    def prepare_role_parameters(self):
        """Differentiable tables, created anew for each optimizer step."""
        tables = []
        for ids, branch in ((self.ligand_ids, self.ligand_branch),
                            (self.receptor_ids, self.receptor_branch)):
            unique, inverse = torch.unique(ids, dim=0, return_inverse=True)
            gene = self._complex(unique)
            tables.append((torch.sigmoid(branch.gene_gate(gene))[inverse],
                           branch.gene_bias(gene)[inverse]))
        return tables

    def score_context(self, context, pair_index, edge_positions=None, logits=False, role_parameters=None):
        if edge_positions is not None:
            context = context[edge_positions]
        if role_parameters is None:
            ligand = self._complex(self.ligand_ids[pair_index])
            receptor = self._complex(self.receptor_ids[pair_index])
            left = self.ligand_branch.read_projected(context[:, :self.rank], ligand)
            right = self.receptor_branch.read_projected(context[:, self.rank:2*self.rank], receptor)
        else:
            (lg, lb), (rg, rb) = role_parameters
            left = self.ligand_branch.output(lg[pair_index] * context[:, :self.rank] + lb[pair_index])
            right = self.receptor_branch.output(rg[pair_index] * context[:, self.rank:2*self.rank] + rb[pair_index])
        score = self.scorer(torch.cat((left, right, left * right, context[:, -1:]), 1)).squeeze(1)
        if logits:
            return score
        return torch.sigmoid(score)


def load_head(path: Path, pairs, device):
    state = torch.load(path, map_location='cpu', weights_only=True)
    head = GatedLRHead(pairs)
    if state['config']['encoder'] != 'uni2-h' or state['config']['head_type'] != 'gated':
        raise ValueError('Checkpoint architecture does not match MORPHIC')
    for key in ('ligand_ids', 'receptor_ids'):
        if not torch.equal(getattr(head, key), state['head'][key]):
            raise ValueError('Checkpoint and LR catalog order disagree')
    head.load_state_dict(state['head'], strict=True)
    return head.to(device).eval()


def load_encoder(weights: Path, device):
    import timm
    model = timm.create_model('vit_giant_patch14_224', pretrained=False, img_size=224,
                              patch_size=14, depth=24, num_heads=24, init_values=1e-5,
                              embed_dim=1536, mlp_ratio=2.66667*2, num_classes=0,
                              no_embed_class=True, mlp_layer=timm.layers.SwiGLUPacked,
                              act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True)
    model.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def embed(encoder, pixels, masks):
    device = next(encoder.parameters()).device
    pixels = pixels.to(device).float() / 255
    mean = pixels.new_tensor([.485, .456, .406])[None, :, None, None]
    std = pixels.new_tensor([.229, .224, .225])[None, :, None, None]
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        tokens = encoder.forward_features((pixels - mean) / std)
    spatial = tokens[:, encoder.num_prefix_tokens:]
    side = math.isqrt(spatial.shape[1])
    if side * side != spatial.shape[1]:
        raise ValueError('Unexpected UNI2-h token count')
    features = spatial.transpose(1, 2).reshape(len(spatial), -1, side, side).float()
    weights = F.adaptive_avg_pool2d(masks.to(device)[:, None].float(), (side, side))
    total = weights.sum((2, 3))
    if (total <= 0).any():
        raise ValueError('Empty nucleus mask')
    return ((features * weights).sum((2, 3)) / total).float()


def sample_cases(matrix, rows, measured, rng, negatives=16):
    """All positives and the original 16 hard-mixture measured negatives per edge."""
    positions, mechanisms, labels, weights = [], [], [], []
    measured = np.asarray(measured, dtype=np.int64)
    dense = matrix[rows][:, measured].toarray()
    for position, values in enumerate(dense):
        positive = np.flatnonzero(values > .5)
        negative = np.flatnonzero(values <= .5)
        nonzero = np.flatnonzero((values > 0) & (values <= .5))
        zero = np.flatnonzero(values == 0)
        if len(nonzero) > 32:
            nonzero = nonzero[np.argsort(values[nonzero], kind='stable')[-32:]]
        n_nonzero = min(negatives // 2, len(nonzero))
        selected_negative = np.r_[rng.choice(nonzero, n_nonzero, replace=False),
            rng.choice(zero, min(negatives-n_nonzero, len(zero)), replace=False)]
        if len(selected_negative) < negatives:
            remaining = np.setdiff1d(negative, selected_negative)
            selected_negative = np.r_[selected_negative, rng.choice(remaining,
                min(negatives-len(selected_negative), len(remaining)), replace=False)]
        selected = np.r_[positive, selected_negative].astype(np.int64)
        positions.extend([position]*len(selected))
        mechanisms.extend(measured[selected])
        labels.extend(values[selected] > .5)
        weights.extend([1/len(selected)]*len(selected))
    return (np.asarray(positions, np.int64), np.asarray(mechanisms, np.int64),
            np.asarray(labels, np.float32), np.asarray(weights, np.float32))


def direction_cases(matrix, rows, measured):
    """Discordant labels for the two directions of each consecutive cell pair."""
    forward = (matrix[rows[::2]][:, measured] > .5).astype(np.int8)
    reverse = (matrix[rows[1::2]][:, measured] > .5).astype(np.int8)
    difference = (forward-reverse).tocoo()
    difference.eliminate_zeros()
    if not len(difference.data):
        return None
    counts = np.bincount(difference.row, minlength=len(rows)//2)
    informative = np.count_nonzero(counts)
    weights = (1/(counts[difference.row]*informative)).astype(np.float32)
    return difference.row, np.asarray(measured)[difference.col], difference.data.astype(np.float32), weights
