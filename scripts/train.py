"""Train the selected gated LR head on the fixed 37/6 HEST split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import yaml
from torch.nn import functional as F

from morphic.model import GatedLRHead, direction_cases, sample_cases, verify_sha256


class Sample:
    def __init__(self, root, name):
        folder = root/name
        self.name = name
        self.edges = pd.read_parquet(folder/'edges.parquet')
        self.matrix = sparse.load_npz(folder/'targets.npz').tocsr()
        self.measured = json.loads((folder/'meta.json').read_text())['measured']
        self.embeddings = np.load(folder/'embeddings.npy', mmap_mode='r')
        if self.matrix.shape[0] != len(self.edges) or self.embeddings.shape[0] != len(pd.read_parquet(folder/'nodes.parquet')):
            raise ValueError(f'{name}: prepared rows disagree')
        sender = self.edges.sender.to_numpy(np.int64)
        receiver = self.edges.receiver.to_numpy(np.int64)
        width = len(self.embeddings)
        key = sender*width+receiver
        order = np.argsort(key)
        reverse = receiver*width+sender
        pos = np.searchsorted(key[order], reverse)
        if (pos==len(key)).any() or not np.array_equal(key[order][pos],reverse):
            raise ValueError(f'{name}: graph lacks reverse edges')
        canonical = np.flatnonzero(sender<receiver)
        self.pairs = np.column_stack((canonical, order[pos[canonical]]))
        if len(self.pairs)*2 != len(self.edges):
            raise ValueError(f'{name}: invalid paired graph')

    def context(self, head, rows, device):
        e = self.edges.iloc[rows]
        s = torch.from_numpy(np.asarray(self.embeddings[e.sender.to_numpy(np.int64)])).to(device)
        r = torch.from_numpy(np.asarray(self.embeddings[e.receiver.to_numpy(np.int64)])).to(device)
        d = torch.from_numpy(e.distance_um.to_numpy(np.float32)).to(device)
        return head.encode_context(s,r,d)


@torch.inference_mode()
def validation_brier(head, samples, device, seed=117, cap=32):
    rng = np.random.default_rng(seed)
    head.eval()
    values = []
    for sample in samples:
        csc = sample.matrix.tocsc()
        for k in sample.measured:
            col = csc.getcol(k).toarray().ravel()
            positive = np.flatnonzero(col>.5)
            negative = np.flatnonzero(col<=.5)
            if len(positive)<5 or len(negative)<5:
                continue
            pi = rng.choice(positive,min(cap,len(positive)),replace=False)
            ni = rng.choice(negative,min(cap,len(negative)),replace=False)
            rows = np.r_[pi,ni]
            weights = np.r_[np.full(len(pi),len(positive)/len(pi)),
                            np.full(len(ni),len(negative)/len(ni))]
            predictions = []
            for part in np.array_split(rows,max(1,math.ceil(len(rows)/512))):
                ctx = sample.context(head,part,device)
                predictions.append(head.score_context(ctx,torch.full((len(part),),k,device=device)).cpu().numpy())
            pred = np.concatenate(predictions)
            values.append(float(np.average((pred-(col[rows]>.5))**2,weights=weights)))
    if not values:
        raise ValueError('No evaluable validation mechanisms')
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=Path('outputs/prepared'))
    p.add_argument('--out', type=Path, default=Path('outputs/trained'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=512)
    args = p.parse_args()
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/cohort.csv'),config['cohort_sha256'])
    if args.batch<2 or args.batch%2:
        raise ValueError('Batch must be positive and even')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)
    cohort = pd.read_csv('data/cohort.csv')
    train_names = sorted(cohort.loc[cohort.split.eq('train') & cohort.in_b,'sample'])
    val_names = sorted(cohort.loc[cohort.split.eq('validation'),'sample'])
    if len(train_names)!=37 or len(val_names)!=6:
        raise ValueError('Expected 37 train and six validation samples')
    training = [Sample(args.prepared,s) for s in train_names]
    validation = [Sample(args.prepared,s) for s in val_names]
    catalog = pd.read_csv('data/lr_catalog.csv')
    verify_sha256(Path('data/lr_catalog.csv'),config['catalog_sha256'])
    seed = int(config['seed'])
    direction_weight = float(config['direction_loss_weight'])
    negatives = int(config['negative_mechanisms_per_edge'])
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    head = GatedLRHead(catalog).to(device)
    optimizer = torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01)
    steps = sum(math.ceil(len(s.pairs)/(args.batch//2)) for s in training)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=int(config['epochs_max'])*steps)
    best, stale = float('inf'), 0
    history = []
    for epoch in range(args.epochs):
        head.train()
        rng = np.random.default_rng(np.random.SeedSequence([seed,epoch]))
        lr_rng = np.random.default_rng(np.random.SeedSequence([seed,epoch,1601]))
        batches = []
        for sample in training:
            pairs = sample.pairs[rng.permutation(len(sample.pairs))]
            for start in range(0,len(pairs),args.batch//2):
                batches.append((sample,pairs[start:start+args.batch//2].reshape(-1)))
        rng.shuffle(batches)
        seen = {s.name:0 for s in training}
        for step,(sample,rows) in enumerate(batches,1):
            positions,mechanisms,labels,weights = sample_cases(sample.matrix,rows,sample.measured,lr_rng,negatives)
            context = sample.context(head,rows,device)
            logits = head.score_context(context,torch.from_numpy(mechanisms).to(device),
                                        torch.from_numpy(positions).to(device),logits=True)
            bce = F.binary_cross_entropy_with_logits(logits.float(),torch.from_numpy(labels).to(device),
                                                     reduction='none')
            loss = (bce*torch.from_numpy(weights).to(device)).sum()/len(rows)
            direction = direction_cases(sample.matrix,rows,sample.measured)
            if direction is not None:
                index,k,sign,w = direction
                ki = torch.from_numpy(k).to(device)
                forward = head.score_context(context,ki,torch.from_numpy(2*index).to(device),logits=True)
                reverse = head.score_context(context,ki,torch.from_numpy(2*index+1).to(device),logits=True)
                loss = loss + direction_weight*(F.softplus(-torch.from_numpy(sign).to(device)*(forward-reverse)) *
                                    torch.from_numpy(w).to(device)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step(); scheduler.step()
            seen[sample.name] += len(rows)
            if step%100==0:
                print(f'epoch={epoch+1} step={step}/{steps} loss={loss.item():.5f}',flush=True)
        if any(seen[s.name]!=len(s.edges) for s in training):
            raise ValueError('Incomplete training edge pass')
        brier = validation_brier(head,validation,device)
        history.append({'epoch':epoch+1,'validation_binary_brier':brier})
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if brier<best:
            best,stale = brier,0
            torch.save({'config':{'encoder':'uni2-h','head_type':'gated','train_samples':train_names,
                                  'validation_samples':val_names,'mechanisms':len(catalog),'seed':seed,
                                  'direction_weight':direction_weight},
                        'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                        'epoch':epoch,'validation':{'binary_brier':brier}},args.out/'best.pt')
        else:
            stale += 1
        print(history[-1],flush=True)
        if epoch+1>=3 and stale>=2:
            break


if __name__ == '__main__':
    main()
