"""Train the selected gated LR head on the fixed 37/6 HEST split."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.nn import functional as F

from morphic.model import GatedLRHead, direction_cases, sample_cases, verify_sha256

from morphic.dataset import Sample, epoch_batches, validation_rows


@torch.inference_mode()
def validation_brier(head, samples, device, seed=117, count=4096, batch=4096):
    head.eval()
    values = []
    for sample in samples:
        rows = validation_rows(sample, count, seed)
        errors = [[] for _ in sample.measured]
        for start in range(0, len(rows), batch):
            part = rows[start:start+batch]
            _, _, _, mat = sample.fetch(part)
            ctx = sample.context(head, part, device)
            for j, k in enumerate(sample.measured):
                y = (mat[:, int(k)].toarray().ravel() > .5).astype(np.uint8)
                pred = head.score_context(ctx, torch.full((len(part),), int(k), device=device)).cpu().numpy()
                errors[j].append((pred-y)**2)
        values.extend(float(np.concatenate(parts).mean()) for parts in errors)
    return float(np.mean(values))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', type=Path, default=Path('outputs/prepared'))
    p.add_argument('--out', type=Path, default=Path('outputs/trained'))
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=4096)
    args = p.parse_args()
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/cohort.csv'),config['cohort_sha256'])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if not 1 <= args.epochs <= config['epochs_max']:
        raise ValueError('Epochs must be between 1 and epochs_max')
    if args.batch<2 or args.batch%2:
        raise ValueError('Batch must be positive and even')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)
    cohort = pd.read_csv('data/cohort.csv')
    train_names = sorted(cohort.loc[cohort.split.eq('train'),'sample'])
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
    steps = sum(math.ceil(s.meta['directed_edges']/args.batch) for s in training)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs*steps)
    best, patience_best, stale = float('inf'), float('inf'), 0
    (args.out/'config.json').write_text(json.dumps(dict(config, epochs=args.epochs, batch=args.batch, train_samples=train_names, validation_samples=val_names), indent=2)+'\n')
    history = []
    for epoch in range(args.epochs):
        head.train()
        lr_rng = np.random.default_rng(np.random.SeedSequence([seed,epoch,1601]))
        batches = epoch_batches(training, args.batch, seed, epoch)
        seen = {s.sample:0 for s in training}
        for step,(sample,rows) in enumerate(batches,1):
            _, _, _, targets = sample.fetch(rows)
            local_rows = np.arange(len(rows))
            positions,mechanisms,labels,weights = sample_cases(targets,local_rows,sample.measured,lr_rng,negatives)
            context = sample.context(head,rows,device)
            roles = head.prepare_role_parameters()
            logits = head.score_context(context,torch.from_numpy(mechanisms).to(device),
                                        torch.from_numpy(positions).to(device),logits=True,role_parameters=roles)
            bce = F.binary_cross_entropy_with_logits(logits.float(),torch.from_numpy(labels).to(device),
                                                     reduction='none')
            loss = (bce*torch.from_numpy(weights).to(device)).sum()/len(rows)
            direction = direction_cases(targets,local_rows,sample.measured)
            if direction is not None:
                index,k,sign,w = direction
                ki = torch.from_numpy(k).to(device)
                forward = head.score_context(context,ki,torch.from_numpy(2*index).to(device),logits=True,role_parameters=roles)
                reverse = head.score_context(context,ki,torch.from_numpy(2*index+1).to(device),logits=True,role_parameters=roles)
                loss = loss + direction_weight*(F.softplus(-torch.from_numpy(sign).to(device)*(forward-reverse)) *
                                    torch.from_numpy(w).to(device)).sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step(); scheduler.step()
            seen[sample.sample] += len(rows)
            if step%100==0:
                print(f'epoch={epoch+1} step={step}/{steps} loss={loss.item():.5f}',flush=True)
        if any(seen[s.sample]!=s.meta['directed_edges'] for s in training):
            raise ValueError('Incomplete training edge pass')
        brier = validation_brier(head,validation,device,seed+100,config['validation_edges_per_sample'],args.batch)
        if not math.isfinite(brier):
            raise ValueError('Nonfinite validation Brier')
        history.append({'epoch':epoch+1,'validation_binary_brier':brier,
                        'training_edges_by_sample':seen,'training_edges_seen':sum(seen.values())})
        (args.out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if brier<best:
            best = brier
            torch.save({'config':{'encoder':'uni2-h','head_type':'gated','train_samples':train_names,
                                  'validation_samples':val_names,'mechanisms':len(catalog),'seed':seed,
                                  'direction_weight':direction_weight},
                        'head':{k:v.detach().cpu() for k,v in head.state_dict().items()},
                        'epoch':epoch,'validation':{'binary_brier':brier}},args.out/'best.pt')
        if brier < patience_best - config['early_stopping_min_delta']:
            patience_best, stale = brier, 0
        else:
            stale += 1
        print(history[-1],flush=True)
        if epoch+1>=config['early_stopping_min_epochs'] and stale>=config['early_stopping_patience']:
            break


if __name__ == '__main__':
    main()
