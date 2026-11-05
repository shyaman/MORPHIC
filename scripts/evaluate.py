"""Evaluate validation or held-out slides using complete outgoing sender neighborhoods."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.stats import rankdata

from morphic.model import load_head, verify_sha256
from morphic.dataset import Sample
from morphic.data import select_cohort
from scipy.special import expit


def average_precision(y, score):
    order = np.argsort(-score, kind='stable')
    labels, ranked = y[order], score[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)), len(ranked)-1]
    positives = np.cumsum(labels)[ends]
    return float(np.sum(np.diff(np.r_[0, positives])*positives/(ends+1))/y.sum())


def expected_random_ap(n, positives):
    harmonic = np.sum(1/np.arange(1, n+1))
    return float((harmonic+(positives-1)*(n-harmonic)/(n-1))/n)


def sender_ap_excess(sender, y, score):
    order = np.argsort(sender, kind='stable')
    starts = np.r_[0, np.flatnonzero(np.diff(sender[order]))+1]
    stops = np.r_[starts[1:], len(order)]
    positives = np.add.reduceat(y[order].astype(int), starts)
    valid = (positives>0) & (positives<stops-starts)
    groups = [order[a:b] for a,b in zip(starts[valid], stops[valid])]
    if not groups:
        return 0, np.nan
    indices = np.concatenate(groups)
    lengths = np.array([len(g) for g in groups])
    group_id = np.repeat(np.arange(len(groups)), lengths)
    local_starts = np.r_[0, np.cumsum(lengths)[:-1]]
    labels = y[indices]
    ranked_score = score[indices]
    order = np.lexsort((-ranked_score, group_id))
    labels, ranked_score = labels[order], ranked_score[order]
    cumulative = labels.cumsum()
    true_positive = cumulative-np.repeat(cumulative[local_starts]-labels[local_starts], lengths)
    ends = np.r_[(group_id[1:]!=group_id[:-1]) | (ranked_score[1:]!=ranked_score[:-1]), True]
    end_group, end_positive = group_id[ends], true_positive[ends]
    previous = np.r_[0, end_positive[:-1]]
    previous[np.r_[True, end_group[1:]!=end_group[:-1]]] = 0
    ranks = np.arange(len(indices))-np.repeat(local_starts, lengths)+1
    ap = np.bincount(end_group, weights=(end_positive-previous)*end_positive/ranks[ends],
                     minlength=len(groups))/positives[valid]
    random_ap = np.array([expected_random_ap(int(n), int(p)) for n,p in
                          zip(lengths, positives[valid])])
    return len(groups), float(np.mean(ap-random_ap))


def metrics(y, score, sender, pairs, direction_y=None, direction_score=None):
    y = np.asarray(y, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    positive, n = int(y.sum()), len(y)
    prevalence = positive/n
    clipped = np.clip(score, 1e-12, 1-1e-12)
    bins = np.minimum((score*20).astype(int), 19)
    predicted = np.bincount(bins, weights=score, minlength=20)
    observed = np.bincount(bins, weights=y, minlength=20)
    ece = np.sum(np.abs(predicted-observed))/n
    result = dict(edges=n, positives=positive, prevalence=prevalence,
                  brier=float(np.mean((score-y)**2)),
                  log_loss=float(-np.mean(y*np.log(clipped)+(~y)*np.log1p(-clipped))),
                  ece=float(ece), ranking_eligible=positive>=5 and n-positive>=5,
                  auroc=np.nan, average_precision=np.nan, ap_excess=np.nan,
                  informative_senders=0, macro_sender_ap_excess=np.nan)
    if result['ranking_eligible']:
        ranks = rankdata(score, method='average')
        result['auroc'] = float((ranks[y].sum()-positive*(positive+1)/2)/(positive*(n-positive)))
        result['average_precision'] = average_precision(y, score)
        result['ap_excess'] = result['average_precision']-prevalence
    if 0<positive<n:
        result['informative_senders'], result['macro_sender_ap_excess'] = sender_ap_excess(sender,y,score)
    if direction_y is not None:
        y, score = direction_y, direction_score
    forward, reverse = pairs.T
    discordant = y[forward]!=y[reverse]
    gap = np.where(y[forward[discordant]],
                   score[forward[discordant]]-score[reverse[discordant]],
                   score[reverse[discordant]]-score[forward[discordant]])
    result['direction_cases'] = int(discordant.sum())
    result['direction_accuracy'] = float(np.mean(np.where(gap==0,.5,gap>0))) if len(gap) else np.nan
    return result


@torch.inference_mode()
def score_sample(head, sample, device, path, batch, rows):
    measured = np.asarray(sample.measured, dtype=np.int64)
    scores = np.lib.format.open_memmap(path, mode='w+', dtype=np.float32,
                                      shape=(len(rows), len(measured)))
    roles = head.prepare_role_parameters()
    for start in range(0,len(rows),batch):
        stop = min(start+batch,len(rows))
        context = sample.context(head,rows[start:stop],device)
        for k0 in range(0,len(measured),64):
            ids = measured[k0:k0+64]
            positions = torch.arange(stop-start,device=device).repeat_interleave(len(ids))
            mechanisms = torch.from_numpy(np.tile(ids,stop-start)).to(device)
            values = expit(head.score_context(context,mechanisms,positions,logits=True,role_parameters=roles).cpu().numpy())
            scores[start:stop,k0:k0+len(ids)] = values.reshape(stop-start,len(ids))
    scores.flush()
    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('outputs/prepared'))
    p.add_argument('--checkpoint',type=Path,default=Path('models/morphic_head.pt'))
    p.add_argument('--out',type=Path,default=Path('outputs/evaluation'))
    p.add_argument('--batch',type=int,default=256)
    p.add_argument('--split', choices=['validation','test'], default='test')
    p.add_argument('--samples', nargs='+')
    p.add_argument('--senders', type=int, default=20000, help='Sampled senders per slide; 0 uses all')
    p.add_argument('--seed', type=int, default=17)
    args = p.parse_args()
    if args.senders < 0:
        p.error('--senders must be nonnegative')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.batch<1:
        p.error('--batch must be positive')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/cohort.csv'),config['cohort_sha256'])
    verify_sha256(Path('data/lr_catalog.csv'),config['catalog_sha256'])
    catalog = pd.read_csv('data/lr_catalog.csv')
    cohort = select_cohort([args.split], args.samples)
    names = cohort['sample'].tolist()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.checkpoint.resolve() == Path('models/morphic_head.pt').resolve():
        manifest = json.loads(Path('models/manifest.json').read_text())
        verify_sha256(args.checkpoint, manifest['files']['models/morphic_head.pt'])
    head = load_head(args.checkpoint,catalog,device)
    args.out.mkdir(parents=True)
    all_rows, summaries = [], []
    for name in names:
        sample = Sample(args.prepared,name)
        path = args.out/f'{name}_scores.tmp.npy'
        sender, _ = sample.edge_arrays()
        n = sample.meta['graph_nodes']
        rng = np.random.default_rng(int(hashlib.sha256(name.encode()).hexdigest()[:8],16)+args.seed)
        selected = np.sort(rng.choice(n, min(args.senders,n), replace=False)) if args.senders else np.arange(n)
        keep = np.isin(sender, selected)
        selected_rows = np.flatnonzero(keep)
        pairs = sample.paired_rows()
        # Extra reverse rows are used only for the direction metric.
        pairs = pairs[keep[pairs].any(axis=1)]
        scored_rows = np.unique(pairs.ravel())
        local_pairs = np.searchsorted(scored_rows, pairs)
        local_selected = np.searchsorted(scored_rows, selected_rows)
        scores = score_sample(head,sample,device,path,args.batch,scored_rows)
        _, _, _, targets = sample.fetch(scored_rows)
        csc = targets.tocsc()
        del targets
        rows = []
        for j,k in enumerate(sample.measured):
            y = csc.getcol(k).toarray().ravel()>.5
            row = metrics(y[local_selected],scores[local_selected,j],sender[selected_rows],local_pairs,
                          direction_y=y,direction_score=scores[:,j])
            rows.append(dict(sample=name,mechanism=catalog.interaction_name.iloc[k],**row))
        del scores
        path.unlink()
        frame = pd.DataFrame(rows)
        all_rows.extend(rows)
        cases = frame.direction_cases.sum()
        ranked = frame.loc[frame.ranking_eligible & frame.informative_senders.gt(0)]
        summaries.append(dict(sample=name,split=args.split,independence_group=cohort.set_index('sample').loc[name,'independence_group'],
                              sampled_senders=len(selected), evaluated_edges=len(selected_rows), measured_mechanisms=len(frame),
                              ranking_mechanisms=len(ranked),
                              **{key:float(frame[key].mean()) for key in ('brier','log_loss','ece')},
                              **{key:float(ranked[key].mean()) for key in
                                 ('auroc','average_precision','ap_excess','macro_sender_ap_excess')},
                              direction_cases=int(cases),
                              direction_accuracy=float(np.average(frame.direction_accuracy.fillna(0),
                                    weights=frame.direction_cases)) if cases else np.nan))
        print(f'{name}: {len(frame)} measured mechanisms, {len(selected_rows)} edges',flush=True)
    pd.DataFrame(all_rows).to_csv(args.out/'metrics.csv',index=False)
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out/'summary.csv',index=False)
    metric_names = ['brier','log_loss','ece','auroc','average_precision','ap_excess',
                    'macro_sender_ap_excess','direction_accuracy']
    grouped = summary.groupby('independence_group')[metric_names].mean()
    grouped.to_csv(args.out/'groups.csv')
    pd.DataFrame([dict(aggregation='equal_weight_source_groups', groups=len(grouped),
                      **grouped.mean().to_dict())]).to_csv(args.out/'overall.csv',index=False)
    (args.out/'run.json').write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
        samples=names,split=args.split,senders=args.senders,seed=args.seed,
        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        direction_scope='unordered pairs incident to sampled senders; reverse rows excluded from ranking',
        grouping='conservative GSE315411 TMA block; 26 groups for the full test cohort',
        score_mode='raw_sigmoid',ranking_min_positive=5,
        ranking_min_negative=5,probability_bins=20,
        checkpoint_selection='validation_binary_brier'),indent=2)+'\n')


if __name__ == '__main__':
    main()
