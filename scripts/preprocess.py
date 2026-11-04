"""Prepare fixed RNA targets and frozen UNI2-h nucleus embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from morphic.data import prepare_sample
from morphic.images import Slide
from morphic.model import embed, load_encoder, verify_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--out', type=Path, default=Path('outputs/prepared'))
    p.add_argument('--stage', choices=['all','targets','embeddings'], default='all')
    p.add_argument('--batch', type=int, default=8)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv('data/cohort.csv')
    cohort = cohort[(cohort.split.eq('validation')) | (cohort.split.eq('train') & cohort.in_b)]
    catalog = pd.read_csv('data/lr_catalog.csv')
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/lr_catalog.csv'),config['catalog_sha256'])
    verify_sha256(Path('data/anchors.csv.gz'),config['anchors_sha256'])
    verify_sha256(Path('data/cohort.csv'),config['cohort_sha256'])
    if len(catalog) != 1821:
        raise ValueError('Expected the frozen 1,821-mechanism catalog')
    anchors = pd.read_csv('data/anchors.csv.gz')
    if args.stage in ('all','targets'):
        for row in cohort.itertuples():
            ids = anchors.loc[anchors['sample'].eq(row.sample),'cell_id'].astype(str).tolist()
            print(row.sample, prepare_sample(row.sample, row.split, args.raw, args.out, catalog, ids,
                float(config['graph_radius_um']),int(config['graph_max_neighbors'])), flush=True)
    if args.stage in ('all','embeddings'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = args.raw/'models/uni2-h/pytorch_model.bin'
        verify_sha256(weights,config['encoder_sha256'])
        encoder = load_encoder(weights, device)
        for row in cohort.itertuples():
            folder = args.out/row.sample
            nodes = pd.read_parquet(folder/'nodes.parquet')
            meta = json.loads((folder/'meta.json').read_text())
            slide = Slide(args.raw/'wsis'/f'{row.sample}.tif', float(meta['mpp']))
            output = np.lib.format.open_memmap(folder/'embeddings.npy', mode='w+',
                                               dtype=np.float32, shape=(len(nodes),1536))
            try:
                for start in range(0, len(nodes), args.batch):
                    part = nodes.iloc[start:start+args.batch]
                    crops = np.stack([slide.crop((r.x_px,r.y_px)) for r in part.itertuples()])
                    masks = np.stack([slide.nucleus_mask((r.x_px,r.y_px),r.nucleus_wkb)
                                      for r in part.itertuples()])
                    x = torch.from_numpy(crops).permute(0,3,1,2)
                    output[start:start+len(part)] = embed(encoder,x,torch.from_numpy(masks)).cpu().numpy()
                output.flush()
            finally:
                slide.close()
            print(row.sample, 'embeddings', len(nodes), flush=True)


if __name__ == '__main__':
    main()
