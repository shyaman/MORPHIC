"""Prepare fixed RNA targets and frozen UNI2-h nucleus embeddings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from morphic.data import prepare_sample, select_cohort
from morphic.images import Slide
from morphic.model import embed, load_encoder, verify_sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw'))
    p.add_argument('--out', type=Path, default=Path('outputs/prepared'))
    p.add_argument('--stage', choices=['all','targets','embeddings'], default='all')
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--split', nargs='+', choices=['train','validation','test'], default=['train','validation'])
    p.add_argument('--samples', nargs='+')
    args = p.parse_args()
    if args.batch < 1:
        p.error('--batch must be positive')
    args.out.mkdir(parents=True, exist_ok=True)
    cohort = select_cohort(args.split, args.samples)
    catalog = pd.read_csv('data/lr_catalog.csv')
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/lr_catalog.csv'),config['catalog_sha256'])
    verify_sha256(Path('data/cohort.csv'),config['cohort_sha256'])
    if len(catalog) != 1821:
        raise ValueError('Expected the frozen 1,821-mechanism catalog')
    if args.stage in ('all','targets'):
        for row in cohort.itertuples():
            print(row.sample, prepare_sample(row.sample, row.split, args.raw, args.out, catalog,
                float(config['graph_radius_um']),int(config['graph_max_neighbors']),int(config['chunk_edges'])), flush=True)
    if args.stage in ('all','embeddings'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = args.raw/'models/uni2-h/pytorch_model.bin'
        verify_sha256(weights,config['encoder_sha256'])
        encoder = load_encoder(weights, device)
        for row in cohort.itertuples():
            folder = args.out/row.sample
            nodes = pd.read_parquet(folder/'nodes.parquet')
            meta = json.loads((folder/'complete.json').read_text())
            slide = Slide(args.raw/'wsis'/f'{row.sample}.tif', float(meta['mpp']))
            if (folder/'embeddings.npy').exists():
                raise FileExistsError(f'Embeddings already exist: {folder}')
            output = np.lib.format.open_memmap(folder/'embeddings.partial.npy', mode='w+',
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
                del output
                (folder/'embeddings.partial.npy').rename(folder/'embeddings.npy')
            finally:
                slide.close()
            print(row.sample, 'embeddings', len(nodes), flush=True)


if __name__ == '__main__':
    main()
