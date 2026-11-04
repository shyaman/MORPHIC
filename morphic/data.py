"""Fixed HEST cohort, directed neighborhood graph, and RNA proxy labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.spatial import cKDTree


def graph(xy_um, anchors, radius=40., neighbors=12):
    tree = cKDTree(xy_um)
    distance, index = tree.query(xy_um[anchors], k=neighbors+1, distance_upper_bound=radius)
    source = np.repeat(anchors, neighbors+1)
    target = index.reshape(-1)
    keep = (target < len(xy_um)) & (source != target) & np.isfinite(distance.reshape(-1))
    pairs = np.column_stack((source[keep], target[keep]))
    if not len(pairs):
        raise ValueError('No candidate cell pairs')
    pairs = np.unique(np.concatenate((pairs, pairs[:, ::-1])), axis=0)
    used = np.unique(pairs.reshape(-1))
    pairs = np.searchsorted(used, pairs)
    d = np.linalg.norm(xy_um[used[pairs[:, 0]]] - xy_um[used[pairs[:, 1]]], axis=1)
    return used, pairs, d.astype(np.float32)


def measured_pairs(catalog, genes):
    return [i for i, row in enumerate(catalog.itertuples()) if
            set((row.ligand_subunits+';'+row.receptor_subunits).split(';')) <= genes]


def prepare_sample(sample, split, raw, out, catalog, anchor_ids, radius=40., neighbors=12):
    """Read released HEST files; keep only nuclei and genes needed for this graph."""
    root = Path(raw)
    info = json.loads((root/'metadata'/f'{sample}.json').read_text())
    mpp = float(info['pixel_size_um_estimated'])
    nuclei = pq.read_table(root/'xenium_seg'/f'{sample}_xenium_nucleus_seg.parquet')
    cells = pq.read_table(root/'xenium_seg'/f'{sample}_xenium_cell_seg.parquet')
    def ids(table):
        name = next((n for n in ('cell_id','__index_level_0__','__null_dask_index__')
                     if n in table.column_names), None)
        if name is None:
            raise ValueError(f'{sample}: no cell ID column')
        values = table[name].to_pylist()
        return pd.Index([x.decode() if isinstance(x, bytes) else str(x) for x in values])
    nucleus_ids, cell_ids = ids(nuclei), ids(cells)
    unique = ~nucleus_ids.duplicated(keep=False) & nucleus_ids.isin(cell_ids[~cell_ids.duplicated(keep=False)])
    nucleus_rows = np.flatnonzero(unique)
    names = nucleus_ids[unique]
    import shapely
    polygons = shapely.from_wkb(nuclei['geometry'].take(nucleus_rows).to_pylist())
    centroids = shapely.centroid(polygons)
    xy = np.column_stack((shapely.get_x(centroids), shapely.get_y(centroids)))
    margin = 56/mpp
    valid = (np.isfinite(xy).all(1) & (shapely.area(polygons)>0) &
             (xy[:, 0]>=margin) & (xy[:, 0]<int(info['fullres_px_width'])-margin) &
             (xy[:, 1]>=margin) & (xy[:, 1]<int(info['fullres_px_height'])-margin))
    nucleus_rows, names, polygons, xy = nucleus_rows[valid], names[valid], polygons[valid], xy[valid]
    # First pass counts all accepted transcripts, including non-LR genes, for CPM10k.
    lookup = pd.Index(names)
    totals = np.zeros(len(names), dtype=np.int32)
    transcript_file = root/'transcripts'/f'{sample}_transcripts.parquet'
    for batch in pq.ParquetFile(transcript_file).iter_batches(batch_size=1_000_000,
                                                               columns=['cell_id','feature_name','qv']):
        frame = batch.to_pandas()
        cell = frame.cell_id.str.decode() if isinstance(frame.cell_id.iloc[0], bytes) else frame.cell_id.astype(str)
        gene = frame.feature_name.str.decode() if isinstance(frame.feature_name.iloc[0], bytes) else frame.feature_name.astype(str)
        row = lookup.get_indexer(cell)
        good = ((row>=0) & (frame.qv.to_numpy()>=20) &
                ~gene.str.startswith(('NegControl','Unassigned','Deprecated')).to_numpy())
        np.add.at(totals, row[good], 1)
    eligible = np.flatnonzero(totals>=30)
    if not len(eligible):
        raise ValueError(f'{sample}: no eligible nuclei')
    chosen = lookup.get_indexer(pd.Index(anchor_ids))
    if (chosen<0).any() or not np.isin(chosen,eligible).all():
        raise ValueError(f'{sample}: frozen anchor IDs do not match downloaded HEST data')
    chosen = np.sort(chosen)
    used, pairs, distance = graph(xy[eligible]*mpp, np.searchsorted(eligible, chosen),radius,neighbors)
    used = eligible[used]
    names, xy, polygons, totals = names[used], xy[used], polygons[used], totals[used]
    sample_dir = Path(out)/sample
    sample_dir.mkdir(parents=True, exist_ok=True)
    nodes = pd.DataFrame(dict(cell_id=names.astype(str), x_px=xy[:, 0], y_px=xy[:, 1],
                              nucleus_wkb=shapely.to_wkb(polygons), transcript_count=totals))
    edges = pd.DataFrame(dict(sender=pairs[:, 0], receiver=pairs[:, 1], distance_um=distance))
    genes = set()
    for batch in pq.ParquetFile(transcript_file).iter_batches(batch_size=1_000_000, columns=['feature_name']):
        genes.update(x.decode() if isinstance(x, bytes) else x for x in batch.column(0).unique().to_pylist())
    measured = measured_pairs(catalog, genes)
    required = sorted({g for k in measured for g in
                       (catalog.iloc[k].ligand_subunits+';'+catalog.iloc[k].receptor_subunits).split(';')})
    gene_index = {g:i for i,g in enumerate(required)}
    node_index = pd.Index(names)
    counts = np.zeros((len(names), len(required)), dtype=np.int32)
    for batch in pq.ParquetFile(transcript_file).iter_batches(batch_size=1_000_000,
                                                               columns=['cell_id','feature_name','qv']):
        frame = batch.to_pandas()
        cell = frame.cell_id.str.decode() if isinstance(frame.cell_id.iloc[0], bytes) else frame.cell_id.astype(str)
        gene = frame.feature_name.str.decode() if isinstance(frame.feature_name.iloc[0], bytes) else frame.feature_name.astype(str)
        row = node_index.get_indexer(cell)
        col = gene.map(gene_index).fillna(-1).to_numpy(np.int32)
        good = (row>=0) & (col>=0) & (frame.qv.to_numpy()>=20)
        np.add.at(counts, (row[good], col[good]), 1)
    expression = np.log1p(counts.astype(np.float32) * (10000/np.maximum(totals,1))[:, None])
    sender, receiver = pairs.T
    rows, cols, values = [], [], []
    for k in measured:
        mechanism = catalog.iloc[k]
        left = np.minimum.reduce([expression[sender, gene_index[g]] for g in mechanism.ligand_subunits.split(';')])
        right = np.minimum.reduce([expression[receiver, gene_index[g]] for g in mechanism.receptor_subunits.split(';')])
        score = left*right
        positive = np.flatnonzero(score>0)
        ratio = score[positive]/float(mechanism.threshold)
        rows.append(positive.astype(np.int32))
        cols.append(np.full(len(positive),k,np.int32))
        values.append((ratio/(1+ratio)).astype(np.float32))
    matrix = sparse.coo_matrix((np.concatenate(values) if values else np.empty(0,np.float32),
                                (np.concatenate(rows) if rows else np.empty(0,np.int32),
                                 np.concatenate(cols) if cols else np.empty(0,np.int32))),
                               shape=(len(edges),len(catalog)), dtype=np.float32).tocsr()
    nodes.to_parquet(sample_dir/'nodes.parquet', index=False)
    edges.to_parquet(sample_dir/'edges.parquet', index=False)
    sparse.save_npz(sample_dir/'targets.npz', matrix)
    (sample_dir/'meta.json').write_text(json.dumps(dict(sample=sample, split=split, mpp=mpp,
            measured=measured, nodes=len(nodes), edges=len(edges)), indent=2)+'\n')
    return len(nodes), len(edges)
