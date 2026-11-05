"""Fixed HEST cohort, directed neighborhood graph, and RNA proxy labels."""
from __future__ import annotations

import json
import hashlib
import tempfile
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


def prepare_sample(sample, split, raw, out, catalog, radius=40., neighbors=12, chunk_edges=100000):
    """Read released HEST files; keep only nuclei and genes needed for this graph."""
    destination = Path(out)/sample
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
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
    valid = (np.isfinite(xy).all(1) & (np.isfinite(shapely.area(polygons))) & (shapely.area(polygons)>0) &
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
    coordinates = xy[eligible]*mpp
    tree = cKDTree(coordinates)
    distance, index = tree.query(coordinates, k=neighbors+1, distance_upper_bound=radius, workers=4)
    source = np.repeat(np.arange(len(eligible), dtype=np.int32), neighbors+1)
    target = index.ravel()
    keep = (target < len(eligible)) & (source != target) & np.isfinite(distance.ravel())
    arcs = sparse.csr_matrix((np.ones(keep.sum(), np.int8), (source[keep], target[keep])),
                             shape=(len(eligible), len(eligible)))
    arcs.data.fill(1)
    paired = arcs.maximum(arcs.T).tocsr()
    paired.sort_indices()
    used = np.flatnonzero(np.diff(paired.indptr)>0)
    if not len(used):
        raise ValueError(f'{sample}: no graph edges')
    remap = np.full(len(eligible), -1, np.int32)
    remap[used] = np.arange(len(used), dtype=np.int32)
    selected = eligible[used]
    names, xy, polygons, totals = names[selected], xy[selected], polygons[selected], totals[selected]
    Path(out).mkdir(parents=True, exist_ok=True)
    sample_dir = Path(tempfile.mkdtemp(prefix=f'.{sample}.', dir=out))
    nodes = pd.DataFrame(dict(cell_id=names.astype(str), x_px=xy[:, 0], y_px=xy[:, 1],
                              nucleus_wkb=shapely.to_wkb(polygons), transcript_count=totals))
    nodes.to_parquet(sample_dir/'nodes.parquet', index=False)
    genes = set()
    for batch in pq.ParquetFile(transcript_file).iter_batches(batch_size=1_000_000, columns=['feature_name']):
        genes.update(x.decode() if isinstance(x, bytes) else x for x in batch.column(0).unique().to_pylist())
    genes = {g for g in genes if isinstance(g, str) and not g.startswith(('NegControl','Unassigned','Deprecated'))}
    measured = measured_pairs(catalog, genes)
    if not measured:
        raise ValueError(f'{sample}: no measured mechanisms')
    required = sorted({g for k in measured for g in
                       (catalog.iloc[k].ligand_subunits+';'+catalog.iloc[k].receptor_subunits).split(';')})
    gene_index = {g:i for i,g in enumerate(required)}
    node_index = pd.Index(names)
    counts = sparse.csr_matrix((len(names), len(required)), dtype=np.int32)
    for batch in pq.ParquetFile(transcript_file).iter_batches(batch_size=1_000_000,
                                                               columns=['cell_id','feature_name','qv']):
        frame = batch.to_pandas()
        cell = frame.cell_id.str.decode() if isinstance(frame.cell_id.iloc[0], bytes) else frame.cell_id.astype(str)
        gene = frame.feature_name.str.decode() if isinstance(frame.feature_name.iloc[0], bytes) else frame.feature_name.astype(str)
        row = node_index.get_indexer(cell)
        col = gene.map(gene_index).fillna(-1).to_numpy(np.int32)
        good = (row>=0) & (col>=0) & (frame.qv.to_numpy()>=20)
        counts += sparse.coo_matrix((np.ones(good.sum(), np.int32), (row[good], col[good])), shape=counts.shape).tocsr()
    counts = counts.tocsc()
    scale = 10000.0/np.maximum(totals.astype(np.float32), 1)
    # Match the reference's per-gene float32 normalization and log1p order.
    expression = {g: np.log1p(counts.getcol(j).toarray().ravel().astype(np.float32)*scale)
                  for g,j in gene_index.items()}
    shards = []
    chunk_first = edge_start = 0
    for row_end in range(1, len(paired.indptr)):
        if paired.indptr[row_end]-paired.indptr[chunk_first] < chunk_edges and row_end < len(paired.indptr)-1:
            continue
        arc_start, arc_stop = int(paired.indptr[chunk_first]), int(paired.indptr[row_end])
        if arc_start == arc_stop:
            continue
        original_sender = np.repeat(np.arange(chunk_first,row_end,dtype=np.int32),
                                     np.diff(paired.indptr[chunk_first:row_end+1]))
        original_receiver = paired.indices[arc_start:arc_stop]
        sender, receiver = remap[original_sender], remap[original_receiver]
        distance = np.linalg.norm(coordinates[original_sender]-coordinates[original_receiver], axis=1).astype(np.float32)
        index = len(shards)
        edges = pd.DataFrame(dict(edge_local=np.arange(edge_start,edge_start+len(sender)),
                                  sender=sender, receiver=receiver, distance_um=distance))
        edges.to_parquet(sample_dir/f'edges_{index:04d}.parquet', index=False)
        rows, cols, values = [], [], []
        for k in measured:
            mechanism = catalog.iloc[k]
            left = np.minimum.reduce([expression[g][sender] for g in mechanism.ligand_subunits.split(';')])
            right = np.minimum.reduce([expression[g][receiver] for g in mechanism.receptor_subunits.split(';')])
            score = left*right
            nonzero = np.flatnonzero(score>0)
            ratio = score[nonzero]/np.float32(mechanism.threshold)
            rows.append(nonzero.astype(np.int32))
            cols.append(np.full(len(nonzero), k, np.int32))
            values.append(ratio/(1+ratio))
        matrix = sparse.coo_matrix((np.concatenate(values), (np.concatenate(rows),np.concatenate(cols))),
                                    shape=(len(sender),len(catalog)), dtype=np.float32).tocsr()
        sparse.save_npz(sample_dir/f'lr_{index:04d}.npz', matrix)
        shards.append(dict(index=index, edge_start=edge_start, edge_stop=edge_start+len(sender)))
        edge_start += len(sender)
        chunk_first = row_end
        print(sample, 'target shard', index, 'edges', edge_start, flush=True)
    meta = dict(sample=sample, split=split, mpp=mpp, graph_nodes=len(nodes), directed_edges=edge_start,
                measured_pair_indices=measured, shards=shards, chunk_edges=chunk_edges,
                radius_um=radius, max_neighbors=neighbors,
                catalog_sha256=hashlib.sha256(Path('data/lr_catalog.csv').read_bytes()).hexdigest())
    (sample_dir/'complete.json').write_text(json.dumps(meta, indent=2)+'\n')
    sample_dir.rename(destination)
    return len(nodes), edge_start


def select_cohort(splits, samples=None):
    """Select slides without changing the frozen cohort or split assignments."""
    import yaml
    from morphic.model import verify_sha256
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/cohort.csv'), config['cohort_sha256'])
    cohort = pd.read_csv('data/cohort.csv')
    cohort = cohort[cohort.split.isin(splits)]
    if samples:
        if not set(samples) <= set(cohort['sample']):
            raise ValueError('Requested samples are outside the selected splits')
        cohort = cohort[cohort['sample'].isin(samples)]
    return cohort.sort_values('sample')
