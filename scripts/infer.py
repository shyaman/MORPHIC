"""Score a new H&E slide with detected, supplied, or annotated nuclei."""
from __future__ import annotations

import argparse
import json
import hashlib
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely
import tifffile
import torch
import yaml
from PIL import Image
from scipy.spatial import cKDTree
from skimage.measure import find_contours, regionprops

from morphic.data import graph
from morphic.images import Slide
from morphic.model import embed, load_encoder, load_head, verify_sha256


def detect(slide, nuclei, device):
    """Tile at 0.5 µm/pixel, retaining only the core of each tile."""
    if nuclei is None:
        from instanseg import InstanSeg
        os.environ.setdefault('INSTANSEG_BIOIMAGEIO_PATH',str(Path('data/raw/models/instanseg').resolve()))
        segmenter = InstanSeg('brightfield_nuclei', device=str(device), verbosity=0)
    else:
        segmenter = None
    mask = None
    array = None
    if nuclei is not None:
        mask = tifffile.TiffFile(nuclei)
        array = __import__('zarr').open(mask.pages[0].aszarr(),mode='r')
        if array.shape[:2] != (slide.height,slide.width):
            raise ValueError('Nucleus label mask must align with the full-resolution H&E image')
    records = []
    tile_px, core_px = 1024, 896
    scale = .5/slide.mpp
    width = slide.width/scale
    height = slide.height/scale
    try:
        if min(width,height)<tile_px:
            raise ValueError('Image must span at least 512 µm in both dimensions')
        ys = np.unique(np.r_[np.arange(tile_px/2,height-tile_px/2+1,core_px),height-tile_px/2])
        xs = np.unique(np.r_[np.arange(tile_px/2,width-tile_px/2+1,core_px),width-tile_px/2])
        for yi,cy in enumerate(ys):
            y_low = 64 if yi==0 else (ys[yi-1]+cy)/2
            y_high = height-64 if yi==len(ys)-1 else (cy+ys[yi+1])/2
            for xi,cx in enumerate(xs):
                x_low = 64 if xi==0 else (xs[xi-1]+cx)/2
                x_high = width-64 if xi==len(xs)-1 else (cx+xs[xi+1])/2
                center = (cx*scale,cy*scale)
                rgb = slide.crop(center,size=tile_px)
                if array is None:
                    labels = segmenter.eval_medium_image(rgb,pixel_size=.5,
                                return_image_tensor=False).squeeze().numpy().astype(np.int32)
                else:
                    x0,y0 = int(round(center[0]-tile_px*scale/2)),int(round(center[1]-tile_px*scale/2))
                    n = int(round(tile_px*scale))
                    labels = np.asarray(Image.fromarray(np.asarray(array[y0:y0+n,x0:x0+n])).resize(
                        (tile_px,tile_px),Image.Resampling.NEAREST)).astype(np.int32)
                for region in regionprops(labels):
                    y,x = region.centroid
                    gx, gy = cx-tile_px/2+x, cy-tile_px/2+y
                    if not (x_low<=gx<x_high and y_low<=gy<y_high) or region.area<12:
                        continue
                    contours = find_contours(region.image.astype(np.uint8),.5)
                    if not contours:
                        continue
                    contour = max(contours,key=len)
                    points = [((cx-tile_px/2+x0p+region.bbox[1]) * scale,
                               (cy-tile_px/2+y0p+region.bbox[0]) * scale)
                              for y0p,x0p in contour]
                    polygon = shapely.Polygon(points)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if polygon.is_empty or polygon.geom_type!='Polygon':
                        continue
                    records.append((len(records),center[0]+(x-tile_px/2)*scale,
                                    center[1]+(y-tile_px/2)*scale,shapely.to_wkb(polygon)))
    finally:
        if mask is not None:
            mask.close()
    if not records:
        raise ValueError('No usable nuclei found')
    return pd.DataFrame(records,columns=['cell_id','x_px','y_px','nucleus_wkb'])


def attach_annotations(cells, path, mpp):
    annotations = pd.read_csv(path)
    required = {'cell_id','x_px','y_px'}
    if not required <= set(annotations) or annotations.cell_id.duplicated().any():
        raise ValueError('Annotations need unique cell_id, x_px, y_px columns')
    if len(cells)==0:
        raise ValueError('No detected nuclei for annotation matching')
    distance,index = cKDTree(cells[['x_px','y_px']].to_numpy()).query(
        annotations[['x_px','y_px']].to_numpy(),distance_upper_bound=10/mpp)
    good = np.isfinite(distance) & (index<len(cells))
    if len(np.unique(index[good]))!=good.sum():
        raise ValueError('Multiple annotations match one nucleus')
    result = cells.iloc[index[good]].copy().reset_index(drop=True)
    selected = annotations.loc[good].reset_index(drop=True)
    result['cell_id'] = selected.cell_id.astype(str)
    for column in selected.columns.difference(['cell_id','x_px','y_px']):
        result[column] = selected[column]
    print(f'Matched {len(result)}/{len(annotations)} annotations to nuclei',flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--mpp',type=float,required=True,help='Microns per full-resolution image pixel')
    p.add_argument('--nuclei',type=Path,help='Optional aligned integer nucleus label TIFF')
    p.add_argument('--annotations',type=Path,help='Optional CSV: cell_id,x_px,y_px[,cell_type]')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--encoder',type=Path,default=Path('data/raw/models/uni2-h/pytorch_model.bin'))
    p.add_argument('--checkpoint',type=Path,default=Path('models/morphic_head.pt'))
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--sample-percent',type=float,default=100.,
                   help='Percent of crop-ready cells to use as graph anchors (default: 100)')
    p.add_argument('--seed',type=int,default=17,help='Seed for anchor sampling')
    args = p.parse_args()
    if not 0 < args.sample_percent <= 100:
        p.error('--sample-percent must be greater than 0 and at most 100')
    if args.out.exists():
        raise FileExistsError(f'Refusing to overwrite {args.out}')
    args.out.mkdir(parents=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    catalog = pd.read_csv('data/lr_catalog.csv')
    config = yaml.safe_load(Path('configs/model.yaml').read_text())
    verify_sha256(Path('data/lr_catalog.csv'),config['catalog_sha256'])
    verify_sha256(args.encoder,config['encoder_sha256'])
    slide = Slide(args.image,args.mpp)
    try:
        cells = detect(slide,args.nuclei,device)
        if args.annotations:
            cells = attach_annotations(cells,args.annotations,args.mpp)
        # The 112-µm crop must remain within image bounds.
        margin = 56/args.mpp
        cells = cells[cells.x_px.between(margin,slide.width-margin) &
                      cells.y_px.between(margin,slide.height-margin)].reset_index(drop=True)
        if len(cells)<2:
            raise ValueError('Need at least two crop-ready cells')
        eligible_cells = len(cells)
        anchor_count = max(1,int(np.ceil(eligible_cells*args.sample_percent/100)))
        anchors = np.sort(np.random.default_rng(args.seed).choice(
            eligible_cells,size=anchor_count,replace=False))
        xy = cells[['x_px','y_px']].to_numpy()*args.mpp
        used,pairs,distance = graph(xy,anchors,float(config['graph_radius_um']),
                                    int(config['graph_max_neighbors']))
        cells = cells.iloc[used].reset_index(drop=True)
        edges = pd.DataFrame({'sender':pairs[:,0], 'receiver':pairs[:,1], 'distance_um':distance})
        cells.drop(columns='nucleus_wkb').to_parquet(args.out/'cells.parquet',index=False)
        edges.to_parquet(args.out/'edges.parquet',index=False)
        encoder = load_encoder(args.encoder,device)
        features = np.lib.format.open_memmap(args.out/'embeddings.npy',mode='w+',dtype=np.float32,
                                             shape=(len(cells),1536))
        for start in range(0,len(cells),args.batch):
            part = cells.iloc[start:start+args.batch]
            crops = np.stack([slide.crop((r.x_px,r.y_px)) for r in part.itertuples()])
            masks = np.stack([slide.nucleus_mask((r.x_px,r.y_px),r.nucleus_wkb) for r in part.itertuples()])
            features[start:start+len(part)] = embed(encoder,torch.from_numpy(crops).permute(0,3,1,2),
                                                    torch.from_numpy(masks)).cpu().numpy()
        features.flush()
        del encoder
        if args.checkpoint.resolve() == Path('models/morphic_head.pt').resolve():
            verify_sha256(args.checkpoint,json.loads(Path('models/manifest.json').read_text())['files']['models/morphic_head.pt'])
        head = load_head(args.checkpoint,catalog,device)
        schema = pa.schema([('sender_id',pa.string()),('receiver_id',pa.string()),
                            ('mechanism',pa.string()),('score',pa.float32())])
        names = catalog.interaction_name.to_numpy()
        ids = cells.cell_id.astype(str).to_numpy()
        with pq.ParquetWriter(args.out/'scores.parquet',schema) as writer, torch.inference_mode():
            for start in range(0,len(edges),256):
                e = edges.iloc[start:start+256]
                sender = torch.from_numpy(np.asarray(features[e.sender.to_numpy()])).to(device)
                receiver = torch.from_numpy(np.asarray(features[e.receiver.to_numpy()])).to(device)
                d = torch.from_numpy(e.distance_um.to_numpy(np.float32)).to(device)
                context = head.encode_context(sender,receiver,d)
                for k0 in range(0,len(catalog),64):
                    ks = np.arange(k0,min(k0+64,len(catalog)),dtype=np.int64)
                    position = torch.arange(len(e),device=device).repeat_interleave(len(ks))
                    mechanism = torch.from_numpy(np.tile(ks,len(e))).to(device)
                    scores = head.score_context(context,mechanism,position).cpu().numpy().astype(np.float32)
                    table = pa.Table.from_pydict({'sender_id':np.repeat(ids[e.sender.to_numpy()],len(ks)),
                        'receiver_id':np.repeat(ids[e.receiver.to_numpy()],len(ks)),
                        'mechanism':np.tile(names[ks],len(e)), 'score':scores},schema=schema)
                    writer.write_table(table)
        (args.out/'run.json').write_text(json.dumps({'image':str(args.image),'mpp':args.mpp,
            'nuclei':str(args.nuclei) if args.nuclei else 'InstanSeg brightfield_nuclei',
            'annotations':str(args.annotations) if args.annotations else None,
            'checkpoint':str(args.checkpoint),
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'cells':len(cells),'edges':len(edges),
            'eligible_cells':eligible_cells,'sample_percent':args.sample_percent,
            'sampled_anchors':anchor_count,'seed':args.seed,
            'mechanisms':len(catalog),'scores_calibrated':False},indent=2)+'\n')
    finally:
        slide.close()


if __name__ == '__main__':
    main()
