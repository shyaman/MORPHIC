"""Download the fixed HEST training cohort and gated UNI2-h weights."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from morphic.data import select_cohort
from huggingface_hub import hf_hub_download, snapshot_download

HEST_REVISION = '7e8d5a0b0aace41d8c8ec0f6ecea80e4ad2a61ec'


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=Path('data/raw'))
    p.add_argument('--skip-hest', action='store_true')
    p.add_argument('--skip-encoder', action='store_true')
    p.add_argument('--skip-segmentation', action='store_true')
    p.add_argument('--split', nargs='+', choices=['train','validation','test'], default=['train','validation'])
    p.add_argument('--samples', nargs='+')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = args.out/'download_manifest.json'
    receipt = json.loads(manifest.read_text()) if manifest.exists() else {'hest_revision': HEST_REVISION, 'files': {}}
    if receipt['hest_revision'] != HEST_REVISION:
        raise ValueError('Existing download manifest uses a different HEST revision')
    if not args.skip_hest:
        samples = select_cohort(args.split, args.samples)['sample'].tolist()
        patterns = [f'{folder}/{sample}{suffix}' for sample in samples for folder, suffix in
                    [('wsis','.tif'), ('metadata','.json'),
                     ('transcripts','_transcripts.parquet'),
                     ('xenium_seg','_xenium_cell_seg.parquet'),
                     ('xenium_seg','_xenium_nucleus_seg.parquet')]]
        snapshot_download('MahmoodLab/hest', repo_type='dataset', revision=HEST_REVISION,
                          local_dir=args.out, allow_patterns=patterns, max_workers=4)
        for name in patterns:
            path = args.out/name
            if not path.is_file():
                raise FileNotFoundError(path)
            receipt['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    if not args.skip_encoder:
        path = Path(hf_hub_download('MahmoodLab/UNI2-h', 'pytorch_model.bin',
                                    local_dir=args.out/'models'/'uni2-h'))
        if sha256(path) != '6e077eda234bebc595868d918d3458d9dd32a050199b0ff04443b2f46a0a3b1e':
            raise ValueError('UNI2-h weights differ from the selected model')
        receipt['encoder'] = {'path': str(path), 'bytes': path.stat().st_size,
                              'sha256': sha256(path)}
    if not args.skip_segmentation:
        os.environ['INSTANSEG_BIOIMAGEIO_PATH'] = str((args.out/'models'/'instanseg').resolve())
        from instanseg.utils.utils import download_model
        download_model('brightfield_nuclei', version='0.1.1')
        path = args.out/'models'/'instanseg'/'brightfield_nuclei'/'0.1.1'/'instanseg.pt'
        receipt['segmentation'] = {'path':str(path),'bytes':path.stat().st_size,
                                   'sha256':sha256(path)}
    (args.out/'download_manifest.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(args.out/'download_manifest.json')


if __name__ == '__main__':
    main()
