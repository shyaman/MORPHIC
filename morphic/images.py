"""Physical-scale H&E crops and nucleus masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
import tifffile
import zarr
from PIL import Image, ImageDraw


class Slide:
    def __init__(self, path: Path, mpp: float):
        if mpp <= 0:
            raise ValueError('MPP must be positive')
        self.mpp = mpp
        self.tif = tifffile.TiffFile(path)
        self.stores = [page.aszarr() for page in self.tif.pages]
        self.arrays = [zarr.open(store, mode='r') for store in self.stores]
        self.height, self.width = self.arrays[0].shape[:2]
        if self.arrays[0].shape[-1] != 3:
            raise ValueError('Expected RGB H&E TIFF')

    def crop(self, center, size=224, target_mpp=.5):
        scale = target_mpp / self.mpp
        levels = np.array([self.width / array.shape[1] for array in self.arrays])
        level = max(np.flatnonzero(levels <= scale * 1.01), default=0)
        array = self.arrays[level]
        sx, sy = self.width / array.shape[1], self.height / array.shape[0]
        w, h = int(round(size * scale / sx)), int(round(size * scale / sy))
        x0, y0 = int(round(center[0] / sx - w / 2)), int(round(center[1] / sy - h / 2))
        if x0 < 0 or y0 < 0 or x0 + w > array.shape[1] or y0 + h > array.shape[0]:
            raise ValueError('Cell crop extends outside H&E image')
        return np.asarray(Image.fromarray(np.asarray(array[y0:y0+h, x0:x0+w])).resize(
            (size, size), Image.Resampling.BILINEAR))

    def nucleus_mask(self, center, polygon_wkb, size=224, target_mpp=.5):
        polygon = shapely.from_wkb(polygon_wkb)
        if polygon.is_empty or polygon.geom_type != 'Polygon':
            raise ValueError('Expected a nucleus polygon')
        mask = Image.new('L', (size, size), 0)
        draw = ImageDraw.Draw(mask)
        scale = self.mpp / target_mpp
        def coords(ring):
            return [((x-center[0])*scale+size/2, (y-center[1])*scale+size/2)
                    for x, y in ring.coords]
        draw.polygon(coords(polygon.exterior), fill=1)
        for ring in polygon.interiors:
            draw.polygon(coords(ring), fill=0)
        result = np.asarray(mask, dtype=np.uint8).copy()
        if not result.any():
            result[size//2, size//2] = 1
        return result

    def close(self):
        for store in self.stores:
            store.close()
        self.tif.close()
