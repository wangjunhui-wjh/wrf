import argparse
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import xarray as xr


def reproject_to_grid(src_path, ref_path, out_path, resampling=Resampling.average, band=1):
    with rasterio.open(ref_path) as ref:
        dst_crs = ref.crs
        dst_transform = ref.transform
        dst_height = ref.height
        dst_width = ref.width

    with rasterio.open(src_path) as src:
        dst = np.zeros((dst_height, dst_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=resampling,
        )

    meta = {
        'driver': 'GTiff',
        'height': dst_height,
        'width': dst_width,
        'count': 1,
        'dtype': 'float32',
        'crs': dst_crs,
        'transform': dst_transform,
    }
    with rasterio.open(out_path, 'w', **meta) as dst_ds:
        dst_ds.write(dst, 1)


def calc_slope(dem_path, out_path):
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        dx = transform.a
        dy = -transform.e
        gy, gx = np.gradient(dem, dy, dx)
        slope = np.sqrt(gx ** 2 + gy ** 2)
        meta = src.meta.copy()
        meta.update(dtype='float32', count=1)
        with rasterio.open(out_path, 'w', **meta) as dst:
            dst.write(slope, 1)


def add_to_grid_static(grid_static, out_tif, var_name):
    ds = xr.open_zarr(grid_static)
    with rasterio.open(out_tif) as src:
        arr = src.read(1).astype(np.float32)
    y_dim, x_dim = list(ds.sizes.keys())[-2:]
    ds[var_name] = (y_dim, x_dim), arr
    ds.to_zarr(grid_static, mode='a')


def main():
    parser = argparse.ArgumentParser(description='Process S2/WorldCover/DEM to WRF grid.')
    parser.add_argument('--grid-static', default='processed_data/grid_static.zarr')
    parser.add_argument('--ref-tif', required=True, help='Reference GeoTIFF (WRF grid)')
    parser.add_argument('--ndvi', default='')
    parser.add_argument('--ndbi', default='')
    parser.add_argument('--ndmi', default='')
    parser.add_argument('--ndvi-band', type=int, default=1)
    parser.add_argument('--ndbi-band', type=int, default=1)
    parser.add_argument('--ndmi-band', type=int, default=1)
    parser.add_argument('--worldcover', default='')
    parser.add_argument('--dem', default='')
    parser.add_argument('--out-dir', default='processed_data/roughness')
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dem:
        slope_path = out_dir / 'slope.tif'
        calc_slope(args.dem, slope_path)
        reproject_to_grid(slope_path, args.ref_tif, out_dir / 'slope_wrfg.tif', Resampling.average)
        add_to_grid_static(args.grid_static, out_dir / 'slope_wrfg.tif', 'SLOPE')

    ndvi_band = args.ndvi_band
    ndbi_band = args.ndbi_band
    ndmi_band = args.ndmi_band
    if args.ndvi and args.ndbi and args.ndmi:
        if args.ndvi == args.ndbi == args.ndmi:
            try:
                with rasterio.open(args.ndvi) as src:
                    if src.count >= 3 and src.descriptions:
                        desc = [d.upper() if d else "" for d in src.descriptions]
                        if "NDVI" in desc:
                            ndvi_band = desc.index("NDVI") + 1
                        if "NDBI" in desc:
                            ndbi_band = desc.index("NDBI") + 1
                        if "NDMI" in desc:
                            ndmi_band = desc.index("NDMI") + 1
            except Exception:
                pass

    if args.ndvi:
        reproject_to_grid(args.ndvi, args.ref_tif, out_dir / 'ndvi_wrfg.tif', Resampling.average, band=ndvi_band)
        add_to_grid_static(args.grid_static, out_dir / 'ndvi_wrfg.tif', 'NDVI')

    if args.ndbi:
        reproject_to_grid(args.ndbi, args.ref_tif, out_dir / 'ndbi_wrfg.tif', Resampling.average, band=ndbi_band)
        add_to_grid_static(args.grid_static, out_dir / 'ndbi_wrfg.tif', 'NDBI')

    if args.ndmi:
        reproject_to_grid(args.ndmi, args.ref_tif, out_dir / 'ndmi_wrfg.tif', Resampling.average, band=ndmi_band)
        add_to_grid_static(args.grid_static, out_dir / 'ndmi_wrfg.tif', 'NDMI')

    if args.worldcover:
        reproject_to_grid(args.worldcover, args.ref_tif, out_dir / 'worldcover_wrfg.tif', Resampling.nearest)
        add_to_grid_static(args.grid_static, out_dir / 'worldcover_wrfg.tif', 'LANDCOVER')


if __name__ == '__main__':
    main()
