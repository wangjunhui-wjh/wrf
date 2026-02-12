# Sentinel-2 粗糙度代理数据流程（简化版）

## 1) 数据获取（GEE 脚本）
- scripts/gee_s2_worldcover.js
- 修改 AOI 与日期后，在 GEE 中导出 NDVI/NDBI/NDMI 与 WorldCover

## 2) 处理与对齐（本地）
- scripts/process_s2_roughness.py
- 输入：NDVI/NDMI/NDBI/WorldCover GeoTIFF + DEM
- 输出：对齐到 WRF 网格的静态变量，并写入 grid_static.zarr

示例：
python scripts/process_s2_roughness.py ^
  --grid-static processed_data/grid_static.zarr ^
  --ref-tif <wrf_grid_ref.tif> ^
  --ndvi NDVI.tif --ndbi NDBI.tif --ndmi NDMI.tif ^
  --worldcover WorldCover.tif ^
  --dem DEM.tif

## 3) 加入模型输入
- run_compute_stats.py 支持 --static-vars
- run_train_sr.py / run_infer_sr.py 支持 --static-vars

示例：
python run_compute_stats.py --static-vars HGT,NDVI,NDBI,NDMI,SLOPE,LANDCOVER --grid-static processed_data/grid_static.zarr
python run_train_sr.py --model unet --static-vars HGT,NDVI,NDBI,NDMI,SLOPE,LANDCOVER --grid-static processed_data/grid_static.zarr --residual --lambda-scale 0.1 --stats processed_data/stats.json

注意：
- 参考网格 ref-tif 需要与你的 WRF 网格一致（可用 QGIS 从 grid_static 导出）
- WorldCover 为分类数据，建议后续转 one-hot 或 embedding
