# WRF 风场超分辨率：数据处理、训练与评估

## 功能概览
- 从 wrfout 提取 U10/V10/T2/PSFC/PBLH 等变量，生成 HR/LR 月度 Zarr
- 站点/雷达观测处理与评估
- 超分模型训练、推理、网格/站点评估与出图

## 目录约定（核心输出）
- `processed_data/hr`：WRF 高分辨率（月度 Zarr）
- `processed_data/lr`：WRF 低分辨率（月度 Zarr，4×下采样）
- `processed_data/pred`：模型推理输出
- `processed_data/eval`：站点/雷达评估结果
- `processed_data/summary`：汇总表与论文图表

## 环境安装（当前使用 venv）
```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

训练需要 PyTorch，请根据显卡与 CUDA 版本安装对应的包（按官方选择器来）。


## 配置修改
在 `src/config.py` 里确认以下路径：
- `RAW_DATA_ROOT`：wrfout 原始数据根目录
- `TEMP_DIR`：本地临时目录（可选）
- `OUTPUT_DIR`：处理后数据输出目录（默认 `processed_data`）
- `DOMAIN`：目标域（默认 `d04`）

## WRF 数据处理
一键处理（生成 HR/LR 月度 Zarr）：
```bash
python run_pipeline.py
```

计算归一化统计（训练/推理用）：
```bash
python run_compute_stats.py --hr-dir processed_data/hr --lr-dir processed_data/lr --out processed_data/stats.json
```

## 站点/雷达数据处理（可选）
```bash
python run_obs_processing.py --station-dir data/station --radar-dir data/radar --out-dir processed_data/obs
```

## 模型训练
默认输入变量：`U10,V10,T2,PSFC,PBLH`，目标变量：`U10,V10`。默认上采样倍率 4×。
```bash
python run_train_sr.py --model unet --stats processed_data/stats.json --out-dir processed_data/checkpoints
```

需要手动指定测试月可用：
```bash
python run_train_sr.py --model unet --stats processed_data/stats.json --test-months 2025-10 --exclude-months 2025-09
```

## 推理
```bash
python run_infer_sr.py --model unet --weights processed_data/checkpoints/unet_x4/best.pt --months 2025-10 --stats processed_data/stats.json --out processed_data/pred/pred_unet_2025-10.zarr
```

## 评估
**WRF-HR 网格对比（内部评估）**
```bash
python run_grid_metrics.py --month 2025-10 --pred-dir processed_data/pred --hr-dir processed_data/hr --out processed_data/summary/metrics_grid_overall_2025-10.csv
```

**站点/雷达评估（外部验证）**
```bash
python run_evaluation.py --pred processed_data/pred/pred_unet_2025-10.zarr
```

## 常用汇总/出图脚本
- `run_eval_all.py`：批量评估多个预测结果
- `run_case_maps.py`：个例风场/误差图
- `run_physics_consistency.py`：散度/频谱一致性统计
- `run_make_paper_figs.py`：论文图表汇总

## 说明
- HR/LR 使用 4×缩放（`DOWNSAMPLE_FACTOR=4`），名义分辨率约为 3 km → 1 km。
- 若在 PowerShell 使用多行命令，请用反引号 `（不是 ^）进行换行。
