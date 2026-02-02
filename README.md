# WRF 数据处理流水线

本项目旨在处理 WRF 原始输出文件（d04 区域），生成高质量的风场数据集。主要功能包括：

1.  **高分辨率 (HR) 风场**：使用 `wrf-python` 提取并将风场旋转至地球坐标系（Earth Coordinates）。
2.  **低分辨率 (LR) 风场**：通过物理一致的平均池化（Average Pooling）下采样生成，用于超分训练。
3.  **Zarr 存储**：按月打包为 Zarr 格式，支持高效的分块读取和并行训练。
4.  **清单与质量报告**：自动生成数据清单，检查时间连续性和缺失值。

## 前置要求

*   **Conda**：必须使用 Conda 环境，因为核心库 `wrf-python` 依赖 Fortran 编译环境。

## 环境安装 (Windows/Linux)

由于 `wrf-python` 的依赖性，强烈建议使用 Conda 而非 pip。

### 1. 配置镜像源（推荐国内用户）
如果下载速度慢，建议配置清华大学镜像源：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

### 2. 创建并激活环境
您可以使用提供的 `environment.yml` 一键创建环境：

```bash
conda env create -f environment.yml
conda activate wrf_env
```

或者手动创建：
```bash
conda create -n wrf_env python=3.10 -y
conda activate wrf_env
conda install -c conda-forge wrf-python xarray dask netcdf4 zarr pandas scipy matplotlib pyyaml tqdm -y
```

## 配置说明

请根据您的实际路径修改 `src/config.py` 文件：

*   `RAW_DATA_ROOT`: 移动硬盘上的原始数据根目录（例如 `E:/`）。
*   `TEMP_DIR`: 本地 SSD 上的临时处理目录（例如 `D:/wrf_tmp`），用于加速 IO。
*   `OUTPUT_DIR`: 处理后数据的保存目录。

## 运行流水线

在终端中执行以下命令启动处理：

```bash
python run_pipeline.py
```

## 处理流程详解

1.  **生成清单 (Manifest)**：
    *   扫描原始数据目录。
    *   生成文件列表 `raw_manifest/manifest_YYYY.csv`。
    *   自动检查是否存在缺失的时次（如某天少于48个文件）。

2.  **数据处理 (Processing)**：
    *   **IO 优化**：将当天的文件复制到本地 SSD 临时目录处理，处理完即删，减少移动硬盘负载。
    *   **变量提取**：优先使用 `wrf.getvar` 提取 `U10`, `V10` 等变量，自动处理风场旋转（Grid -> Earth）和去交错。
    *   **生成 LR**：对 HR 数据进行 4x（可配置）下采样生成 LR 数据。
    *   **格式转换**：将单月数据合并并保存为 Zarr 格式。

## 常见问题

*   **找不到 `wrf-python`**：请确保您已通过 `conda activate wrf_env` 激活了环境。
*   **安装时网络错误**：请尝试配置上述的国内镜像源。
