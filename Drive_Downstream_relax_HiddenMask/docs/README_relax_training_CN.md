# D001c AnyVisiblePatch：四模型正式训练最终版

## 1. 项目根目录

```text
/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/
Downstream_Task_Bathy_relax_HiddenMask
```

D001c tile仍是预处理输入，保留在：

```text
/tank/data/SFS/xinyis/data/bathymetry/Processed_Results/
Tiles_for_MAE_v2_D001c_AnyVisiblePatch/Tiles_1m
```

所有split、训练日志、checkpoint与后续结果写入：

```text
Downstream_Task_Bathy_relax_HiddenMask/results
```

## 2. 本最终版完成的统一

四条路线全部调用同一个训练入口和同一个engine：

```text
mae_Retrain_relax/main_pretrain_dem_unified_relax.py
mae_Retrain_relax/engine_pretrain_unified_relax.py
```

因此NormOnly和MeterThenNorm使用的是完全相同的exact core-pixel
`normalized_mse`实现，不再只是概念上相同、代码入口不同。

区别只保留在初始化和checkpoint选择规则：

```text
NormOnly:
  init         = upstream checkpoint
  objective    = normalized_mse
  best/ES      = val_loss

MeterOnly:
  init         = upstream checkpoint
  objective    = meter_mae
  best/ES      = val_mae_m_mask

NormThenMeter:
  init         = NormOnly checkpoint-best
  objective    = meter_mae
  best/ES      = val_mae_m_mask
  epoch -1     = untouched NormOnly

MeterThenNorm:
  init         = MeterOnly checkpoint-best
  objective    = normalized_mse
  best/ES      = val_mae_m_mask
  epoch -1     = untouched MeterOnly
```

在统一engine中，normalized模式下：

```text
val_loss == exact pixel-weighted normalized_mse_mask
```

监督范围始终为：

```text
Loss_Mask_Pixel
AND prediction/core patch mask
AND valid patch
```

## 3. Encoder设置

正式四模型基线默认：

```text
TRAINABLE_LAST_N_ENCODER_BLOCKS=0
```

即冻结整个encoder，只训练decoder。

四模型完成比较后，同一套代码可设置：

```text
TRAINABLE_LAST_N_ENCODER_BLOCKS=1
```

用于最后一个encoder block的低学习率微调。主实验不要提前开启。

## 4. 安装

解压最终包后：

```bash
cd Drive_Downstream_relax_HiddenMask_FINAL_4Models_20260729
bash setup_relax_project.sh
```

安装程序会：

1. 复制服务器当前`MAE-Topography/mae_Retrain`到隔离目录；
2. 覆盖加入统一RELAX Python入口和engine；
3. 安装D040–D059脚本；
4. 创建四套独立results目录。

## 5. 训练前检查

```bash
RELAX_ROOT=/tank/data/SFS/xinyis/data/bathymetry/MAE-Topography/Downstream_Task_Bathy_relax_HiddenMask

cd "$RELAX_ROOT/script"
bash D059_relax_preflight_four_models.sh
```

必须看到：

```text
Preflight passed.
```

## 6. 正式提交

```bash
bash D058_relax_submit_all_four.sh
```

任务关系：

```text
NormOnly  ── afterok ──> NormThenMeter
MeterOnly ── afterok ──> MeterThenNorm
```

NormOnly与MeterOnly并行。CA、CO、Santiam每条河四个模型，共12个训练任务。

## 7. 默认训练配置

```text
NormOnly:
  epochs=400
  lr=1e-4
  patience=60

MeterOnly:
  epochs=400
  lr=1e-4
  patience=60

NormThenMeter:
  epochs=120
  lr=1e-5
  patience=30

MeterThenNorm:
  epochs=120
  lr=1e-5
  patience=30
```

四条路线均使用：

```text
D001c AnyVisiblePatch
tile_norm_visible_only
tile_norm_std_scale=1.5
lcc_mask_mode=exact
loss_region_mode=core
core_patch_radius=3
encoder frozen
```

## 8. 结果目录

```text
results/NormOnly
results/MeterOnly
results/NormThenMeter
results/MeterThenNorm
```

原D052仅是三模型旧提交器。正式四模型训练统一使用D058。
