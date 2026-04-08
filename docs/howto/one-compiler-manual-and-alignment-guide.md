# ONE 编译器使用手册 + 量化对齐排查指南

> 目标读者：已经能跑通模型转换，但在量化后出现“数值严重偏离”的开发者。

---

## 1. ONE 编译器常用命令总览

`compiler/one-cmds` 目录下常用命令可按功能分成 4 类。

### 1.1 环境与总入口

- `one-prepare-venv`：准备 one-cmds Python 虚拟环境
- `onecc`：统一编排入口（推荐）
- `one-init`：初始化配置/工程辅助

### 1.2 导入与转换

- `one-import`：按 driver 分发导入
- `one-import-tf`
- `one-import-tflite`
- `one-import-onnx`
- `one-import-bcq`
- `one-import-pytorch`

### 1.3 编译流程命令

- `one-optimize`：图优化（可选）
- `one-quantize`：量化（可选）
- `one-pack`：打包为 `nnpackage`
- `one-partition`：分区到后端
- `one-codegen`：后端代码生成（视后端能力）

### 1.4 运行与分析

- `one-infer`：调用 backend driver 推理
- `one-profile`：性能分析
- `one-create-quant-dataset`：辅助构建量化数据集

---

## 2. 推荐工作流（从稳到快）

对于 onert 运行场景，建议从最稳流程开始：

1. `one-import`
2. `one-optimize`（可先关闭）
3. `one-quantize`（可先关闭）
4. `one-pack`

建议先跑“无优化、无量化”基线，确认端到端正确后，再逐步打开优化和量化。

---

## 3. onecc 配置方式（推荐）

`onecc.template.cfg` 支持每个步骤独立开关。核心开关在 `[onecc]` 段：

- `one-import-*`
- `one-optimize`
- `one-quantize`
- `one-pack`
- `one-codegen`
- `one-infer`

这对排查非常重要：你可以做 A/B 流程实验，而不是一次改很多参数。

---

## 4. one-quantize 的关键参数（必须掌握）

### 4.1 校准数据

- `--input_data`：量化校准输入数据
- `--input_data_format`：`h5/hdf5`, `list/filelist`, `dir/directory`

⚠️ 如果不提供 `--input_data`，会使用随机输入做 PTQ，常导致严重精度漂移。

### 4.2 量化类型与 I/O 类型

- `--quantized_dtype`：输出量化类型（如 `uint8`, `int16`）
- `--granularity`：`layer` / `channel`
- `--input_type` / `--output_type`：模型外部输入输出类型

当 `input_type/output_type` 与 `quantized_dtype` 不同，模型会插入量化相关转换节点。

### 4.3 量化校准算法参数

- `--mode`：`percentile` / `moving_average`
- `--min_percentile` / `--max_percentile`
- `--moving_avg_batch` / `--moving_avg_const`

---

## 5. 内置“对齐评估”能力（是真的）

`one-quantize` 内置了对齐评估开关，可以直接比较 fp32 模型与量化模型结果：

- `--evaluate_result`
- `--test_data`
- `--print_mae`
- `--print_mape`
- `--print_mpeir`
- `--print_top1_match`
- `--print_top5_match`
- `--print_mse`

这意味着你可以在工具链内直接得到误差指标，而不必先自建比较脚本。

---

## 6. 数值严重偏离排查 Guide（实操版）

### Step A：先做“阶段切分”定位

在同一输入集上跑 4 条链路：

1. import only（基线）
2. import + optimize
3. import + quantize
4. import + optimize + quantize

如果偏离在 #3 首次出现，优先看量化配置；如果在 #2 出现，优先看优化 pass。

### Step B：量化专项排查（优先）

1. **强制使用真实校准集**（不要随机输入）
2. 先让 `input_type/output_type=float32`，减少 I/O 转换带来的额外误差
3. 打开 `--evaluate_result` 并输出 `MAE/MSE`
4. 调整 `--mode`（percentile ↔ moving_average）
5. 再调整 percentile/moving average 参数

### Step C：建立误差阈值（工程化）

建议给每个模型固定阈值（示例）：

- 回归类：MAE / MSE 阈值
- 分类类：Top1/Top5 match 阈值

并在 CI 中自动判断 pass/fail，避免“人工感觉对不对”。

### Step D：仍严重偏离时，按三层模型定位

把问题拆到以下 3 层：

1. compiler/IR（导入/导出/图变换）
2. runtime loader/IR（能否正确加载并映射）
3. backend kernel（CPU/ACL/NPU 的具体实现差异）

先用 CPU 建立正确性基线，再看 ACL/NPU 差异。

---

## 7. 推荐学习路径（你现在最该学什么）

若目标是“能修 unsupported op + 能修量化对齐”，建议顺序：

1. `onecc` 工作流与配置切分能力
2. `one-import` + 各 driver（onnx/tflite/tf）边界
3. `one-quantize` 全参数语义与误差指标
4. `luci` importer/exporter + shape/type inference
5. `runtime` 的 loader → IR → backend 链路

这样你就能快速回答：

- 是 compiler 问题？
- 是量化参数问题？
- 还是 runtime/backend 实现差异？

---

## 8. 最小执行清单（给当前问题）

- [ ] 准备真实代表性校准集（不要随机）
- [ ] 跑 4 条链路定位首次漂移阶段
- [ ] 打开 `--evaluate_result --print_mae --print_mse`
- [ ] 固定并记录阈值
- [ ] CPU 先对齐，再看 ACL/NPU

做到这 5 步，通常就能把“数值严重偏离”收敛到可定位的问题。
