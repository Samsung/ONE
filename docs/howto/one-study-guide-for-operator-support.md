# ONE 项目学习资料：以“读懂并补算子支持”为目标

这份资料不是官方文档的拷贝，而是按“先能跑、再能定位、最后能改算子”的顺序整理的学习路线。

目标读者：

- 想把模型编译成 `circle` / `nnpackage`
- 遇到 unsupported op，想判断问题是在 compiler 还是 runtime
- 最终需要自己补一个算子

---

## 1. 先用一句话理解这个项目

`ONE` 是一套 on-device AI 工具链，核心由三部分组成：

- `compiler`：把 TensorFlow / TFLite / ONNX 等模型变成 `circle`，并做优化、量化、打包
- `runtime`：加载 `circle` 或 `nnpackage`，编译成可执行形式，并调用 backend 执行
- `onert-micro`：面向 MCU / 轻量环境的解释器和 kernel 集

如果你只关心“模型能不能跑”，最常见路径是：

1. 原始模型
2. `one-import-*`
3. `circle`
4. `one-optimize` / `one-quantize`
5. `one-pack`
6. `nnpackage`
7. `nnfw` / `onert`

---

## 2. 顶层目录地图

仓库根目录里最重要的目录和入口如下：

- `nncc`
  - compiler 侧构建入口
- `nnfw`
  - runtime 侧构建入口
- `Makefile.template`
  - 一键串起 compiler overlay + runtime 的总入口
- `compiler/`
  - 编译器、IR、优化 pass、one-* 命令
- `runtime/`
  - onert runtime、backend、API、测试工具
- `onert-micro/`
  - MCU/轻量解释器
- `tools/`
  - 配套工具
- `docs/`
  - 文档

对“补算子支持”最重要的子目录：

- `compiler/one-cmds`
- `compiler/luci`
- `runtime/onert/api/nnfw`
- `runtime/onert/core`
- `runtime/onert/backend`

---

## 3. 两条主线：编译链和运行链

### 3.1 编译链

用户看到的是 `one-*` 命令，代码入口主要在：

- `compiler/one-cmds/onecc`
- `compiler/one-cmds/one-import`
- `compiler/one-cmds/one-optimize`
- `compiler/one-cmds/one-quantize`
- `compiler/one-cmds/one-pack`
- `compiler/one-cmds/one-infer`

`onecc` 本质上是 Python 驱动器：

- 直接调单个子命令
- 或按配置文件 / workflow 顺序执行多个步骤

关键文件：

- `compiler/one-cmds/onecc`
- `compiler/one-cmds/onelib/CfgRunner.py`
- `compiler/one-cmds/onelib/WorkflowRunner.py`
- `compiler/one-cmds/onecc.template.cfg`

一个典型流转：

1. `one-import tflite`
2. `one-optimize`
3. `one-quantize`
4. `one-pack`

对应产物变化：

- `.tflite` / `.onnx` / `.pb`
- `.circle`
- 优化后 `.circle`
- 量化后 `.circle`
- `nnpackage/`

### 3.2 运行链

用户侧入口一般是 `nnfw` C API。

主入口文件：

- `runtime/onert/api/nnfw/include/nnfw.h`
- `runtime/onert/api/nnfw/src/APIImpl.cc`
- `runtime/onert/api/nnfw/src/Session.cc`

执行主路径：

1. `nnfw_create_session`
2. `nnfw_load_model_from_file`
3. `nnfw_prepare`
4. `nnfw_set_input` / `nnfw_set_output`
5. `nnfw_run`

在代码中：

- `APIImpl.cc` 只是 C API 封装
- 真实逻辑在 `Session.cc`
- `Session::prepare()` 里会调用 `Compiler`
- `Compiler` 生成 executor
- `Execution` 负责实际执行

关键链路：

- `runtime/onert/api/nnfw/src/Session.cc`
- `runtime/onert/core/src/compiler/Compiler.cc`
- `runtime/onert/core/src/exec/*`

---

## 4. 这个项目里“算子支持”到底分几层

这是最重要的概念。

一个算子“支持”通常要分成三层看：

1. **compiler/IR 层**
   - 能不能表示这个算子
   - 能不能导入这个算子
   - 能不能导出回 `circle`
2. **runtime loader/IR 层**
   - `onert` 能不能把 `circle` 里的这个 op 读成自己的运行时 IR
3. **backend kernel 层**
   - CPU / ACL / 其他 backend 有没有实际 kernel

因此，模型报“不支持算子”时，可能是：

- 导入失败：前端没接
- 能导入但运行不了：runtime 没接
- runtime 接了但某 backend 没 kernel：换 CPU 也许能跑

---

## 5. compiler 侧怎么组织

### 5.1 `luci` 是核心

`luci` 是 Circle / TFLite dialect IR 的核心。

相关目录：

- `compiler/luci/lang`
  - IR 节点定义
- `compiler/luci/import`
  - 从 circle 文件导入到 luci graph
- `compiler/luci/export`
  - 从 luci graph 导回 circle
- `compiler/luci/pass`
  - 图变换 / 优化 pass
- `compiler/luci/service`
  - shape/type inference、克隆、帮助逻辑
- `compiler/luci/partition`
  - 图分割

这些 README 都很短，真正要看代码。

### 5.2 一个新算子在 compiler 里通常要补哪些地方

可以把 `luci` 看成这几个点：

1. **IR 节点**
   - `compiler/luci/lang/include/luci/IR/Nodes/Circle*.h`
2. **导入**
   - `compiler/luci/import/include/luci/Import/Nodes/Circle*.h`
   - `compiler/luci/import/src/Nodes/Circle*.cpp`
   - `compiler/luci/import/src/GraphBuilderRegistry.cpp`
3. **导出**
   - `compiler/luci/export/src/CircleOps.lst`
   - `compiler/luci/export/src/CircleBuiltinTypesExtractor.h`
4. **类型推导**
   - `compiler/luci/service/src/CircleTypeInferenceRule.cpp`
5. **形状推导**
   - `compiler/luci/service/src/CircleShapeInferenceRule.cpp`
6. **克隆/分区/摘要**
   - `compiler/luci/service/src/Nodes/Circle*.cpp`
   - `compiler/luci/partition/src/Nodes/Circle*.cpp`
   - `compiler/luci/logex/src/CircleNodeSummaryBuilder.cpp`

其中最关键的入口文件有两个：

- `compiler/luci/import/src/GraphBuilderRegistry.cpp`
- `compiler/luci/export/src/CircleOps.lst`

这两个文件基本可以回答：

- importer 认不认识这个 op
- exporter 能不能把它写回 circle

### 5.3 编译器支持算子的一个判断方法

如果你想判断某个 op 在 compiler 侧是否接了，先查这几类文件：

```sh
rg "CircleGatherNd|CircleScatterNd|CircleSin" compiler/luci
```

优先看：

- `compiler/luci/import/src/GraphBuilderRegistry.cpp`
- `compiler/luci/export/src/CircleOps.lst`
- `compiler/luci/service/src/CircleTypeInferenceRule.cpp`
- `compiler/luci/service/src/CircleShapeInferenceRule.cpp`

如果这几处都齐，通常说明 compiler 这层已经比较完整。

---

## 6. runtime 侧怎么组织

### 6.1 `onert` 的四个核心模块

官方文档 `docs/runtime/core.md` 里给了比较准确的概念划分：

- `ir`
  - 运行时 IR
- `compiler`
  - 把运行时 IR 编译成可执行结构
- `exec`
  - executor / execution
- `backend`
  - kernel 和 tensor 管理接口

代码位置主要在：

- `runtime/onert/core/include/ir`
- `runtime/onert/core/src/compiler`
- `runtime/onert/core/src/exec`
- `runtime/onert/backend`

### 6.2 runtime 中算子支持的三步

#### 第一步：loader 能读

关键文件：

- `runtime/onert/core/src/loader/BaseLoader.h`
- `runtime/onert/core/src/loader/CircleLoader.cc`
- `runtime/onert/core/src/loader/TFLiteLoader.cc`

这里负责：

- 把 `BuiltinOperator_*` 映射到 `onert::ir::operation::*`
- 特判 Circle 扩展 builtin
- 处理少量 custom op 名字

如果这里没有 case，模型会在加载时报：

- `Unsupported operation: XXX`

#### 第二步：runtime IR 有定义

关键文件：

- `runtime/onert/core/include/ir/Operations.lst`
- `runtime/onert/core/include/ir/operation/*.h`
- `runtime/onert/core/include/ir/Operations.Include.h`

这些文件决定：

- `onert` 内部有哪些 operation family
- 每个 family 的参数长什么样

注意：`onert` 运行时 IR 不是严格“一 builtin 对一个类”。

它有很多 grouped op，例如：

- `BinaryArithmetic`
  - `ADD/SUB/MUL/DIV`
- `Comparison`
  - `Equal/NotEqual/Greater/...`
- `ElementwiseUnary`
  - `ABS/COS/SIN/LOG/...`
- `ElementwiseBinary`
  - `MIN/MAX/FLOOR_DIV/...`
- `Reduce`
  - `MEAN/MAX/MIN/PROD/SUM/...`
- `Pool2D`
  - `AVG/L2/MAX`

所以 runtime 支持判断，不能只看类名，要看类里的 enum。

#### 第三步：backend 有 kernel

关键目录：

- `runtime/onert/backend/cpu`
- `runtime/onert/backend/acl_cl`
- `runtime/onert/backend/acl_neon`
- `runtime/onert/core/src/backend/builtin`

其中：

- `cpu/Operation.lst`
- `acl_cl/Operation.lst`
- `acl_neon/Operation.lst`

可以先粗看这个 backend 声称支持哪些 operation family。

但最终还是要看 `KernelGenerator` 和具体 `ops/*.cc`：

- `runtime/onert/backend/cpu/KernelGenerator.cc`
- `runtime/onert/backend/cpu/ops/*`
- `runtime/onert/backend/acl_cl/KernelGenerator.cc`
- `runtime/onert/backend/acl_neon/KernelGenerator.cc`

控制流和一些内部逻辑走 builtin backend：

- `runtime/onert/core/src/backend/builtin/KernelGenerator.cc`

---

## 7. 你真正应该怎么判断一个 unsupported op

建议按下面顺序排查。

### 7.1 如果是 `one-import-*` 就报错

优先看 compiler 侧：

- `compiler/luci/import/src/GraphBuilderRegistry.cpp`
- `compiler/luci/import/src/Nodes/Circle*.cpp`

这说明模型还没进 `circle` / luci graph，就卡住了。

### 7.2 如果 `circle` 能产出，但 `nnfw_load_model_from_file` 报错

优先看 runtime loader：

- `runtime/onert/core/src/loader/BaseLoader.h`
- `runtime/onert/core/src/loader/CircleLoader.cc`

这说明 circle 里有这个 op，但 onert loader 没接。

### 7.3 如果能 load / prepare，但跑某 backend 报不支持

优先看 backend：

- `runtime/onert/backend/<backend>/Operation.lst`
- `runtime/onert/backend/<backend>/KernelGenerator.cc`
- `runtime/onert/backend/<backend>/ops/*`

这时先试 CPU backend，常见情况是：

- CPU 能跑
- ACL/特定 backend 不能跑

### 7.4 如果是 custom op

分两种：

- compiler custom：`CircleCustom`
- runtime custom：`ir::operation::Custom`

runtime 能加载 `CUSTOM` 不等于已经支持任意 custom op。

要看：

- `runtime/onert/core/src/loader/BaseLoader.h`
- `runtime/onert/api/nnfw/src/Session.cc`
- `runtime/onert/core/src/backend/builtin/KernelGenerator.cc`

以及有没有注册自定义 kernel。

---

## 8. 用一个具体例子理解：`sin / cos / abs / gathernd / scatternd`

这几个是很好的学习样例。

### 8.1 `abs / sin / cos`

这三个在主线里是比较典型的“compiler + runtime + CPU backend 都通”的 op。

你可以顺着下面路径读：

1. compiler importer 注册
   - `compiler/luci/import/src/GraphBuilderRegistry.cpp`
2. luci 节点
   - `compiler/luci/lang/include/luci/IR/Nodes/CircleAbs.h`
   - `compiler/luci/lang/include/luci/IR/Nodes/CircleSin.h`
   - `compiler/luci/lang/include/luci/IR/Nodes/CircleCos.h`
3. export 映射
   - `compiler/luci/export/src/CircleOps.lst`
4. runtime loader
   - `runtime/onert/core/src/loader/BaseLoader.h`
5. runtime IR
   - `runtime/onert/core/include/ir/operation/ElementwiseUnary.h`
6. CPU backend kernel
   - `runtime/onert/backend/cpu/ops/ElementwiseUnaryLayer.cc`

这里你会看到一个非常重要的设计：

- compiler 层是按具体节点建模
- runtime 层把它们归并到 `ElementwiseUnary`

### 8.2 `gathernd / scatternd`

这两个是很好的“compiler 支持但 runtime 主线不一定支持”的例子。

你可以顺着下面路径读：

1. compiler importer
   - `compiler/luci/import/src/GraphBuilderRegistry.cpp`
2. compiler export
   - `compiler/luci/export/src/CircleOps.lst`
3. luci type/shape inference
   - `compiler/luci/service/src/CircleTypeInferenceRule.cpp`
   - `compiler/luci/service/src/CircleShapeInferenceRule.cpp`
4. runtime loader 搜索
   - `rg "GATHER_ND|SCATTER_ND" runtime/onert`

如果 `runtime/onert` 里只有 schema，没有 loader case、没有 IR class、没有 backend kernel，
就说明它们停留在 compiler 层，没有接到 onert 主线执行链上。

这类 op 就是你后续补支持时最常见的工作对象。

---

## 9. 如果你要新增一个算子，通常要改哪里

下面分 compiler 和 runtime 两边列。

### 9.1 compiler 侧

最常见需要改的地方：

1. `compiler/luci/lang/include/luci/IR/Nodes/CircleMyOp.h`
2. `compiler/luci/import/include/luci/Import/Nodes/CircleMyOp.h`
3. `compiler/luci/import/src/Nodes/CircleMyOp.cpp`
4. `compiler/luci/import/src/GraphBuilderRegistry.cpp`
5. `compiler/luci/export/src/CircleOps.lst`
6. `compiler/luci/export/src/CircleBuiltinTypesExtractor.h`
7. `compiler/luci/service/src/CircleTypeInferenceRule.cpp`
8. `compiler/luci/service/src/CircleShapeInferenceRule.cpp`
9. 可能还要补：
   - `compiler/luci/service/src/Nodes/*`
   - `compiler/luci/partition/src/Nodes/*`
   - `compiler/luci/logex/src/CircleNodeSummaryBuilder.cpp`

### 9.2 runtime 侧

最常见需要改的地方：

1. 运行时 IR
   - `runtime/onert/core/include/ir/Operations.lst`
   - `runtime/onert/core/include/ir/operation/MyOp.h`
   - `runtime/onert/core/include/ir/Operations.Include.h`
2. verifier / validator / shape inference
   - `runtime/onert/core/src/ir/OperationValidator.cc`
   - `runtime/onert/core/src/compiler/StaticShapeInferer.*`
   - `runtime/onert/core/include/exec/DynamicShapeInferer.h`
   - `runtime/onert/core/src/exec/DynamicShapeInferer.cc`
3. loader
   - `runtime/onert/core/src/loader/BaseLoader.h`
   - `runtime/onert/core/src/loader/CircleLoader.cc`
   - `runtime/onert/core/src/loader/TFLiteLoader.cc`
4. backend 支持
   - `runtime/onert/backend/cpu/Operation.lst`
   - `runtime/onert/backend/cpu/KernelGenerator.cc`
   - `runtime/onert/backend/cpu/ops/*`
   - ACL 如需支持，再分别补
5. builtin backend
   - 如果是控制流 / 内部搬运 / custom 路径，要看 `runtime/onert/core/src/backend/builtin/*`
6. 测试
   - `runtime/tests/nnfw_api/lib/CircleGen.cc`
   - `runtime/tests/nnfw_api/src/*`

### 9.3 学习建议：先补 CPU，再考虑 ACL

如果你的目标是“先把模型跑起来”，建议顺序是：

1. compiler 能导入和导出
2. runtime loader 能接
3. runtime IR + shape/type/check 完整
4. CPU backend kernel
5. 测试通过
6. 再考虑 ACL / 其他 backend

否则你会同时和两套复杂度打架。

---

## 10. 文档里哪些内容可以信，哪些要小心

这个项目的文档不是完全跟代码同步的。

### 基本可信

- `docs/runtime/core.md`
- `docs/runtime/backend-api.md`
- `docs/runtime/executors.md`
- `docs/howto/how-to-build-compiler.md`
- `docs/howto/how-to-build-runtime.md`
- `docs/howto/how-to-introduce-a-new-operation-into-runtime.md`

### 需要谨慎

- `docs/overview/supported-operations.md`
  - 文档自己写了 `As of 2020-06-26`
- `docs/runtime/supported-operations-backend.md`
  - 文档自己写了 `As of 2021-03-08`

这两份文档可以当“历史参考”，但不要把它们当最终真相。

真正的真相在代码里：

- compiler 支持看 `compiler/luci/import` + `compiler/luci/export`
- runtime 支持看 `loader` + `ir` + `backend`

---

## 11. 推荐阅读顺序

如果你想“超级详细地学”，我建议按下面顺序。

### 第 1 阶段：先建立整体心智图

先读：

1. `README.md`
2. `docs/howto/how-to-build-compiler.md`
3. `docs/howto/how-to-build-runtime.md`
4. `docs/runtime/api.md`
5. `docs/runtime/core.md`
6. `docs/runtime/executors.md`

目标：

- 知道仓库不是单一程序，而是 toolchain + runtime
- 知道 `nncc`、`nnfw`、`Makefile.template` 的关系
- 知道 runtime 的核心模块分层

### 第 2 阶段：把编译链跑通

重点读：

1. `compiler/one-cmds/onecc`
2. `compiler/one-cmds/one-import`
3. `compiler/one-cmds/one-optimize`
4. `compiler/one-cmds/one-quantize`
5. `compiler/one-cmds/one-pack`
6. `compiler/one-cmds/onecc.template.cfg`

目标：

- 知道 one-* 命令怎么串起来
- 知道 `circle` 和 `nnpackage` 分别是什么阶段的产物

### 第 3 阶段：理解 `luci`

重点读：

1. `compiler/luci/import/src/GraphBuilderRegistry.cpp`
2. `compiler/luci/export/src/CircleOps.lst`
3. `compiler/luci/service/src/CircleTypeInferenceRule.cpp`
4. `compiler/luci/service/src/CircleShapeInferenceRule.cpp`
5. `compiler/luci/pass/src/*`

目标：

- 知道 compiler 如何建模一个 op
- 知道怎么判断一个 op 是不是 compiler 已支持

### 第 4 阶段：理解 runtime

重点读：

1. `runtime/onert/api/nnfw/src/APIImpl.cc`
2. `runtime/onert/api/nnfw/src/Session.cc`
3. `runtime/onert/core/src/loader/BaseLoader.h`
4. `runtime/onert/core/src/loader/CircleLoader.cc`
5. `runtime/onert/core/include/ir/Operations.lst`
6. `runtime/onert/core/src/compiler/Compiler.cc`

目标：

- 知道 `load -> prepare -> run` 各发生什么
- 知道 loader、runtime IR、compiler、executor 是怎么连起来的

### 第 5 阶段：理解 backend

重点读：

1. `runtime/onert/backend/cpu/Operation.lst`
2. `runtime/onert/backend/cpu/KernelGenerator.cc`
3. `runtime/onert/backend/cpu/ops/*`
4. `runtime/onert/core/src/backend/builtin/KernelGenerator.cc`

目标：

- 知道“一个 op 真能跑”最后落在哪
- 知道 builtin backend 和设备 backend 的边界

### 第 6 阶段：拿一个具体 op 做贯通练习

推荐用：

- 简单案例：`Abs`
- grouped runtime IR 案例：`Sin` / `Cos`
- compiler/runtime断层案例：`GatherNd` / `ScatterNd`

目标：

- 从 importer 一路追到 backend
- 形成自己的补算子模板

---

## 12. 实用命令清单

### 12.1 查一个算子在哪些层出现

```sh
rg "CircleGatherNd|GatherNd|GATHER_ND" compiler runtime onert-micro
```

### 12.2 看 compiler 是否接了

```sh
rg "CircleSin|CircleScatterNd" compiler/luci
```

### 12.3 看 runtime loader 是否接了

```sh
rg "BuiltinOperator_GATHER_ND|BuiltinOperator_SCATTER_ND" runtime/onert/core/src/loader
```

### 12.4 看 backend 是否有 kernel

```sh
rg "visit\\(const ir::operation::.*\\)|ElementwiseUnary::Type::SIN|ScatterNd|GatherNd" runtime/onert/backend
```

### 12.5 看某 backend 宣称支持哪些 operation family

```sh
sed -n '1,200p' runtime/onert/backend/cpu/Operation.lst
sed -n '1,200p' runtime/onert/backend/acl_cl/Operation.lst
sed -n '1,200p' runtime/onert/backend/acl_neon/Operation.lst
```

### 12.6 看 runtime IR 总表

```sh
sed -n '1,200p' runtime/onert/core/include/ir/Operations.lst
```

---

## 13. 你当前这个目标的最短学习路线

如果你的直接目标是“改 unsupported op”，我建议你不要平均用力，而是按下面路线学：

1. 先学 `onecc -> circle -> nnpackage` 这条编译链
2. 再学 `Session::load_model_from_path -> prepare -> run`
3. 再学 `BaseLoader / CircleLoader`
4. 再学 `runtime IR`
5. 最后学 CPU backend kernel

原因很简单：

- 大多数 unsupported op，不是全项目都要改
- 最常见是某一层没接上
- 把这几层分清楚后，你才能快速判断应该补哪里

---

## 14. 最后给你的行动建议

如果你现在就要开始实战，我建议按下面做：

1. 选一个你模型里真的报错的 op
2. 用 `rg` 在 `compiler/luci` 和 `runtime/onert` 里各搜一遍
3. 先判断卡在 compiler、loader 还是 backend
4. 如果要补支持，优先只做 CPU backend
5. 补一个最小测试

对这个仓库来说，最有价值的能力不是“记住所有目录”，而是形成下面这条反射链：

> 模型报错 -> 找到是哪一层 -> 找到入口文件 -> 看相邻算子的实现 -> 按最小闭环补齐

这才是你后面真的能持续改算子的关键。
