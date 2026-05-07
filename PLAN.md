# VISoR Reconstruction / Refinement 任务总结

日期：2026-05-01  
项目目录：`d:\USERS\yq\gitlab\heightVISoR`  
主要解释器：`D:\Tools\Anaconda_envs\envs\napari-env\python.exe`

## 1. 初始目标

本轮任务围绕 `VISoR_Reconstruction` 工程展开，重点是理解并改造 `brain_reconstruction` 相关流程，将 `YQReconstructionScripts\B85_Test` 的后处理/精修流程嵌入原有 VISoR Reconstruction 页面。

初始约束和偏好如下：

- 不完整阅读整个 `YQReconstructionScripts`，只关注 `YQReconstructionScripts\B85_Test` 必需逻辑。
- B85 流程不是从原始 `.flsm` 开始，而是基于标准 Reconstruction 已生成的：
  - `Reconstruction/SliceImage/4.0`
  - `Reconstruction/SliceTransform`
- B85 需要合并到现有 Reconstruction 页面中。
- 后续 UI 需要优化。
- 默认测试数据集为：
  - `D:\USERS\yq\Dataset\b85\151_640.visor`
- 用户已在以下目录补充过部分 `.npy` 文件，可用于跳过部分 B85 中间步骤：
  - `D:\USERS\yq\Dataset\b85\Reconstruction\B85Refinement_DebugRun\temp_block`

## 2. 对原工程流程的理解

### 2.1 主入口

`VISoR_Reconstruction/visor_reconstruction_ui/brain_reconstruction.py` 主要负责启动 PyQt 主窗口、加载 pipeline 和页面。

真实 Reconstruction 流程主要发生在：

- `VISoR_Reconstruction/visor_reconstruction_ui/pages/whole_brain_reconstruct/whole_brain_reconstruct.py`
- `VISoR_Reconstruction/reconstruction_executor/generator.py`
- `VISoR_Reconstruction/reconstruction_executor/executor.py`

### 2.2 标准 Reconstruction 流程

原始 Reconstruction 页通过：

```python
gen_brain_reconstruction_pipeline(...)
```

生成任务 JSON，然后通过子进程调用：

```python
executor.main(...)
```

标准任务通过 `multiprocessing.Pipe` 回传：

- `message`
- `status`
- `progress`
- `stop` 控制信号

### 2.3 Image Reconstruction 相关流程

标准流程中，`reconstruct image` 前面及其自身依赖前置处理结果，主要围绕：

- 原始数据 / channel / slice 元信息
- sample transform / slice transform
- resample
- image reconstruction
- 输出 `SliceImage/4.0`

后续用户明确要求：`Refinement` 中需要涵盖 `reconstruct image` 前面的流程，并且最终也包括 `reconstruct image` 自身。

## 3. B85 / Refinement 集成方案

### 3.1 模式名称调整

最初设计为：

- `Standard VISoR`
- `B85 Refinement`

后来按用户要求将界面名称改为：

- `Standard VISoR`
- `Refinement`

### 3.2 新增 B85 运行适配层

新增或持续改造的核心文件：

```text
VISoR_Reconstruction/reconstruction_executor/b85_runner.py
```

核心职责：

- 定义 Refinement 配置结构。
- 从当前 `.visor` 数据集自动推导输入路径、通道、slice 范围、输出目录。
- 调用标准 Reconstruction 的前置步骤。
- 参数化封装 B85_Test 中原有硬编码脚本逻辑。
- 通过 Pipe 回传状态、日志和进度。

配置重点包括：

- `dataset_file`
- `dataset_path`
- `output_root`
- `temp_root`
- `reference_channel_id`
- `output_channel_ids`
- `start_slice`
- `end_slice_exclusive`
- `pixel_size`
- `block_size`
- `gap`
- `selected_steps`

### 3.3 自动通道和路径推导

Refinement 自动从当前 `VISoRData` 和已有 Reconstruction 结果推导：

- `nameFormat`
- `imgFormat`
- `stFormat`
- 参考通道
- 输出通道

参考通道规则：

- 优先选择 `640nm`
- 如果没有 `640nm`，选择最高波长通道

输出通道规则：

- 默认包含所有能够找到切片图像的通道

### 3.4 Refinement 步骤

按 B85_Test 脚本语义整理后的主要步骤：

1. `reconstruct_sample`
2. `reconstruct_image`
3. `step1_1`
4. `step1_2`
5. `step1_3`
6. `step2`
7. `extract_surface_failed`
8. `step3`
9. `check_xy`
10. `step4`
11. `step4_channel`

其中：

- `reconstruct_sample` 和 `reconstruct_image` 是后来按用户要求加入 Refinement 的标准流程内容。
- `step1_1` 到 `step4_channel` 来自 B85_Test 的精修后处理逻辑。

### 3.5 输出元数据

Refinement 成功后设计为写出：

- `Parameters.json`
- `RunSummary.json`
- `BrainTransform/BrainTransform.json`
- `BrainImage/BrainImage.json`

并且不覆盖原始 `.visor`。如后续 UI 继续使用 ROI / registration，则在内存中把当前 dataset 的 `brain_transform` 指向 Refinement 输出。

## 4. UI 改造和优化

### 4.1 WholeBrainReconstructPage

主要修改文件：

```text
VISoR_Reconstruction/visor_reconstruction_ui/pages/whole_brain_reconstruct/whole_brain_reconstruct.py
```

完成内容：

- 增加模式切换：
  - `Standard VISoR`
  - `Refinement`
- 增加 Refinement 参数区：
  - 输出目录
  - temp 目录
  - 参考通道
  - 输出通道
  - slice 起止
  - pixel size
  - block size
  - gap
  - step 勾选列表
- Start / Stop 复用原有子进程模型。
- Refinement 通过 Pipe 回传 `status/progress/message/result`。
- 增加运行状态区、日志显示、进度更新。

### 4.2 主界面美化

主要修改文件：

```text
VISoR_Reconstruction/visor_reconstruction_ui/brain_reconstruction.py
```

优化方向：

- 主窗口整体视觉更现代。
- 左侧导航控件字体增大。
- 左侧条目宽度和高度调整，避免：
  - `Reconstruction`
  - `Manual Image ...`
  等文字显示不完整。
- 色彩、边界、间距更符合现代桌面 UI。

### 4.3 Refinement 参数区布局修复

用户反馈 UI 优化后，Refinement 参数设置区域数字和控件被挤压。

对应修复：

- 增加参数区最小宽度。
- 使用 scroll area 容纳 Refinement 参数。
- 数字控件设置合理最小宽度。
- `Slice range`、`Pixel size`、`Block size`、`Gap` 等控件避免文字和数字挤压。

## 5. 去除对 YQReconstructionScripts 的运行时依赖

用户要求：

> 将 utils 用到的内容全部复制到 VISoR Reconstruction 文件夹里面，不要再调用到 YQReconstructionScripts 里面的内容。

已新增本地包：

```text
VISoR_Reconstruction/reconstruction_executor/b85_utils/
```

主要包含：

- `__init__.py`
- `common.py`
- `common0313.py`
- `common0424.py`
- `ome_tiff.py`
- `step1_1_methods.py`
- `step1_2_use_block.py`
- `step1_3_CalLoss.py`
- `step2_extract.py`
- `step3_align.py`
- `step4_ContiuneProcessTransform.py`
- `torch_losses.py`
- `yq_elastix_files.py`

同时将 `b85_runner.py` 中的导入改为：

```python
VISoR_Reconstruction.reconstruction_executor.b85_utils
```

不再通过：

```python
utils.step1_1_methods
YQReconstructionScripts...
```

等路径导入。

## 6. 编码和注释修复

复制 B85 utils 后，曾出现：

```text
SyntaxError: (unicode error) 'utf-8' codec can't decode bytes ...
```

原因：

- 原脚本中存在非 UTF-8 编码内容。
- 部分中文注释或历史编码被 Python 以 UTF-8 读取时失败。

处理：

- 将 B85 utils 相关文件统一清理为可被 Python 读取的编码。
- 将引发问题的注释改为英文。
- 对 `b85_utils` 相关文件做过导入和编译检查。

## 7. reconstruct image 阶段 Pipe / UI 崩溃修复

### 7.1 问题现象

用户反馈：

- `reconstruct image` 完成部分 image 任务后界面消失。
- 日志中出现：

```text
BrokenPipeError: [WinError 109] 管道已结束。
EOFError
```

### 7.2 原因分析

主要原因有两层：

1. `whole_brain_reconstruct.py` 中的 `listen()` 在 `WorkerThread` 后台线程里直接更新 Qt 控件：
   - `progressBar.setValue(...)`
   - `label_status.setText(...)`
   - `textBrowser.append(...)`

   PyQt 控件必须在主线程更新，否则在高频日志/进度更新时可能导致主窗口崩溃。

2. Refinement 内部调用标准 executor 的 `reconstruct_image` 时，标准 executor 也会读写 Pipe，容易与外层 UI 控制 Pipe 产生冲突。

### 7.3 修复内容

修改文件：

```text
VISoR_Reconstruction/visor_reconstruction_ui/pages/whole_brain_reconstruct/whole_brain_reconstruct.py
VISoR_Reconstruction/reconstruction_executor/b85_runner.py
VISoR_Reconstruction/tools/common/common.py
```

修复点：

- 新增 `pipe_message_received = QtCore.pyqtSignal(dict)`。
- 后台线程只负责读 Pipe 并 emit signal。
- UI 更新统一在主线程 `_handle_pipe_message()` 中执行。
- Stop 信号发送改为安全处理。
- Pipe 关闭时不让界面消失，而是写日志并显示状态。
- 在 `b85_runner.py` 中增加 `_StandardExecutorPipe`，隔离标准 executor 与外层 UI Pipe。
- `WorkerThread` 使用 `finally` 恢复 `sys.stdout`，避免异常后污染 UI 日志状态。

对应提交：

```text
8d4b210 Fix refinement UI process communication
```

## 8. reconstruct image 日志不换行修复

### 8.1 问题现象

用户反馈：

> 在 reconstruct image 的时候，输出的内容并在了一行，没有分行。

### 8.2 原因分析

Pipe 或 stdout 回传的日志可能包含：

- `\n`
- `\r`
- `\r\n`

之前直接调用：

```python
self.textBrowser.append(...)
```

在当前 QTextBrowser 模式下，多次 append 或混合换行内容可能被压成一行显示。

### 8.3 修复内容

在 `whole_brain_reconstruct.py` 中新增统一日志函数：

```python
_append_log(...)
```

处理逻辑：

- 将 `\r\n` 和 `\r` 统一为 `\n`。
- 按行拆分。
- 使用 `QTextCursor.insertBlock()` 和 `insertText()` 写入日志。
- 自动滚动到最新日志。

验证结果：

```text
'line1\nline2\nline3\nline4'
Running 250
```

对应提交并已推送：

```text
60500d7 Fix reconstruction log line breaks
```

推送命令：

```powershell
git push -u origin master
```

推送结果：

```text
master -> master
```

## 9. gitignore 调整

用户要求：

- 根据修改内容同步 `.gitignore`。
- 上传修改部分。
- 剔除被改动的文件夹，不要让新改动被忽略。

已做方向：

- 确认 `VISoR_Reconstruction/reconstruction_executor` 相关新增内容不被 `.gitignore` 排除。
- 确认 `b85_utils` 可被 Git 跟踪。
- 保留对 `YQReconstructionScripts/` 等非目标目录的忽略策略，避免误上传大量旧脚本或数据。

## 10. 运行和测试命令

### 10.1 启动 UI

推荐使用当前环境解释器：

```powershell
D:\Tools\Anaconda_envs\envs\napari-env\python.exe VISoR_Reconstruction\visor_reconstruction_ui\brain_reconstruction.py
```

### 10.2 编译检查

```powershell
D:\Tools\Anaconda_envs\envs\napari-env\python.exe -m py_compile VISoR_Reconstruction\visor_reconstruction_ui\pages\whole_brain_reconstruct\whole_brain_reconstruct.py
```

也曾对以下文件做过编译检查：

```text
VISoR_Reconstruction/reconstruction_executor/b85_runner.py
VISoR_Reconstruction/reconstruction_executor/executor.py
VISoR_Reconstruction/tools/common/common.py
VISoR_Reconstruction/reconstruction_executor/b85_utils/*.py
```

### 10.3 UI smoke test

使用 offscreen 方式加载：

```text
D:\USERS\yq\Dataset\b85\151_640.visor
```

验证项：

- 页面可实例化。
- `.visor` 可加载。
- 可切换到 `Refinement`。
- Pipe message 可更新状态和进度。
- 多行日志能正确显示为多行。

## 11. 关键文件清单

### 11.1 Reconstruction / Refinement UI

```text
VISoR_Reconstruction/visor_reconstruction_ui/brain_reconstruction.py
VISoR_Reconstruction/visor_reconstruction_ui/pages/whole_brain_reconstruct/whole_brain_reconstruct.py
```

### 11.2 Executor / Runner

```text
VISoR_Reconstruction/reconstruction_executor/b85_runner.py
VISoR_Reconstruction/reconstruction_executor/executor.py
VISoR_Reconstruction/reconstruction_executor/generator.py
```

### 11.3 本地 B85 utils

```text
VISoR_Reconstruction/reconstruction_executor/b85_utils/
```

### 11.4 通用 UI 线程工具

```text
VISoR_Reconstruction/tools/common/common.py
```

### 11.5 数据 / 测试路径

```text
D:\USERS\yq\Dataset\b85\151_640.visor
D:\USERS\yq\Dataset\b85\Reconstruction\B85Refinement_DebugRun\temp_block
```

## 12. 已提交记录

最近相关提交：

```text
60500d7 Fix reconstruction log line breaks
8d4b210 Fix refinement UI process communication
235dc2a complete fix 5
8728016 complete struct fix 4
ff94704 complete fix 3
31c8bd4 revise b85 runner
a973687 try fix complete 2
eb7a73e add gpu resample37
```

其中明确在本轮末尾完成并 push 的提交：

```text
60500d7 Fix reconstruction log line breaks
```

## 13. 当前状态

截至本总结生成时：

- Refinement 已集成到 Reconstruction 页。
- Refinement 包含标准 Reconstruction 的前置内容和 `reconstruct_image`。
- B85 utils 已复制到 `VISoR_Reconstruction` 内部，不再依赖 `YQReconstructionScripts` 运行时导入。
- UI 主界面和 Refinement 参数区已做过现代化和可读性优化。
- `reconstruct image` 阶段 UI 消失问题已修复。
- `reconstruct image` 阶段日志不换行问题已修复。
- 最近一次修复已提交并推送到 `origin/master`。

### 13.1 本次继续任务补充

在继续检查下游页面时发现：

- Refinement 成功后虽然会把 `dataset.brain_transform` 指向新的 `visor_brain.txt`，但 ROI Reconstruction 页面不会自动刷新启用。
- ROI pipeline 内部仍硬编码读取标准 `Reconstruction/BrainTransform/visor_brain.txt`，没有使用当前 dataset 上的 refinement transform。
- Brain Registration 页面依赖标准 `BrainImage/freesia_*.json` 命名，而 Refinement 之前只写出纯文本 image list，后续 registration 可能找不到输入文件。

本次补充修复：

- `roi_reconstruction_generator.py` 改为优先使用 `dataset.brain_transform`，没有时再回退标准路径。
- `whole_brain_reconstruct.py` 在 Refinement 成功后加载新的 `BrainTransform.json`、`BrainImage.json` 到当前 dataset，并刷新 Brain Registration / ROI Reconstruction 页面。
- `b85_runner.py` 在生成 Refinement BrainImage 后同步写出兼容 Brain Registration 的 `freesia_{pixel_size}_C{channel}_{channel_name}.json`，并在 `BrainImage.json` 中记录 `FreesiaFile`。
- `brain_registration.py` 刷新 dataset 时清空旧通道列表，避免多次刷新产生重复通道；运行时优先使用当前 `dataset.brain_transform` 推导 registration 输出目录。

已完成编译检查：

```powershell
python -m py_compile VISoR_Reconstruction\reconstruction_executor\b85_runner.py VISoR_Reconstruction\reconstruction_executor\roi_reconstruction_generator.py VISoR_Reconstruction\visor_reconstruction_ui\pages\whole_brain_reconstruct\whole_brain_reconstruct.py VISoR_Reconstruction\visor_reconstruction_ui\pages\brain_registration\brain_registration.py
```

当前环境限制：

- 本机 `D:\Tools\Anaconda_envs\envs\napari-env\python.exe` 不存在。
- 默认 `python` 缺少 `SimpleITK`，无法实际启动完整 UI 或跑真实 Refinement。
- `D:\USERS\yq\Dataset\b85\151_640.visor` 当前不可访问。

### 13.2 B94 测试与 step1 强制运行补充

用户指定当前测试解释器：

```powershell
d:\MiceRecon\venv\Scripts\python.exe
```

在当前工具环境中，直接执行该解释器会因为找不到 Python 标准库 `encodings` 失败；临时设置以下变量后可正常使用该解释器和 venv 依赖：

```powershell
$env:PYTHONHOME='C:\Users\Xucheng\Anaconda3'
$env:PYTHONPATH='D:\MiceRecon\venv\Lib\site-packages'
```

已用上述方式完成：

```powershell
d:\MiceRecon\venv\Scripts\python.exe -m py_compile VISoR_Reconstruction\reconstruction_executor\b85_runner.py VISoR_Reconstruction\reconstruction_executor\b85_utils\step1_1_methods.py VISoR_Reconstruction\reconstruction_executor\b85_utils\step1_2_use_block.py VISoR_Reconstruction\reconstruction_executor\b85_utils\step1_3_CalLoss.py
```

并完成 offscreen UI smoke：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
$env:VISOR_UI_SMOKE_EXIT_MS='1000'
d:\MiceRecon\venv\Scripts\python.exe VISoR_Reconstruction\visor_reconstruction_ui\brain_reconstruction.py
```

用户指定测试数据：

```text
Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\67-73test.visor
```

当前工具普通权限下不可见 `Y:` 映射盘；提升权限下曾验证 `Test-Path` 为 `True`，但后续读取/运行命令的审批超时，尚未完成真实数据运行。

为满足“step1_1、step1_2、step1_3 遇到可跳过文件也必须真实运行后才能进入后续步骤”的要求，已修改：

- `b85_runner.py`
  - step1_1 开始前清理当前 slice-pair 的旧 block 输出。
  - step1_2 开始前清理旧 `pos_*.txt` 输出。
  - step1_3 开始前清理旧 `refine_*_pars.npy`、`*_np_array.npy`、`loss_*.txt`、`moved/save` 输出。
  - 每个 step 结束后新增输出校验；缺少 block、position 或 refine 参数时直接抛错停止，不允许进入后续步骤。
- `step1_1_methods.py`
  - block tif 已存在时不再 `continue`，改为覆盖写入。
  - `taskFun` 支持外部传入 temp 目录名，避免 UI 自定义 `temp_root` 时写错位置。
- `step1_2_use_block.py`
  - `pos_*.txt` 已存在时不再提前 return，改为重新计算并覆盖。
- `step1_3_CalLoss.py`
  - `loss_*.txt` 已存在时不再跳过，改为重新计算。

### 13.3 B94 step1_1 空目录问题修复

用户反馈：

```text
Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\Reconstruction\Refinement\temp_block\67_68
```

没有对应图像生成。

定位结果：

- `67-73test.visor` 可读取，配置推导正常：
  - dataset: `1_1_B94`
  - slice range: `67 -> 73`
  - reference channel: `2 / 561nm_10X`
  - temp root: `Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\Reconstruction\Refinement\temp_block`
- 单独运行 `67 -> 68` 的 `step1_1` 后发现 worker 内实际报错：

```text
NameError: name 'write_ome_tiff' is not defined
```

此前 `step1_1_multiprocess()` 使用 `Pool.apply_async()` 后没有调用 `res.get()`，导致 worker 异常被吞掉，主流程只看到 `All end--`，最终留下空目录。

修复内容：

- `step1_1_methods.py`
  - 显式导入 `from .ome_tiff import write_ome_tiff`。
  - `step1_1_multiprocess()` 调用 `res.get()`，worker 异常会抛回主流程。
  - 当没有 block 通过原前景阈值时，fallback 到信号最强的最多 64 个 block，避免 wholebody 数据前景比例偏低时生成空目录。
  - 移除第二层会把 fallback block 再次过滤掉的 `hollow_scale < 0.4` 跳过逻辑。
- `step1_2_use_block.py`、`step1_3_CalLoss.py`
  - 同样对 multiprocessing result 调用 `res.get()`，避免后续 step worker 异常被吞。

验证结果：

```text
Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\Reconstruction\Refinement\temp_block\67_68
```

已生成：

- `209` 个 `*up_temp_all.tif`
- `209` 个 `*down_temp_all.tif`
- 共 `418` 个 block 图像文件

并通过 `_validate_step1_1_outputs(...)` 校验。

### 13.4 B94 step1_1 删除权限问题修复

用户继续反馈：

```text
PermissionError: [WinError 5] 拒绝访问。:
Y:/SIAT_SIAT/YaoYuchen/Wholebody/Mouse/B94\Reconstruction\Refinement\temp_block\67_68\0_12down_temp_all.tif
```

定位结果：

- `Y:` 网络盘连接正常，目标文件不是只读。
- 当前账号可以在该目录中新建和覆盖文件。
- 当前账号没有删除该目录下部分文件的权限，`Remove-Item` 会返回 `Access denied`。
- 前一次为了强制重跑 step1 加入的清理逻辑会先尝试删除旧 `temp_block\67_68`，因此在没有 delete 权限的 SMB 目录上触发 `PermissionError`。

修复内容：

- `b85_runner.py`
  - `_clear_path()` 捕获 `OSError`，删除失败时不再中断，而是记录：
    `Cannot delete existing Refinement output, will overwrite if possible`
  - step1 校验改为基于本轮运行开始后的文件修改时间，避免旧文件冒充新输出。
  - step1_1 / step1_2 / step1_3 仍会重新覆盖写入输出；只是不再依赖删除权限。

验证：

- 在同一目录下用 `write_ome_tiff()` 对已有 probe tif 连续覆盖写入两次成功。
- 关键文件重新 `py_compile` 通过。

### 13.5 B94 step1_1 SimpleITK TIFF 读取问题修复

用户反馈 step1_1 报错：

```text
RuntimeError: Exception thrown in SimpleITK ReadImage:
itk::ERROR: TIFFImageIO(...): Problem reading the row: 0
```

复现与定位：

- 单独读取 `1_1_B94_068_561nm_10X.tif` 可成功，但耗时较长。
- 读取 `1_1_B94_073_561nm_10X.tif` 时，SimpleITK/ITK TIFF reader 报：

```text
AdobeDeflate scanline decoding is not implemented
Problem reading the row: 0
```

- `tifffile` 可以读取该文件的局部 z range。
- 原 step1_1 会用 `SimpleITK.ReadImage()` 读取整张大 TIFF，再从中取 overlap 区域；对 B94 这种 2.7GB 网络 TIFF 既慢又容易触发 ITK TIFF reader 的压缩/目录兼容问题。

修复内容：

- `step1_1_methods.py`
  - 新增 `tifffile` 局部读取逻辑。
  - step1_1 不再整张读取 TIFF，而是根据 z 大小只读取用于 block 生成的 overlap z range。
  - 局部数组转为 SimpleITK 图像后再 resample 到统一 `refSize`。
- `b85_runner.py`
  - step1_1 改为顺序执行，避免多个 worker 同时读取 2.7GB 网络 TIFF 造成 SimpleITK / SMB 读取不稳定。

实测结果：

- 单独运行 `72 -> 73` 的 `step1_1` 已完成。
- 输出目录：

```text
Y:\SIAT_SIAT\YaoYuchen\Wholebody\Mouse\B94\Reconstruction\Refinement\temp_block\72_73
```

已生成：

- `358` 个 `*up_temp_all.tif`
- `358` 个 `*down_temp_all.tif`

全范围 `67 -> 73` 的 step1_1 输出统计：

```text
67_68: up 368, down 368
68_69: up 374, down 374
69_70: up 385, down 385
70_71: up 400, down 400
71_72: up 409, down 409
72_73: up 358, down 358
```

并通过：

```python
_validate_step1_1_outputs(config)
```

## 14. 后续建议

建议后续重点验证：

1. 使用小范围 slice，例如 `151` 到 `155` 的 exclusive 语义，跑完整 Refinement。
2. 检查 `reconstruct_sample`、`reconstruct_image`、B85 各 step 的跳过逻辑是否符合已有中间文件状态。
3. 验证 Stop 按钮在长时间 `reconstruct image` 和 B85 step 中的响应。
4. 验证 `BrainTransform/BrainTransform.json` 和 `BrainImage/BrainImage.json` 能否被后续 ROI / registration UI 正确使用。
5. 如果 B85 utils 中仍有历史调试 `print` 太多，可进一步改成受控 logger 或按 UI 选项控制日志详细程度。

