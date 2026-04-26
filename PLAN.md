# 将 B85_Test 合并进 VISoR Reconstruction 页

## Summary
- `brain_reconstruction.py` 只负责启动 PyQt 主窗口、加载 `ReconstructionPipeline`，真正流程在 Reconstruction 页和 `reconstruction_executor`。
- 现有 Reconstruction 页通过 `gen_brain_reconstruction_pipeline(...)` 生成任务 JSON，再用 `executor.main(...)` 在子进程执行，并通过 `Pipe` 回传日志、状态和进度。
- `YQReconstructionScripts\B85_Test` 是基于已有 `Reconstruction\SliceImage\4.0` 和 `SliceTransform` 的后处理/精修流程，不是从原始 `.flsm` 直接开始。
- 按已确认偏好：B85 合并进现有 Reconstruction 页，提供可选步骤、自动通道推导，并优先优化参数、进度和日志控制。

## Key Changes
- 在 Reconstruction 页增加模式选择：`Standard VISoR` / `B85 Refinement`。标准流程保持原行为不变。
- 新增 B85 运行适配层，例如 `VISoR_Reconstruction/reconstruction_executor/b85_runner.py`：
  - 定义 `B85Config`：`output_root`、`reference_channel_id`、`output_channel_ids`、`start_slice`、`end_slice_exclusive`、`pixel_size=4.0`、`block_size=250`、`gap=500`、`selected_steps`。
  - 自动从当前 `VISoRData.file/path/channels` 和已存在的 `Reconstruction/SliceImage/4.0`、`Reconstruction/SliceTransform` 推导 `nameFormat/imgFormat/stFormat`。
  - 默认参考通道优先选 `640nm`，否则选最高波长；输出通道默认包含所有能找到切片图像的通道。
  - 不直接导入 `MainGUI_b85_Test*.py`，而是把其中 step 调用封装为参数化 runner，避免硬编码 `Y:\...` 路径。
- B85 步骤按当前脚本语义执行：
  - `step1_1`：生成相邻切片 block。
  - `step1_2`：block 粗配准，生成位置结果。
  - `step1_3`：计算 loss/NCC，生成 `refine_{slice}_pars.npy`。
  - `step2`：从 refine 参数提取 `uz/lz/us/ls` surface。
  - `extract_surface_failed`：补齐缺失 surface。
  - `step3`：surface 对齐，生成 `uxy/lxy`。
  - `check_xy`：补齐首尾 `uxy/lxy`。
  - `step4`：生成 `udf/ldf`、`visor_brain.txt` 和参考通道 brain image。
  - `step4_channel`：为非参考通道生成 brain image。
- Reconstruction 页 UI 优化集中在 `WholeBrainReconstructPage`：
  - B85 参数区：参考通道、输出通道、slice 起止、输出目录、temp 目录、step 勾选。
  - 运行状态区：当前步骤、总进度、每步状态、日志。
  - Start/Stop 复用现有子进程模型；B85 runner 通过 `Pipe` 回传 `status/progress/message`。
- 成功后写入 B85 结果元数据：
  - `Parameters.json` 记录 B85 配置。
  - `RunSummary.json` 记录步骤、输入模板、输出路径。
  - 生成可被后续 UI 使用的 `BrainTransform/BrainTransform.json` 和 `BrainImage/BrainImage.json`。
  - 不覆盖原始 `.visor`；如需继续 ROI/registration，UI 在内存中把当前 dataset 的 `brain_transform` 指向 B85 输出。
- 更新启动脚本使用当前解释器：
  - `D:\Tools\Anaconda_envs\envs\napari-env\python.exe`
  - 保持项目根目录加入 `PYTHONPATH`，避免旧 bat 里的 `C:\softwares\...` 路径。

## Test Plan
- 用 napari-env 做导入检查：`PyQt5`、`SimpleITK`、`torch`、`VISoR_Reconstruction`、`YQReconstructionScripts.B85_Test` 相关封装。
- 单元测试 B85 配置推导：
  - 从 `VISoRData` 推导通道列表。
  - 从 `SliceImage/4.0` 推导 `nameFormat/imgFormat`。
  - 从 `SliceTransform` 推导 `stFormat`。
  - 缺少输入文件时给出明确错误，不启动长任务。
- UI smoke test：
  - 启动主窗口。
  - 载入 `.visor`。
  - 切换 Standard/B85 模式。
  - B85 参数区按 dataset 自动填充。
- 手工验收：
  - 先用小范围 slice，例如 `151` 到 `155` 的 exclusive 语义，运行单步/多步。
  - 验证 temp 输出、`visor_brain.txt`、brain image 序列、日志和停止按钮。
  - 验证 Standard Reconstruction 未受影响。

## Assumptions
- 不继续阅读整个 `YQReconstructionScripts`；只迁移/适配 `YQReconstructionScripts\B85_Test` 必需逻辑。
- B85 流程依赖标准 Reconstruction 已生成的 `SliceImage/4.0` 和 `SliceTransform`。
- slice 结束值沿用脚本语义：`end_slice_exclusive`，即 Python `range(start, end)`。
- B85 外部旧硬编码路径全部改为 UI 参数或自动推导路径。
- `YQReconstructionScripts.CRH.common0313`、`Test0604` 等跨目录依赖不作为运行时依赖保留；需要的 `Preprocess/fill_outside_yq/create_folder` 使用 B85 内已有实现或主工程 helper 替代。
