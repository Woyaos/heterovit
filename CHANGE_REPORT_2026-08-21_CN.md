# 2026-08-21 异构 ViT 项目变更与 Git 使用说明

## 1. 版本状态

- Windows 已安装 Git：`git version 2.47.0.windows.2`，因此没有重复下载安装。
- 本轮开始前，`D:\VS_project\heterogeneous` 内没有 `.git`，不存在可查询的提交历史。
- 现已初始化 `main` 分支，并建立当前状态基线提交：`6d391a9`。
- 基线跟踪 1644 个文件，约 22.5 MB；提交说明为
  `chore: establish heterogeneous ViT project baseline`。
- 该提交可以恢复“本轮工作完成后的当前状态”，但不能恢复本轮开始前的状态，因为当时
  没有 Git 快照或其他备份。

## 2. 本轮新增代码

### SimGrid 动态运行时

- `simgrid_baseline/15_dynamic_runtime/run_dynamic_case.py`：加入确定性请求到达、GPU/FPGA
  排队、共享链路竞争、设备侧编解码、双缓冲和请求完成时间统计。
- `simgrid_baseline/15_dynamic_runtime/run_load_sweep.py`：批量运行 20 至 120 req/s、四种
  调度策略的 24 个实验 case。
- `simgrid_baseline/15_dynamic_runtime/tune_adaptive_threshold.py`：扫描在途请求阈值
  1/2/4/8，依据 SLO 违约率、P95 和吞吐率选择阈值。
- `simgrid_baseline/15_dynamic_runtime/validate_results.py`：自动检查 case 数、任务记录、
  策略覆盖和预期结构性结论。
- `simgrid_baseline/15_dynamic_runtime/dynamic_config.json`：最终自适应阈值为 4，SLO 为
  30 ms，每个 case 运行 32 个请求。

### IRIS 异构运行时桥接

- `iris-main/apps/hetero_vit_runtime/PolicyHeteroEFT.cpp`：新增 IRIS 自定义
  `hetero_eft` 策略，综合 GPU/FPGA 计算、PCIe 通信、队列和运行目标选择设备。
- `iris-main/apps/hetero_vit_runtime/hetero_cost_model.h`：定义八个任务元数据字段及
  延迟/吞吐两种代价函数，并加入终端输出返回代价。
- `iris-main/apps/hetero_vit_runtime/generate_manifest.py`：把 SimGrid DAG 与 profile 转换为
  IRIS C 头文件。
- `iris-main/apps/hetero_vit_runtime/vit_task_manifest.h`：生成 101 个任务、124 条依赖和
  37 个 FPGA 候选任务；终端分类器计入 4000 字节返回数据。
- `iris-main/apps/hetero_vit_runtime/vit_dag_skeleton.c`：注册策略、创建完整 DAG，并支持
  `--latency` 与 `--throughput` 两种目标。
- `iris-main/apps/hetero_vit_runtime/kernel.openmp.c`：无硬件时使用的 OpenMP no-op kernel。
- `iris-main/apps/hetero_vit_runtime/test_cost_model.cpp`：验证通信权重、队列和 FPGA 驻留
  代价。
- `iris-main/apps/hetero_vit_runtime/Makefile`：构建 policy、测试、OpenMP kernel 和 DAG
  骨架。

### 真实硬件标定接口

- `simgrid_baseline/16_hardware_calibration/generate_measurement_plan.py`：从融合 DAG 生成
  138 行任务设备测量计划和 60 行双向/分段传输测量计划。
- `simgrid_baseline/16_hardware_calibration/validate_and_build_profile.py`：拒绝不完整数据，
  拟合 PCIe 固定延迟与有效带宽，并生成 SimGrid/IRIS 共用的实测 profile。
- `simgrid_baseline/16_hardware_calibration/test_measured_cost_model.py`：验证模拟器可直接使用
  实测任务时间。
- `simgrid_baseline/16_hardware_calibration/test_profile_builder.py`：在临时目录中验证传输
  曲线拟合和 profile 生成，不留下伪造硬件结果。

## 3. 本轮修改的已有代码

- `simgrid_baseline/12_full_dag_baselines/run_policy.py`：原先只接受敏感性 profile；现在
  仍保持原行为，同时支持完整、可追溯的 `target_measurement` 实测 profile。输出中的
  `valid_for_target_prediction` 由 profile 来源决定，不再固定为 false。
- `simgrid_baseline/EXPERIMENT_LOG.md`：追加动态运行时、IRIS 运行时桥接和硬件标定阶段。

## 4. 新增文档与实验输出

- `PROJECT_STATUS_CN.md`：项目当前完成度、结论和停止边界。
- `simgrid_baseline/15_dynamic_runtime/RESULTS_CN.md`：24 个负载 case 的中文结果说明。
- `simgrid_baseline/16_hardware_calibration/HARDWARE_STEPS_CN.md`：硬件到位后的逐步操作。
- `simgrid_baseline/15_dynamic_runtime/results/dynamic_load_sweep.csv`：动态负载汇总。
- `simgrid_baseline/15_dynamic_runtime/results_thresholds/threshold_sweep.csv`：阈值扫描汇总。
- `simgrid_baseline/16_hardware_calibration/results/target_measurement/`：待填写的任务、通信和
  平台测量模板。

## 5. 删除和排除项

- 本轮没有删除项目源代码或已有实验数据。
- `.gitignore` 排除了 `.venv`、`__pycache__`、`.pyc`、`.so`、`.exe`、IRIS 构建/安装
  目录和可重复生成的逐请求时间线。
- `vit.onnx` 大小约 346 MB，没有提交到普通 Git。其 SHA-256 为
  `0927578EEBCB7C10D329CEB3F6D4B2FC51C43BEFF1C120CC1189702D8D5FEE43`。
- 大模型通常使用对象存储、发布附件、数据集仓库或 Git LFS，而不是直接塞进普通 Git
  历史。只要本机文件未被手动删除，`.gitignore` 不会删除它。

## 6. 已完成验证

- 24 个动态负载 case 和 12 个阈值 case 的自动检查通过。
- 原敏感性 profile 回归结果保持 GPU-only `22.184 ms`。
- 实测 profile 读取测试和临时标定 profile 拟合测试通过。
- IRIS 3.0 在 OpenMP 后端完成编译；101 个任务和 124 条依赖分别在 latency、throughput
  目标下运行成功。
- 当前真实硬件模板故意验证失败：138 个任务设备时间、PCIe 测量和 26 个平台字段为空。
  这表示系统正确停在“必须接真实硬件”的位置。

## 7. Git 日常使用

查看当前修改：

```powershell
git status
git diff
```

开始一个新实验时建立分支：

```powershell
git switch -c experiment/real-pcie-calibration
```

选择文件并提交：

```powershell
git add path/to/file1 path/to/file2
git diff --cached
git commit -m "feat: calibrate PCIe transfer model"
```

查看历史和某次提交：

```powershell
git log --oneline --graph --decorate --all
git show 6d391a9
```

恢复某个尚未提交的文件到最近提交状态。该命令会丢弃该文件的本地修改，先用
`git diff` 检查：

```powershell
git restore path/to/file
```

从当前基线恢复某个文件：

```powershell
git restore --source 6d391a9 -- path/to/file
```

安全撤销一个已经提交的改动，团队项目通常使用 `revert`，因为它保留历史：

```powershell
git revert <commit-id>
```

查看旧版本而不影响当前分支，可以从旧提交创建一个恢复分支：

```powershell
git switch -c inspect/old-baseline 6d391a9
```

## 8. 工程团队通常怎么用

- `main` 保持可运行；每个实验或功能建立短期分支，完成后通过合并请求进入 `main`。
- 提交应小而完整，一次提交只做一件事，并在提交前检查 `git diff --cached` 和测试结果。
- 不提交虚拟环境、编译产物、密码、驱动、临时日志和大模型；模型使用 Git LFS 或外部
  存储，并记录版本、下载地址和校验和。
- 本地 Git 只能防止误改，不能防止硬盘损坏。应在 GitHub、GitLab 或 Gitee 创建私有
  远程仓库，再执行：

```powershell
git remote add origin <远程仓库地址>
git push -u origin main
```

- 不建议初学阶段使用 `git reset --hard` 或强制推送；优先使用分支、`git restore` 和
  `git revert`，这些操作更容易审计和补救。
