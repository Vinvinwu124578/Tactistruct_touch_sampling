# Tactistruct Touch Sampling

这个仓库是当前 Tactile Lab / Isaac Sim / TacTip 触觉采样代码的备份版本。它重点保存代码、脚本和训练/可视化工具，不包含正在生成的数据集、ShapeNet 模型、IsaacLab 主仓库或外部机器人 USD 资产。

## 当前主要用途

当前最重要的流程是用 Isaac Sim 中的 UR5 + TacTip 对 ShapeNet 物体采集触觉图像和对应的 GT local patch 点云。

推荐主入口：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_large_touch_model_clone_collection_aligned_gt.ps1
```

当前这版采集逻辑的重点：

- 使用 `surface_normal_coverage` 选触点，优先覆盖物体表面不同区域。
- TacTip 接触方向尽量与表面法向量一致。
- GT patch 使用 TacTip raycast 命中点。
- 默认关闭 dense mesh-surface fallback，避免把触觉图像没有反映出来的物体表面点加入 GT。
- 如果某个触点无法得到真实 TacTip raycast GT，会跳过该触点，而不是用不真实的 mesh 表面点补齐。
- 触觉图像每像素 deformation 受 `--tactile_gym_max_pixel_deformation 0.008` 限制。

## 外部依赖和未包含内容

这个仓库不是完整运行环境。下面这些目录需要在本机另外准备：

| 路径/资源 | 用途 |
| --- | --- |
| `IsaacLab-main/` | Isaac Lab 主代码和 Isaac Sim 运行入口。 |
| `Tactile_Lab_External_Assets/` | UR5 + TacTip USD、机器人资产和物理配置。 |
| `ShapeNetCore/ShapeNetCore/` | ShapeNet 物体 mesh 数据。 |
| `TouchSDF-master/` | TouchSDF 相关数据、TacTip contact points、结果目录等。 |
| `outputs/` | 采集输出、日志、HTML 可视化、训练结果；此仓库不包含。 |

## 顶层文件和目录

| 文件/目录 | 作用 |
| --- | --- |
| `README.md` | 当前说明文件，解释仓库结构和关键脚本用途。 |
| `BACKUP_NOTES.md` | 本次备份说明，记录备份包含/排除内容和当前 GT 行为。 |
| `.codex` | Codex 桌面环境标记文件。 |
| `.dockerignore` | Docker 构建时忽略的文件规则。 |
| `.flake8` | Python flake8 代码检查配置。 |
| `.gitattributes` | Git 文件属性配置。 |
| `.gitignore` | Git 忽略规则，避免提交输出、缓存等文件。 |
| `.pre-commit-config.yaml` | pre-commit 自动格式化/检查配置。 |
| `.vscode/.gitignore` | VS Code 本地配置忽略规则。 |
| `.vscode/tools/setup_vscode.py` | 生成 VS Code Python/Isaac Sim 环境路径的辅助脚本。 |
| `scripts/` | 采集、训练、测试、可视化和 RL 启动脚本。 |
| `source/` | Isaac Lab extension 源码、任务环境、资产引用和工具模块。 |
| `third_party/` | 第三方代码目录，目前备份中为空或仅作为占位。 |

## `scripts/` 文件说明

### 触觉采集脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/collect_shapenet_touch_charts_isaac_tactile_gym_clone.py` | 当前最重要的采集脚本。用 Isaac Sim + UR5 + TacTip 生成触觉图像和 GT patch。包含 surface normal coverage、法向对齐、TacTip raycast GT、真实触觉 mask/range 过滤、跳过坏 GT 等逻辑。 |
| `scripts/run_large_touch_model_clone_collection_aligned_gt.ps1` | 当前推荐的大规模采集 launcher。配置 1000 样本、aligned GT、raycast-only GT、8mm 像素 deformation 限制等参数。 |
| `scripts/run_large_touch_model_clone_collection_fast.ps1` | 较早的快速采集 launcher，主要用于快速试跑。 |
| `scripts/run_large_touch_model_clone_collection_fast.cmd` | Windows cmd 版本的快速采集入口。 |
| `scripts/run_large_touch_model_clone_collection_resume.ps1` | 续跑/恢复采集用 launcher。 |
| `scripts/collect_shapenet_touch_charts_isaac.py` | 旧版/基础 Isaac ShapeNet 触觉 chart 采集脚本。 |
| `scripts/collect_shapenet_touch_charts_isaac.py.bak_20260520_135530` | 旧版采集脚本备份。 |
| `scripts/collect_shapenet_touch_charts_isaac_ur_ik_experiment.py` | UR5 IK 实验版本采集脚本，用于调试路径规划、IK 和触觉采样。 |
| `scripts/run_isaac_shapenet_touch_charts.ps1` | 旧版 ShapeNet 触觉 chart 采集 launcher。 |
| `scripts/run_isaac_shapenet_touch_charts_ur_ik_experiment.ps1` | UR IK 实验版 launcher。 |
| `scripts/collect_tactile_samples.py` | Tactile Lab 环境里的通用触觉样本采集脚本。 |
| `scripts/run_tactile_data_sampling.ps1` | 调用 `collect_tactile_samples.py` 的 PowerShell 入口。 |
| `scripts/run_tactile_lab.ps1` | 启动 Tactile Lab/Isaac Lab 任务的简短入口。 |

### touch model 训练和测试

| 文件 | 作用 |
| --- | --- |
| `scripts/touch_model/__init__.py` | touch model Python package 初始化文件。 |
| `scripts/touch_model/touch_model.py` | patch-only touch model 网络结构。包含触觉图像 CNN 编码器、ResBlock、点云输出头等。 |
| `scripts/train_touch_model.py` | 训练 touch model。读取 touch chart 数据，训练触觉图像到 local patch 点云的预测模型。 |
| `scripts/test_touch_model.py` | 测试 touch model，并生成 prediction vs ground truth 的 HTML/NPZ 结果。 |
| `scripts/run_train_touch_model.ps1` | touch model 训练 launcher。 |
| `scripts/run_test_touch_model.ps1` | touch model 测试 launcher。 |
| `scripts/run_touch_model_formal_large1000_after_collect.ps1` | 大样本采集完成后的正式训练/测试流程入口。 |

### 可视化脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/visualize_touch_charts_html.py` | 把 touch chart 数据生成 HTML 可视化。支持 original_touch、dashboard、sampling_process 等风格。 |
| `scripts/run_touch_charts_html_visualization.ps1` | 调用 `visualize_touch_charts_html.py` 的 PowerShell 入口。 |
| `scripts/visualize_touch_model_prediction_gallery.py` | 生成 touch model prediction vs GT 的 gallery 页面。 |

### Isaac Lab / RL 通用脚本

| 文件 | 作用 |
| --- | --- |
| `scripts/list_envs.py` | 列出当前 extension 注册的 Isaac Lab 任务环境。 |
| `scripts/random_agent.py` | 随机动作 agent，用于快速检查环境能否运行。 |
| `scripts/zero_agent.py` | 零动作 agent，用于快速检查环境能否运行。 |
| `scripts/rl_games/train.py` | 用 rl_games 训练 Isaac Lab 任务。 |
| `scripts/rl_games/play.py` | 用 rl_games checkpoint 回放/测试任务。 |
| `scripts/rsl_rl/cli_args.py` | rsl_rl 的命令行参数定义。 |
| `scripts/rsl_rl/train.py` | 用 rsl_rl 训练 Isaac Lab 任务。 |
| `scripts/rsl_rl/play.py` | 用 rsl_rl checkpoint 回放/测试任务。 |
| `scripts/sb3/train.py` | 用 Stable-Baselines3 训练 Isaac Lab 任务。 |
| `scripts/sb3/play.py` | 用 Stable-Baselines3 checkpoint 回放/测试任务。 |
| `scripts/skrl/train.py` | 用 skrl 训练 Isaac Lab 任务。 |
| `scripts/skrl/play.py` | 用 skrl checkpoint 回放/测试任务。 |

## `source/Tactile_Lab/` 文件说明

### Extension 配置

| 文件 | 作用 |
| --- | --- |
| `source/Tactile_Lab/pyproject.toml` | Python/Isaac Lab extension 项目配置。 |
| `source/Tactile_Lab/setup.py` | 可编辑安装入口，例如 `pip install -e source/Tactile_Lab`。 |
| `source/Tactile_Lab/config/extension.toml` | Omniverse/Isaac Sim extension 描述和加载配置。 |
| `source/Tactile_Lab/docs/CHANGELOG.rst` | extension 变更记录。 |
| `source/Tactile_Lab/Tactile_Lab/__init__.py` | Python package 初始化，并触发任务注册。 |
| `source/Tactile_Lab/Tactile_Lab/ui_extension_example.py` | Omniverse UI extension 示例。 |

### 触觉资产和参考深度

| 文件/目录 | 作用 |
| --- | --- |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/__init__.py` | 资产 package 初始化。 |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/reference_images/**/nodef_dep.npy` | 不同触觉传感器/分辨率下的无接触参考深度图。采集时用于把深度相机图转换为 deformation/tactile image。 |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/tactile_lab_assets/__init__.py` | 嵌套资产 package 初始化。 |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/tactile_lab_assets/robots/__init__.py` | 机器人资产 package 初始化。 |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/tactile_lab_assets/robots/tg3_ur5.py` | UR5 + TacTip 机器人资产配置。 |
| `source/Tactile_Lab/Tactile_Lab/tactile_lab_assets/tactile_lab_assets/robots/tg3_tactile_allegro_hand.py` | Allegro hand + tactile sensor 资产配置。 |

### 通用工具模块

| 文件 | 作用 |
| --- | --- |
| `source/Tactile_Lab/Tactile_Lab/utility/__init__.py` | utility package 初始化。 |
| `source/Tactile_Lab/Tactile_Lab/utility/setup_targets.py` | 任务/目标 setup 相关工具。 |
| `source/Tactile_Lab/Tactile_Lab/utility/tactile_sensor.py` | 触觉传感器封装和深度/触觉图处理工具。 |
| `source/Tactile_Lab/Tactile_Lab/utility/utils.py` | 通用辅助函数。 |

## Isaac Lab 任务目录说明

所有任务都在 `source/Tactile_Lab/Tactile_Lab/tasks/direct/` 下。每个任务目录通常包含：

| 文件模式 | 作用 |
| --- | --- |
| `__init__.py` | 注册任务，使 Isaac Lab 能通过 task name 找到环境。 |
| `*_env.py` | 任务环境实现，定义 step/reset、观测、奖励、终止条件等。 |
| `*_env_cfg.py` | 任务配置，定义场景、机器人、传感器、仿真参数、奖励权重等。 |
| `agents/rl_games_ppo_cfg.yaml` | rl_games PPO 训练配置。 |
| `agents/rsl_rl_ppo_cfg.py` | rsl_rl PPO 训练配置。 |
| `agents/skrl_ppo_cfg.yaml` | skrl PPO 训练配置。 |
| `agents/skrl_amp_cfg.yaml` | skrl AMP 训练配置。 |

具体任务：

| 目录 | 作用 |
| --- | --- |
| `tasks/direct/bipush/` | 双触觉/双点 push 任务环境。 |
| `tasks/direct/collect_depth_data/` | 深度/触觉数据采集任务环境。 |
| `tasks/direct/edge_follow/` | 边缘跟随任务环境，包含普通版和 distillation 版。 |
| `tasks/direct/hand_gait/` | 触觉手 gait 任务环境。 |
| `tasks/direct/hand_grasp/` | 触觉手抓取任务环境。 |
| `tasks/direct/obj_push/` | 物体推动任务环境。 |

## 安装和运行

在本机需要先准备 Isaac Lab/Isaac Sim 环境，并用对应 Python 安装 extension：

```powershell
python -m pip install -e .\source\Tactile_Lab
```

列出环境：

```powershell
python .\scripts\list_envs.py
```

启动当前 aligned GT 采集：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_large_touch_model_clone_collection_aligned_gt.ps1
```

训练 touch model：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_train_touch_model.ps1
```

测试 touch model：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\run_test_touch_model.ps1
```

## 输出文件在哪里

运行采集/训练后，输出通常写到：

```text
outputs/
```

这个目录包含 `.npy` 样本、`.pkl` 数据集、HTML 可视化、训练 checkpoint、日志等。为了让 GitHub 仓库保持轻量，`outputs/` 没有包含在当前备份里。
