# ALFWorld -- WebStudio 环境搭建指南

既然您都看到这了，说明您已经知道 WebStudio 是什么，并且需要在 WebStudio 环境搭建 ALFWorld

> ⚠️ 前提：
> 1. 您使用了`ailab-910b-pt_2.9.0-sgl_0.5.9-vllm_0.18.0-cann_9.0.0-py_3.11:25.6.1.401`镜像启动 WebStudio
> 2. 脚本中标注 `[webstudio]` 的步骤，是因为 WebStudio 环境直接运行`alfworld-download`下载非常不稳定，可以从华山平台复制离线数据/包，具体地址私聊获取
> 3. 虽然镜像内默认conda环境名是`PyTorch-2.7.1`（基础镜像自带），但实际已经升级`torch==2.9.0`，（历史问题没有修改，此处不赘述原因）
---

## 1. 修改初始化脚本 `init_mtp.sh`

备份原文件并删除其中自动激活 conda 环境的行（避免环境冲突）。

```bash
sed -i.bak '/conda activate \$ENV_NAME/d' /home/ma-user/init_mtp.sh
```

## 2. 创建 conda 环境

基于已有的 `PyTorch-2.7.1` 环境 clone 名为 `alfworld` 的新环境。

```bash
conda create --name alfworld --clone PyTorch-2.7.1
conda activate alfworld
```


## 3. 安装依赖

### 3.1 TextWorld 问题

> - `textworld[pddl]>=1.6.1`是 ALFWorld 的核心依赖（见 `requirements.txt`），文本环境通过 textworld 加载每个游戏的 `game.tw-pddl` 文件（见 [`alfworld/agents/environment/alfred_tw_env.py`](https://github.com/alfworld/alfworld/blob/master/alfworld/agents/environment/alfred_tw_env.py)）。
> - 官方 textworld 依赖某个版本非常旧的 Inform 7 组件（见 [`TextWorld/setup.sh`](https://github.com/microsoft/TextWorld/blob/main/setup.sh#L21)）**并不支持`aarch64`**。

本项目基于2026/06/26的新版 inform 开源项目，在当前`aarch64`环境进行重新编译，并对 textworld 进行了新版 inform 适配。

新版 textworld 项目位于 https://github.com/imhmhm/TextWorld/tree/zhanghengheng/aarch64-inform_10.2.0

提供了基于源码安装和 whl 安装两种方式
- whl 位于 [`TextWorld/releases/textworld-1.7.0-cp311-cp311-linux_aarch64.whl`](https://github.com/imhmhm/TextWorld/tree/zhanghengheng/aarch64-inform_10.2.0/releases)
- 源码安装详见[`TextWorld
/BUILD_FROM_SOURCE.md`](https://github.com/imhmhm/TextWorld/blob/zhanghengheng/aarch64-inform_10.2.0/BUILD_FROM_SOURCE.md)


### 3.2 TextWorld 安装

以 whl 安装为例，通过上述目录获取 whl 文件

```bash
pip install textworld-1.7.0-cp311-cp311-linux_aarch64.whl
```

### 3.3 ALFWorld 源码安装

```bash
git clone -b webstudio https://github.com/imhmhm/alfworld.git
cd alfworld
pip install -e .
```
> **Note:** 如果是 LLM 场景不需要 \[full] 或 \[vis]

## 4. 离线数据加载

> `[webstudio]` `alfworld-download` 会从 github 链接下载 ALFWorld 数据，在 WebStudio 环境下载非常不稳定。
> 可以从华山平台复制相关数据，并使用`alfworld-extract-offline`进行解压等操作。


华山平台的 ALFWorld 离线数据内容如下：

| 文件 | 必需 | 来源 |
|---|---|---|
| `json_2.1.1_json.zip` | 是 | `alfworld-download` JSON_FILES_URL |
| `json_2.1.1_pddl.zip` | 是 | `alfworld-download` PDDL_FILES_URL |
| `json_2.1.3_tw-pddl.zip` | 是 | `alfworld-download` TW_PDDL_FILES_URL |
| `mrcnn_alfred_objects_sep13_004.pth` | 是 | `alfworld-download` MRCNN_URL |
| `pretrained_checkpoints.zip` | `--extra` | `alfworld-download` CHECKPOINTS_URL |
| `seq2seq_data.zip` | `--extra` | `alfworld-download` SEQ2SEQ_DATA_URL |

将环境变量 `ALFWORLD_OFFLINE_DATA` 指向该数据的本地目录。
将环境变量 `ALFWORLD_DATA` 设置为自定义的目录。（未设置时默认为 `~/.cache/alfworld`（见 `alfworld/info.py`）。也可以直接通过 `--data-dir` 指定）

```bash
export ALFWORLD_OFFLINE_DATA=/path/to/offline_data
export ALFWORLD_DATA=/path/to/data_dir
alfworld-extract-offline          # 加 --extra 可同时解压 checkpoints + seq2seq；加 -f 可覆盖
```

> 执行后的 `$ALFWORLD_DATA` 数据目录结构如下：
> ```
>    $ALFWORLD_DATA/
>    ├── json_2.1.1/{train,valid_seen,valid_unseen}/<problem>/trial_*/
>    │       ├── traj_data.json          (来自 json_2.1.1_json.zip)
>    │       ├── initial_state.pddl      (来自 json_2.1.1_pddl.zip)
>    │       └── game.tw-pddl            (来自 json_2.1.3_tw-pddl.zip)
>    ├── logic/{alfred.pddl,alfred.twl2}        (从 alfworld 包复制)
>    ├── detectors/mrcnn_alfred_objects_sep13_004.pth
>    ├── agents/pretrained_checkpoints/*.pt      # 仅在带 --extra 时
>    └── seq2seq_data/*.json                    # 仅在带 --extra 时
>  ```


#### 另：mrcnn 文件名（仅视觉相关）

`configs/base_config.yaml` 和 `eval_config.yaml` 引用的是 `$ALFWORLD_DATA/detectors/mrcnn.pth`，但实际文件名为 `mrcnn_alfred_objects_sep13_004.pth`。这只会影响视觉 / MaskRCNN 控制器路径；**纯文本使用不受影响**。视觉用户可自行修改：

```bash
cp $ALFWORLD_DATA/detectors/mrcnn_alfred_objects_sep13_004.pth \
   $ALFWORLD_DATA/detectors/mrcnn.pth
```

