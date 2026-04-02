# 纯本地单机图片特征检索（基于 Chinese-CLIP）

本项目实现了完全本地运行的图片检索系统，严格拆分为两个独立功能：

1. 生成向量库：读取本地图片 -> 提取向量 -> 保存到你指定的本地目录
2. 检索向量库：加载你指定的本地向量库 -> 输入文字 -> 返回匹配图片路径

## 环境要求

- Python 3.10+
- 本地模型目录：`mod/`，里面包含 `config.json`、`pytorch_model.bin`、`preprocessor_config.json`、`vocab.txt`

安装依赖：

```bash
conda create -n rag-image python=3.11 -y
conda activate rag-image
pip install -r requirements.txt
```

## 统一入口（交互模式）

```bash
python app.py
```

启动后可选择：

1. 生成向量库：输入图片目录（递归扫描）和向量库存储目录，终端显示提取进度
2. 检索向量库：输入向量库目录和关键词，返回按相似度排序的图片路径

说明：

- 当前代码默认直接加载 `mod/` 目录下的 Chinese-CLIP 模型

## 功能 1：生成本地向量库（独立运行）

```bash
python build_index.py \
  --images-dir /你的本地图片目录 \
  --db-dir /你指定的向量库目录
```

常用可选参数：

```bash
--model-path /你的模型目录/mod
--device cpu
--batch-size 32
--no-recursive
```

执行完成后，`--db-dir` 下会生成 `index.db`：SQLite 向量库（默认）。

SQLite 中包含：

- `meta` 表：`created_at`、`model_name`、`model_path`、`embedding_dim`、`count` 等
- `items` 表：`path`、`width`、`height`、`embedding_dim`、`embedding`(BLOB)

兼容说明：

- 检索时会优先读取 `index.db`
- 如果没有 `index.db`，会自动回退读取旧格式 `vectors.npy + paths.json + meta.json`

## 功能 2：检索本地向量库（独立运行）

```bash
python search_index.py \
  --db-dir /你指定的向量库目录 \
  --query "一个人坐在电脑前面" \
  --top-k 10
```

常用可选参数：

```bash
--model-path /你的模型目录/mod
--device cpu
```

输出为按相似度排序的本地图片路径。

## 最简工作流

1. 生成库：选图片目录 + 选向量库存储目录 -> 运行 `build_index.py`
2. 检索：指定向量库目录 + 输入文本描述 -> 运行 `search_index.py`

## 目录说明

- `build_index.py`：功能 1（建库）入口
- `search_index.py`：功能 2（检索）入口
- `image_rag/modeling.py`：Chinese-CLIP 模型加载与编码
- `image_rag/storage.py`：本地向量库存取和图片收集
- `mod/`：本地模型目录
