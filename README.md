# Interactive PDF Reader

一个面向学习的交互式 PDF 智能问答项目骨架，使用 Streamlit 和 LangChain
展示 PDF 文档加载、文本切分、向量检索与 AI 问答的完整流程。

## 技术栈

- PDF 文本加载与切分
- NVIDIA 远程嵌入模型 `nvidia/nemotron-3-embed-1b`
- 本地持久化向量数据库 Chroma
- NVIDIA 聊天模型 `nvidia/nemotron-3.5-lightning-30b-a3b`
- Streamlit 聊天界面
- LangChain 检索增强问答流水线
- uv 依赖与虚拟环境管理

## 目录结构

```text
interactive-pdf-reader/
├── .streamlit/          # Streamlit 本地配置
├── data/
│   ├── uploads/         # 上传的 PDF
│   └── chroma/          # Chroma 持久化数据
├── docs/                # 架构与学习文档
├── scripts/             # 辅助脚本
├── src/pdf_reader/
│   ├── config/          # 环境变量与应用配置
│   ├── document/        # PDF 加载和文本切分
│   ├── llm/             # NVIDIA 模型客户端
│   ├── prompts/         # 提示词模板
│   ├── qa/              # 检索增强问答流水线
│   ├── ui/              # Streamlit 界面
│   └── vectorstore/     # 嵌入、向量库与检索器
└── tests/               # 按模块划分的测试
```

## 本地配置

前置条件：

- Python 3.12
- uv
- NVIDIA API Key

复制 `.env.example` 为 `.env`，并填写 NVIDIA API Key：

```bash
cp .env.example .env
```

模型名称、向量库路径和上传目录的占位配置也可在 `.env` 中调整。
请勿将包含真实密钥的 `.env` 提交到版本库。

主要环境变量：

| 变量名 | 默认值 | 用途 |
| --- | --- | --- |
| `NVIDIA_API_KEY` | 空 | NVIDIA API 认证 |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | NVIDIA API 基础地址 |
| `NVIDIA_CHAT_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | 聊天模型 |
| `NVIDIA_EMBEDDING_MODEL` | `nvidia/nemotron-3-embed-1b` | 文本嵌入模型 |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | Chroma 持久化目录 |
| `UPLOAD_DIRECTORY` | `./data/uploads` | 上传文件目录 |
| `CHUNK_SIZE` | `1000` | 文本切分块大小 |
| `CHUNK_OVERLAP` | `200` | 相邻文本块重叠大小 |
| `RETRIEVAL_K` | `4` | 每次检索的文档片段数 |

## 环境初始化

```bash
uv sync
```

同步完成后，可运行项目入口：

```bash
uv run pdf-reader
# 或
uv run streamlit run src/pdf_reader/ui/app.py
```

开发检查：

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

根目录 [main.py](main.py) 目前保留为 uv 项目入口占位；
安装后的 `pdf-reader` 命令将调用 [src/pdf_reader/main.py](src/pdf_reader/main.py)。
业务模块仍为空，尚未实现 PDF 阅读和问答功能。

## 当前状态

此仓库目前只包含项目骨架、模块占位、配置模板和文档入口；
尚未实现 PDF 加载、向量检索或问答业务逻辑。