# 项目架构

## 目标

本项目通过本地 Streamlit 界面接收 PDF，提取并切分文本，
使用嵌入模型建立向量索引，再基于检索到的文档片段回答用户问题。

## 模块边界

- `document.loader`：读取上传的 PDF，并保留文件名、页码等元数据。
- `document.splitter`：将文档内容划分为适合嵌入和检索的文本块。
- `vectorstore.embeddings`：通过官方 NVIDIA AI Endpoints 集成封装远程嵌入模型调用。
- `vectorstore.store`：封装 Chroma 向量库的创建与持久化。
- `vectorstore.retriever`：配置相似度检索参数并返回相关文档片段。
- `llm.client`：封装 NVIDIA Nemotron 模型客户端。
- `prompts.templates`：集中维护系统提示和问答提示模板。
- `qa.pipeline`：组织检索、上下文组合、模型调用和来源结果。
- `ui.app`：实现 PDF 上传、聊天记录和检索来源展示。
- `config.settings`：集中读取环境变量和应用配置。

## 基础数据流

```text
PDF 上传
  -> 文档加载
  -> 文本切分
  -> 文本嵌入
  -> Chroma 持久化

用户问题
  -> 问题嵌入与相似度检索
  -> 上下文组合
  -> Nemotron 生成回答
  -> 来源片段与答案返回 UI
```
