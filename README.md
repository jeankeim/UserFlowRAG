# Enhanced RAG System

一个基于检索增强生成(RAG)技术的增强问答系统，提供文档上传、自然语言查询和智能回答功能。

## 功能特性

- 支持多种文档格式上传(PDF/TXT/DOCX)
- 基于BM25和向量检索的混合检索
- 流式响应API
- Gradio交互式前端界面
- 可配置的授权令牌
- 详细的日志记录

## 技术栈

### 后端
- Python 3.9+
- FastAPI (REST API)
- LangChain (RAG框架)
- ChromaDB (向量数据库)
- BM25 (传统检索)
- Sentence Transformers (嵌入模型)

### 前端
- Gradio (Web界面)
- JavaScript/HTML/CSS

### 数据处理
- PyPDF2 (PDF处理)
- docx2txt (DOCX处理)

## 安装指南

1. 克隆仓库：
```bash
git clone https://your-repository-url.git
cd rag_project
```

2. 创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate    # Windows
```

3. 安装依赖：
```bash
pip install -r requirements.txt
```

4. 下载模型：
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## 配置说明

编辑`config/config.yaml`：
```yaml
auth:
  tokens: ["your_api_token"]  # 授权令牌

storage:
  document_dir: "data/documents"  # 文档存储目录
  vector_db: "data/vector_store"  # 向量数据库路径

models:
  embedding: "all-MiniLM-L6-v2"  # 嵌入模型
  cross_encoder: "cross-encoder/ms-marco-MiniLM-L6-v2"  # 重排序模型
```

## 使用说明

### 启动后端API
```bash
python src/main.py
```

### 启动前端界面
```bash
python gradio_app.py
```

### API端点

- `POST /query` - 处理自然语言查询
- `POST /documents/upload` - 上传文档

## 开发指南

### 项目结构
```
.
├── config/            # 配置文件
├── data/              # 数据存储
│   ├── documents/     # 上传的文档
│   └── vector_store/  # 向量数据库
├── src/               # 源代码
│   ├── api/           # API端点
│   ├── core/          # 核心逻辑
│   └── utils/         # 工具类
├── tests/             # 单元测试
└── models/            # 模型缓存
```

### 主要组件

- `rag_system.py` - RAG系统核心实现
- `agent_manager.py` - 代理管理
- `document_processor.py` - 文档处理
- `retrieval.py` - 检索逻辑

## 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/your-feature`)
3. 提交更改 (`git commit -am 'Add some feature'`)
4. 推送到分支 (`git push origin feature/your-feature`)
5. 创建Pull Request

## 许可证

MIT License
