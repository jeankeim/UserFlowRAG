import os
from typing import Dict, Any, Optional, AsyncGenerator
import asyncio
import math
from venv import logger

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from src.core.retrieval import EnhancedRetrieval
from src.core.document_loader import DocumentLoader
from src.core.evaluator import EnhancedEvaluator
from src.core.cache_manager import CacheManager
from src.core.agent_manager import AgentManager
from src.utils.monitor import RAGMonitor
import yaml

class EnhancedRAGSystem:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        # 预加载模型
        # from sentence_transformers import SentenceTransformer
        # self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2', 
        #     cache_folder=os.path.join(self.config['paths']['models_dir'], 'all-MiniLM-L6-v2'))
        # self.cross_encoder = SentenceTransformer('ms-marco-MiniLM-L6-v2',
        #     cache_folder=os.path.join(self.config['paths']['models_dir'], 'ms-marco-MiniLM-L6-v2'))
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线
        os.environ["HF_HUB_OFFLINE"] = "1"       # 双重保险
        model_dir = "/Users/xieming/Desktop/rag_project/models"
        print(f"Loading embedding model  all-MiniLM-L6-v2 from local cache: {model_dir}") 
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # 标准名称
            cache_folder=model_dir,  # 关键：指定模型存储的根目录
            model_kwargs={"device": "cpu"}        )
        # Use full local path to model
        print(f"Loading cross encoder model  ms-marco-MiniLM-L6-v2 from local cache: {model_dir}") 
        cross_encoder_path = os.path.join(model_dir, "models--cross-encoder--ms-marco-MiniLM-L6-v2", "snapshots", "739bce82df32cacea8ff0edf73ab49ae315e5a5f")
        self.cross_encoder = CrossEncoder(
            cross_encoder_path
        )

        self.document_loader = DocumentLoader(self.config.get('text_processing', {}))
        self.retrieval = EnhancedRetrieval(self.config, 
            embedding_model=self.embedding_model,
            cross_encoder=self.cross_encoder)
        self.evaluator = EnhancedEvaluator()
        self.cache_manager = CacheManager(self.config)
        self.agent_manager = AgentManager(
            "config/agents.yaml",
            "config/tasks.yaml"
        )
        self.monitor = RAGMonitor(self.config['paths']['logs_dir'])
        self.monitor.start_monitoring()

    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    async def initialize(self, load_documents: bool = False):
        """初始化系统
        Args:
            load_documents: 是否立即加载文档
        """
        try:
            # self.monitor.start_monitoring()
            # 初始化向量库(不加载文档)
            self.retrieval.initialize_vector_store()
            
            if load_documents:
                # 按需加载文档
                docs = self.document_loader.load_from_directory(
                    self.config['paths']['docs_dir']
                )
                processed_docs = docs  # 文档加载器已经完成处理
                self.retrieval.add_documents(load_documents,processed_docs)
        
                
        except Exception as e:
            self.monitor.log_error(f"Initialization error: {e}")
            raise


    async def process_query(self, question: str) -> AsyncGenerator[Dict[str, Any], None]:
        """处理查询"""
        start_time = asyncio.get_event_loop().time()
        logger.info(f"开始处理查询: {question}")
        
        # 检查缓存
        cached_result = self.cache_manager.get(question)
        if cached_result is not None:
            logger.info("问题命中缓存，直接给出答案！")
            self.monitor.log_cache_hit()
            cached_result['metrics'] = self.monitor.get_metrics()
            yield cached_result
            return
        
        logger.info("问题未命中缓存，结果生成中！")

        # 检索相关文档
        relevant_docs = self.retrieval.hybrid_search(question)
        logger.info(f"检索到 {len(relevant_docs)} 个相关文档")
        
        # 初始化空答案
        answer = ""
        logger.info("开始生成回答...")
        
        # 构建上下文提示（考虑长度限制）
        context_parts = []
        total_length = 0
        max_context_length = self.config.get('retrieval', {}).get('max_context_length', 4000)
        
        for i, doc in enumerate(relevant_docs):
            doc_content = f"文档 {i+1}:\n{doc.page_content}"
            if total_length + len(doc_content) > max_context_length:
                break
            context_parts.append(doc_content)
            total_length += len(doc_content)
        
        context = "\n\n".join(context_parts)

        
        # 构建优化后的prompt
        if context_parts:
            prompt = f"""请基于以下提供的上下文信息回答问题（共{len(context_parts)}个相关文档）：
{context}

请按照以下要求回答：
1. 优先基于上述上下文内容回答
2. 如果上下文没有提供足够信息，可以结合你的常识补充回答
3. 引用具体文档编号支持你的回答
4. 保持回答简洁客观

问题：{question}

请按照以下格式回答：
【回答】
[你的回答]

【依据】
引用相关文档编号和内容片段（如适用）"""
        else:
            prompt = f"""请回答以下问题，可以自由使用你的知识：

问题：{question}

请按照以下格式回答：
【回答】
[你的回答]

【补充说明】
[任何额外的解释或说明]"""

        # 使用新的stream_llm_response接口
        async for response in self.agent_manager.generate_stream("researcher", prompt):
            # if 'output' in response:
            #     yield {
            #         'question': question,
            #         'answer': response['output'],
            #         'metrics': self.monitor.get_metrics(),
            #         'relevant_docs': [{
            #             'page_content': doc.page_content,
            #             'metadata': doc.metadata
            #         } for doc in relevant_docs] if relevant_docs else [],
            #         'is_final': False
            #     }
            # elif 'token' in response:
            #     yield {
            #         'question': question,
            #         'answer': response['token'],
            #         'metrics': self.monitor.get_metrics(),
            #         'relevant_docs': [{
            #             'page_content': doc.page_content,
            #             'metadata': doc.metadata
            #         } for doc in relevant_docs] if relevant_docs else [],
            #         'is_final': False
            #     }
            yield response
        
        # 最终结果
        # yield {
        #     'question': question,
        #     'answer': answer,
        #     'metrics': self.monitor.get_metrics(),
        #     'relevant_docs': [{
        #         'page_content': doc.page_content,
        #         'metadata': doc.metadata
        #     } for doc in relevant_docs] if relevant_docs else [],
        #     'is_final': True
        # }
            


    async def process_document(self, file) -> Dict[str, Any]:
        """处理上传的文档"""
        try:
            # 保存临时文件
            temp_dir = os.path.join(self.config['paths']['docs_dir'], 'uploads')
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, file.filename)
            
            with open(temp_path, 'wb') as f:
                f.write(await file.read())
            
            # 使用load_single_file方法处理文档
            document = DocumentLoader.load_single_file(temp_path)
            self.retrieval.add_documents(True, [document])
            
            return {
                'status': 'success',
                'document_id': file.filename,
                'metadata': {
                    'processed_chunks': 1,
                    'message': 'Document processed successfully'
                }
            }
        except Exception as e:
            self.monitor.log_error(str(e))
            raise




    async def load_documents(self, doc_path: Optional[str] = None):
        """按需加载文档"""
        doc_path = doc_path or self.config['paths']['docs_dir']
        processed_docs = self.document_loader.load_from_directory(doc_path)
        self.retrieval.add_documents(processed_docs)
        return len(processed_docs)
