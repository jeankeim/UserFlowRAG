from typing import List, Dict, Set, Optional, Callable
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

import chromadb
from chromadb.config import Settings
import os, numpy as np 
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


class EnhancedRetrieval:
    """
    RAG检索增强类，支持混合搜索(向量+BM25)
    
    BM25初始化流程:
    1. 构造函数中初始化为None
    2. initialize_vector_store()尝试从文件加载已有索引
    3. 如果加载失败，在add_documents()时通过_update_bm25_index()创建新索引
    4. 索引会定期保存到文件(_save_bm25_index)
    """
    def __init__(self, config: dict,embedding_model,cross_encoder):
        self.config = config
        from .document_loader import DocumentLoader
        self.document_loader = DocumentLoader(config.get('text_processing', {}))
        # Configure for local model loading with official name
        # 指定模型所在父目录（不是具体模型路径）
        os.environ["TRANSFORMERS_OFFLINE"] = "1"  # 强制离线
        os.environ["HF_HUB_OFFLINE"] = "1"       # 双重保险
        model_dir = "/Users/xieming/Desktop/rag_project/models"
        # print(f"Loading embedding model  all-MiniLM-L6-v2 from local cache: {model_dir}") 
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name="sentence-transformers/all-MiniLM-L6-v2",  # 标准名称
        #     cache_folder=model_dir,  # 关键：指定模型存储的根目录
        #     model_kwargs={"device": "cpu"}        )
        self.embeddings=embedding_model
        # Use full local path to model
        # print(f"Loading cross encoder model  ms-marco-MiniLM-L6-v2 from local cache: {model_dir}") 
        # cross_encoder_path = os.path.join(model_dir, "models--cross-encoder--ms-marco-MiniLM-L6-v2", "snapshots", "739bce82df32cacea8ff0edf73ab49ae315e5a5f")
        # self.cross_encoder = CrossEncoder(
        #     cross_encoder_path
        # )
        self.cross_encoder=cross_encoder
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['retrieval'].get('chunk_size', 50), 
            chunk_overlap=config['retrieval'].get('chunk_overlap', 10),
            length_function=len,
            is_separator_regex=False,
        )
        self.vector_store = None
        self.documents = []
        self.bm25 = None  # 延迟初始化直到有文档


    def _save_bm25_index(self):
        """保存BM25索引到文件"""
        if not self.bm25:
            return
            
        index_path = os.path.join(self.config['paths']['vector_store'], 'bm25_index.pkl')
        try:
            import pickle
            with open(index_path, 'wb') as f:
                pickle.dump({
                    'bm25': self.bm25,
                    'documents': self.documents
                }, f)
            print(f"BM25索引已保存到 {index_path}")
        except Exception as e:
            print(f"保存BM25索引失败: {str(e)}")

    def _load_bm25_index(self) -> bool:
        """从文件加载BM25索引"""
        index_path = os.path.join(self.config['paths']['vector_store'], 'bm25_index.pkl')
        if not os.path.exists(index_path):
            return False
            
        try:
            import pickle
            with open(index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data['bm25']
                self.documents = data['documents']
            print(f"从 {index_path} 加载BM25索引成功")
            return True
        except Exception as e:
            print(f"加载BM25索引失败: {str(e)}")
            return False

    def initialize_vector_store(self):
        """初始化向量库连接和BM25索引"""
        os.makedirs(self.config['paths']['vector_store'], exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.config['paths']['vector_store'],
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 创建或获取空集合
        self.vector_store = Chroma(
            persist_directory=self.config['paths']['vector_store'],
            embedding_function=self.embeddings,
            client=self.client,
            collection_name="rag_collection",
            collection_metadata={"hnsw:space": "cosine"}  
        )
        
        # 尝试加载已有BM25索引，失败则初始化空索引
        if not self._load_bm25_index():
            self.documents = []
            self.bm25 = None


    def similarity_search(self, query: str) -> List[Document]:
        """带分数过滤的相似度搜索"""
        query_embedding = self.embeddings.embed_query(query)
        print(f"Query embedding sample: {query_embedding[:5]}")  # 打印前5维向量
        
        threshold = self.config['retrieval'].get('similarity_threshold', 0.6)
        results = self.vector_store.similarity_search_by_vector_with_relevance_scores(
            query_embedding,
            k=self.config['retrieval']['top_k']
        )
        
        # 打印详细分数信息并验证相似度计算
        print(f"Raw similarity scores:")
        for i, (doc, score) in enumerate(results):
            print(f"[{i}] Score: {score:.4f} | Content: {doc.page_content[:100]}...")
            
            # 获取文档向量并手动计算余弦相似度
            doc_embedding = self.embeddings.embed_query(doc.page_content)
            dot_product = np.dot(query_embedding, doc_embedding)
            query_norm = np.linalg.norm(query_embedding)
            doc_norm = np.linalg.norm(doc_embedding)
            manual_score = dot_product / (query_norm * doc_norm)
            print(f"  Manual cosine: {manual_score:.4f} | "
                  f"Dot: {dot_product:.4f} | "
                  f"Query norm: {query_norm:.4f} | "
                  f"Doc norm: {doc_norm:.4f}")
            
        results = [doc for doc, score in results if score <=(threshold)]
        return results



    def hybrid_search(self, query: str, top_k: int = 2) -> List[Document]:
        threshold = self.config['retrieval'].get('similarity_threshold', 0.7)
        """混合搜索策略"""
        if self.config['retrieval']['hybrid_search_enabled']:
            # 向量搜索（带分数）
            query_embedding=self.embeddings.embed_query(query)
            vector_results = self.vector_store.similarity_search_by_vector_with_relevance_scores(query_embedding, k=4)
            vector_results = [(doc, score) for doc, score in vector_results if score <= threshold]
            # for i, vector_result in enumerate(vector_results):
            #     print(f"[{i+1}] Score: {vector_result[1]:.4f} | Content: {vector_result[0]}...\n")
            
            # BM25搜索
            bm25_results = []
            if self.bm25 is not None and query.strip():
                try:
                    import jieba
                    # 中文分词处理
                    tokenized_query = list(jieba.cut(query))
                    if not tokenized_query:
                        return []  # 空查询直接返回
                        
                    # 获取BM25分数
                    bm25_scores = self.bm25.get_scores(tokenized_query)
                    if not len(bm25_scores):
                        return []  # 无分数返回
                        
                    # 归一化分数到[0,1]范围
                    max_score = max(bm25_scores) or 1  # 避免除零
                    normalized_scores = [score/max_score for score in bm25_scores]
                    
                    # 获取top_k*2个最佳结果(扩大范围避免内容重复)
                    top_indices = np.argsort(normalized_scores)[-top_k*2:][::-1]
                    # 过滤越界索引并获取文档
                    bm25_results = [
                        self.documents[i] 
                        for i in top_indices 
                        if i < len(self.documents)
                    ]
                    
                    # 打印搜索结果(调试用)
                    # print("BM25搜索结果(前{}个):".format(min(top_k, len(bm25_results))))
                    # for i, doc in enumerate(bm25_results[:top_k]):
                    #     print(f"[{i+1}] Score: {normalized_scores[top_indices[i]]:.4f} | Content: {doc.page_content[:50]}...\n")
                        
                except Exception as e:
                    print(f"BM25搜索出错: {str(e)}")


            # 直接合并结果并去重
            seen_contents = set()
            final_results = []
            
            # 添加向量搜索结果
            for doc, score in vector_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    final_results.append((doc, score))
            
            # 添加BM25搜索结果
            for doc in bm25_results:
                if doc.page_content not in seen_contents:
                    seen_contents.add(doc.page_content)
                    score = normalized_scores[self.documents.index(doc)] if doc in self.documents else 0
                    final_results.append((doc, score))
            
            # 按分数排序并取top_k
            final_results.sort(key=lambda x: x[1], reverse=True)
            final_results = [doc for doc, _ in final_results[:top_k]]
            
            if not final_results:
                return []
                
            # print("Top {} 搜索结果:".format(top_k))
            # for i, doc in enumerate(final_results):
            #     print(f"[{i+1}] Content: {doc.page_content[:100]}...")

            # 使用Cross-Encoder重排序
            if self.config['retrieval']['reranking_enabled']:
                return self._rerank_results(query, final_results, top_k)
            
            return final_results[:top_k]
        else:
            return self.similarity_search(query)


    def _rerank_results(
        self,
        query: str,
        documents: List[Document],
        top_k: int
    ) -> List[Document]:
        """使用Cross-Encoder重排序"""
        if not documents:
            return []
            
        try:
            if not documents or not query.strip():
                return documents[:top_k]
                
            pairs = [[query, doc.page_content] for doc in documents if doc.page_content]
            if not pairs:
                return documents[:top_k]
                
            scores = self.cross_encoder.predict(pairs)
            
            # 将文档和分数配对并排序，过滤分数<0.6的结果
            doc_score_pairs = list(zip(documents, scores))
            ranked_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
            filtered_results = [doc for doc, score in ranked_pairs if score >= 0.6]
            return filtered_results[:top_k] if filtered_results else documents[:top_k]
        except Exception as e:
            print(f"重排序出错: {str(e)}")
            return documents[:top_k]



    def _get_existing_doc_ids(self, documents: List[Document]) -> Set[str]:
        """获取已存在于向量库中的文档ID集合"""
        existing_ids = set()
        
        if not documents:
            return existing_ids
        
        try:
            # 批量查询已存在的ID（优化性能）
            batch_size = 10  # 每批查询100个
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                results = self.vector_store._collection.get(
                    ids=[doc.metadata["doc_id"] for doc in batch],
                    include=[]  # 不返回实际内容，只检查存在性
                )
                existing_ids.update(results["ids"])
        except Exception as e:
            print(f"检查存在文档时出错: {e}")
        
        return existing_ids


    def _update_bm25_index(self, documents: List[Document]):
        """更新BM25索引并处理中文分词"""
        if not documents:
            return
        
        # 中文分词处理（使用jieba）
        def chinese_tokenizer(text: str) -> List[str]:
            import jieba
            # 加载用户词典（如果有）
            if hasattr(self, 'user_dict') and self.user_dict:
                jieba.load_userdict(self.user_dict)
            # 精确模式分词
            return [word for word in jieba.lcut(text) if word.strip()]
        
        # 准备语料库
        corpus = []
        for doc in documents:
            if doc.page_content.strip():
                # 使用更精细的分词可以替换这里
                tokens = chinese_tokenizer(doc.page_content)
                corpus.append(tokens)
        
        # 初始化或更新BM25索引
        try:
            if not self.bm25:
                print(f"初始化BM25索引，文档数: {len(corpus)}")
                # 创建新的BM25Okapi索引
                # corpus: 分词后的文档列表，每个文档是词项列表
                self.bm25 = BM25Okapi(corpus)
            else:
                print(f"更新BM25索引，新增文档数: {len(corpus)}")
                # 向现有索引添加新文档
                # corpus: 分词后的新文档列表，每个文档是词项列表
                # 该方法会更新索引的文档频率和逆文档频率统计
                self.bm25.add_documents(corpus)
            
            # 打印BM25索引统计信息
            if hasattr(self.bm25, 'doc_len'):
                print(f"BM25索引统计 - 平均文档长度: {np.mean(self.bm25.doc_len):.2f}")
            print(f"BM25索引统计 - 文档总数: {len(self.documents)}")
                
        except Exception as e:
            print(f"更新BM25索引失败: {str(e)}")
            raise



    def add_documents(self, load_documents, documents: List[Document], batch_size: int = 100) -> int:
        """
        添加文档并自动处理去重和索引更新
        
        调用时机:
        1. 系统初始化时加载已有文档
        2. 用户手动添加新文档时
        3. 自动更新知识库时
        
        主要功能:
        1. 文档分块处理
        2. 去重检查
        3. 批量存入向量库
        4. 更新BM25索引
        
        Args:
            load_documents: 是否加载到向量库(True/False)
            documents: 要添加的文档列表
            batch_size: 批量处理大小
            
        Returns:
            实际添加的文档数量
        """
        if not documents:
            return 0
        
        # 1. 文档预处理和分块
        split_docs = self._chunk_documents(documents)
        
        # 2. 过滤已存在文档
        existing_ids = self._get_existing_doc_ids(split_docs)
        new_docs = [doc for doc in split_docs if doc.metadata["doc_id"] not in existing_ids]
        
        if not new_docs:
            return 0
            
        # 3. 分批存入向量库
        added_count = 0
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i:i + batch_size]
            try:
                if load_documents:
                    self.vector_store.add_documents(batch)
                # 4. 更新内存中的文档列表和BM25索引
                self.documents.extend(batch)
                self._update_bm25_index(batch)
                added_count += len(batch)
            except Exception as e:
                print(f"添加文档批次 {i//batch_size} 失败: {str(e)}")
                continue
                
        print(f"成功添加 {added_count}/{len(new_docs)} 个新文档块")
        # 保存更新后的BM25索引
        self._save_bm25_index()
        return added_count
        

    def _chunk_documents(self, documents):
        """内部方法：执行实际分块逻辑"""
        split_docs = []
        for doc in documents:
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata.update({
                    "chunk_id": i,
                    "doc_id": f"{doc.metadata.get('doc_id', id(doc))}_{i}"  # 唯一ID
                })
                split_docs.append(Document(page_content=chunk, metadata=metadata))
        return split_docs
