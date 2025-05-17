from typing import List, Optional, Dict
from langchain.schema import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
import os
import hashlib
import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

class DocumentLoader:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._init_jieba()

    def _init_jieba(self):
        """初始化中文分词器"""
        try:
            import jieba
            jieba.initialize()
            if self.config.get('custom_dict_path'):
                jieba.load_userdict(self.config['custom_dict_path'])
        except ImportError:
            pass

    def load_from_directory(self, 
                          doc_path: str,
                          progress_callback: Optional[callable] = None) -> List[Document]:
        """从目录加载文档并自动处理中文分词"""
        if not os.path.exists(doc_path):
            raise FileNotFoundError(f"文档目录不存在: {doc_path}")

        loader = DirectoryLoader(
            path=doc_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=False
        )
        raw_docs = loader.load()
        
        processed_docs = []
        with ThreadPoolExecutor() as executor:
            futures = []
            for doc in raw_docs:
                futures.append(executor.submit(
                    self._process_single_document,
                    doc
                ))
            
            for i, future in enumerate(tqdm(futures, desc="处理文档")):
                processed_doc = future.result()
                processed_docs.append(processed_doc)
                if progress_callback:
                    progress_callback(i+1, len(raw_docs))
        
        return processed_docs

    def _process_single_document(self, doc: Document) -> Document:
        """处理单个文档"""
        # 生成文档指纹
        content = doc.page_content
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        # 中文分词处理
        if self.config.get('use_jieba', False):
            content = self._chinese_tokenize(content)
        
        # 更新元数据
        doc.page_content = content
        doc.metadata.update({
            'doc_id': doc_id,
            'source_file': os.path.basename(doc.metadata['source']),
            'version': 1,
            'last_updated': datetime.datetime.now().isoformat(),
            'content_length': len(content)
        })
        return doc

    def _chinese_tokenize(self, text: str) -> str:
        """中文分词处理"""
        try:
            import jieba
            return " ".join(jieba.lcut(text))
        except ImportError:
            return text

    @staticmethod 
    def load_single_file(file_path: str) -> Document:
        """加载单个文档文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        loader = TextLoader(file_path)
        doc = loader.load()[0]
        doc_id = hashlib.md5(doc.page_content.encode()).hexdigest()
        doc.metadata.update({
            'doc_id': doc_id,
            'version': 1,
            'last_updated': datetime.datetime.now().isoformat()
        })
        return doc
