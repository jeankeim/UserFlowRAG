import os
from typing import List, Dict
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from pypdf import PdfReader
import docx2txt
# from pptx2text import Pptx2Text
import json
from src.utils.logger import setup_logger
from langchain_community.document_loaders import DirectoryLoader, TextLoader


class DocumentProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.logger = setup_logger('DocumentProcessor',log_file='../logs/document_processor.log')
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config['retrieval']['chunk_size'],
            chunk_overlap=config['retrieval']['chunk_overlap']
        )


    def load_documents(self) -> List:
        """加载文档"""
        loader = DirectoryLoader(
            # self.docs_dir,
            self.config['paths']['docs_dir'],
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        return documents

    def process_documents(self, documents: List) -> List:
        """处理文档"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents(documents)
        return texts


    def process_file(self, file_path: str) -> List[Document]:
        """处理单个文件"""
        try:
            extension = os.path.splitext(file_path)[1].lower()
            content = self._extract_content(file_path, extension)
            
            if not content:
                return []

            metadata = {
                'source': file_path,
                'file_type': extension,
                'timestamp': os.path.getmtime(file_path)
            }

            doc = Document(page_content=content, metadata=metadata)
            return self.text_splitter.split_documents([doc])

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return []

    def _extract_content(self, file_path: str, extension: str) -> str:
        """根据文件类型提取内容"""
        try:
            if extension == '.pdf':
                return self._extract_pdf(file_path)
            elif extension == '.docx':
                return self._extract_docx(file_path)
            elif extension == '.pptx':
                return self._extract_pptx(file_path)
            elif extension == '.txt':
                return self._extract_txt(file_path)
            elif extension == '.json':
                return self._extract_json(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {extension}")
                return ""
        except Exception as e:
            self.logger.error(f"Error extracting content from {file_path}: {e}")
            return ""

    # def _extract_pdf(self, file_path: str) -> str:
    #     reader = PdfReader(file_path)
    #     text = ""
    #     for page in reader.pages:
    #         text += page.extract_text() + "\n"
    #     return text

    def _extract_docx(self, file_path: str) -> str:
        return docx2txt.process(file_path)

    # def _extract_pptx(self, file_path: str) -> str:
    #     return Pptx2Text.extract_text(file_path)

    def _extract_txt(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _extract_json(self, file_path: str) -> str:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return json.dumps(data, ensure_ascii=False, indent=2)
