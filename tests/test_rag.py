from src.core.retrieval import RetrievalSystem
from src.core.document_processor import DocumentProcessor

class TestRAGSystem:
    def test_retrieval(self):
        """Test basic document retrieval functionality"""
        retrieval = RetrievalSystem()
        results = retrieval.query("test query")
        assert isinstance(results, list)
        
    def test_document_processing(self):
        """Test document processing pipeline"""
        processor = DocumentProcessor()
        processed = processor.process("Sample document text")
        assert "processed_text" in processed
        assert "embeddings" in processed

    def test_integration(self):
        """Test end-to-end RAG workflow"""
        processor = DocumentProcessor()
        retrieval = RetrievalSystem()
        
        doc = processor.process("Test document")
        results = retrieval.query("test query")
        
        assert len(results) > 0
