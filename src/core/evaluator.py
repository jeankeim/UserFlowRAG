from typing import Dict, List
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import numpy as np

class EnhancedEvaluator:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.metrics = {
            'relevance': 0.0,
            'coherence': 0.0,
            'completeness': 0.0,
            'consistency': 0.0,
            'source_coverage': 0.0
        }

    def evaluate_response(
        self,
        question: str,
        answer: str,
        context: List[Document]
    ) -> Dict[str, float]:
        # 计算各项指标
        self.metrics['relevance'] = self._evaluate_relevance(question, answer)
        self.metrics['coherence'] = self._evaluate_coherence(answer)
        self.metrics['completeness'] = self._evaluate_completeness(question, answer)
        self.metrics['consistency'] = self._evaluate_consistency(answer, context)
        self.metrics['source_coverage'] = self._evaluate_source_coverage(answer, context)

        # 计算总分
        self.metrics['overall_score'] = round(np.mean(list(self.metrics.values())),4)
        
        return self.metrics

    def _evaluate_relevance(self, question: str, answer: str) -> float:
        """评估答案与问题的相关性"""
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(question_embedding, answer_embedding)
        return round(float(similarity[0][0]), 4)

    def _evaluate_coherence(self, answer: str) -> float:
        """评估答案的连贯性"""
        sentences = answer.split('。')
        if len(sentences) < 2:
            return 1.0

        coherence_scores = []
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        for i in range(len(sentences)-1):
            similarity = util.pytorch_cos_sim(embeddings[i], embeddings[i+1])
            coherence_scores.append(float(similarity[0][0]))

        return round(np.mean(coherence_scores), 4)

    def _evaluate_completeness(self, question: str, answer: str) -> float:
        """评估答案的完整性"""
        # 基于答案长度和关键词覆盖率评估完整性
        key_terms = set(question.lower().split())
        answer_terms = set(answer.lower().split())
        coverage = len(key_terms.intersection(answer_terms)) / len(key_terms)
        
        length_score = min(len(answer.split()) / 100, 1.0)  # 假设理想答案长度为100个词
        return round((coverage + length_score) / 2, 4)

    def _evaluate_consistency(self, answer: str, context: List[Document]) -> float:
        """评估答案与上下文的一致性"""
        context_text = " ".join([doc.page_content for doc in context])
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)
        context_embedding = self.model.encode(context_text, convert_to_tensor=True)
        
        similarity = util.pytorch_cos_sim(answer_embedding, context_embedding)
        return round(float(similarity[0][0]), 4)

    def _evaluate_source_coverage(self, answer: str, context: List[Document]) -> float:
        """评估答案对源文档的覆盖程度"""
        source_coverages = []
        answer_embedding = self.model.encode(answer, convert_to_tensor=True)
        
        for doc in context:
            doc_embedding = self.model.encode(doc.page_content, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(answer_embedding, doc_embedding)
            source_coverages.append(float(similarity[0][0]))
            
        return round(np.mean(source_coverages), 4) if source_coverages else 0.0
