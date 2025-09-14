"""
Agentic Hybrid RAG System for Atlan Customer Support
Following the exact architecture from customer_support_rag_readme.md:
- Query Analysis Agent
- Multi-Modal Retrieval with Reciprocal Rank Fusion  
- Semantic Reranking Agent (simplified for reliability)
- Citation-Aware Generation using Gemini LLM
"""

import json
import os
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import google.generativeai as genai

class QueryAnalysisAgent:
    """Analyzes queries and determines optimal retrieval strategy"""
    
    def __init__(self):
        self.strategy_mapping = {
            'How-to': 'lexical_heavy',      # Users want specific steps
            'Product': 'balanced',          # Mix of concepts and specifics
            'API/SDK': 'lexical_heavy',     # Exact API names, methods  
            'Best practices': 'semantic_heavy',  # Conceptual content
            'SSO': 'lexical_heavy'          # Specific configuration steps
        }
    
    def analyze_query(self, query_text: str, classification: Dict) -> Dict:
        """Determine retrieval strategy based on query and classification"""
        
        topic = classification.get('topic', 'Other')
        priority = classification.get('priority', 'P2')
        
        # Determine retrieval strategy
        strategy = self._determine_strategy(topic, priority)
        
        # Extract key entities for query expansion
        entities = self._extract_entities(query_text)
        
        return {
            'retrieval_strategy': strategy,
            'key_entities': entities,
            'query_type': self._classify_query_type(query_text),
            'complexity_score': self._assess_complexity(query_text)
        }
    
    def _determine_strategy(self, topic: str, priority: str) -> str:
        """Map topics to retrieval strategies"""
        if topic in self.strategy_mapping:
            base_strategy = self.strategy_mapping[topic]
            
            # Adjust for priority - high priority gets lexical boost for precision
            if priority == 'P0':
                if 'semantic' in base_strategy:
                    return 'balanced'
                else:
                    return base_strategy
            
            return base_strategy
        
        return 'balanced'  # Default
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key technical entities from query"""
        # Simple entity extraction - could be enhanced with NER
        technical_terms = []
        words = query.lower().split()
        
        # Common technical entities in Atlan context
        atlan_entities = ['snowflake', 'tableau', 'fivetran', 'dbt', 'redshift', 
                         'api', 'sdk', 'connector', 'lineage', 'glossary', 'sso']
        
        for word in words:
            if word in atlan_entities:
                technical_terms.append(word)
        
        return technical_terms
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for better processing"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['how', 'setup', 'configure', 'install']):
            return 'procedural'
        elif any(word in query_lower for word in ['error', 'fail', 'issue', 'problem']):
            return 'troubleshooting'
        elif any(word in query_lower for word in ['what', 'explain', 'understand']):
            return 'conceptual'
        else:
            return 'general'
    
    def _assess_complexity(self, query: str) -> float:
        """Assess query complexity (0-1 scale)"""
        complexity_indicators = ['integrate', 'configure', 'multiple', 'complex', 'advanced']
        words = query.lower().split()
        
        complexity_score = sum(1 for word in words if word in complexity_indicators)
        return min(complexity_score / len(complexity_indicators), 1.0)


class HybridRetriever:
    """Multi-Modal Retrieval System with Reciprocal Rank Fusion"""
    
    def __init__(self, docs_file: str = 'atlan_knowledge_base.json'):
        print("🔧 Initializing Hybrid Retrieval System...")
        
        # Try to load pre-computed embeddings first
        if self._load_cached_embeddings():
            print("⚡ Loaded pre-computed embeddings - instant startup!")
        else:
            print("📚 Computing embeddings from scratch...")
            # Load documents and create chunks
            self.documents = self._load_documents(docs_file)
            self.chunks = self._create_chunks_with_metadata()
            
            # Initialize embedding model
            print("📥 Loading embedding model...")
            self.embedder = SentenceTransformer('all-mpnet-base-v2')  # Higher quality as per your plan
            
            # Create embeddings with deterministic behavior
            print("Creating embeddings...")
            chunk_texts = [chunk['content'] for chunk in self.chunks]
            self.embeddings = self.embedder.encode(chunk_texts, show_progress_bar=True)
        
        # Initialize BM25
        print("🔍 Setting up BM25 search...")
        texts = [chunk['content'] for chunk in self.chunks]
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized_texts)
        
        print("✅ Hybrid Retrieval System ready!")
    
    def _load_cached_embeddings(self) -> bool:
        """Load pre-computed embeddings if available"""
        cache_dir = 'embeddings_cache'
        
        if not os.path.exists(f'{cache_dir}/embeddings.npy'):
            return False
        
        try:
            # Load embeddings
            self.embeddings = np.load(f'{cache_dir}/embeddings.npy')
            
            # Load chunks
            with open(f'{cache_dir}/chunks.json', 'r') as f:
                self.chunks = json.load(f)
            
            # Load model info
            with open(f'{cache_dir}/model_info.json', 'r') as f:
                model_info = json.load(f)
            
            # Initialize embedding model (needed for query encoding)
            self.embedder = SentenceTransformer(model_info['model_name'])
            
            print(f"📦 Loaded {len(self.chunks)} cached chunks with embeddings")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load cached embeddings: {e}")
            return False
    
    def _load_documents(self, docs_file: str) -> List[Dict]:
        """Load scraped documents"""
        if not os.path.exists(docs_file):
            print(f"Documents file {docs_file} not found.")
            return []
        
        with open(docs_file, 'r') as f:
            docs = json.load(f)
        
        print(f"📚 Loaded {len(docs)} documents")
        return docs
    
    def _create_chunks_with_metadata(self) -> List[Dict]:
        """Create chunks with proper metadata preservation as per your plan"""
        chunks = []
        
        for doc in self.documents:
            content = doc['content']
            
            # Hierarchical chunking: Page → Section → Paragraph
            sections = content.split('\n\n')
            
            for i, section in enumerate(sections):
                if len(section.strip()) > 100:  # Minimum chunk size
                    
                    # Create overlapping chunks (400 chars with 100 overlap)
                    chunk_size = 400
                    overlap = 100
                    
                    if len(section) <= chunk_size:
                        # Single chunk
                        chunks.append({
                            'content': section.strip(),
                            'metadata': {
                                'source_url': doc['url'],
                                'title': doc['title'],
                                'section_index': i,
                                'chunk_type': 'complete_section'
                            }
                        })
                    else:
                        # Multiple overlapping chunks
                        start = 0
                        while start < len(section):
                            end = start + chunk_size
                            chunk_content = section[start:end]
                            
                            chunks.append({
                                'content': chunk_content.strip(),
                                'metadata': {
                                    'source_url': doc['url'],
                                    'title': doc['title'],
                                    'section_index': i,
                                    'chunk_type': 'overlapping',
                                    'chunk_start': start
                                }
                            })
                            
                            if end >= len(section):
                                break
                            start = end - overlap
        
        print(f"Created {len(chunks)} chunks with metadata")
        return chunks
    
    def retrieve(self, query: str, strategy: str, top_k: int = 20) -> List[Dict]:
        """Retrieve using hybrid approach with strategy-based weighting"""
        
        # Get strategy weights
        weights = self._get_strategy_weights(strategy)
        
        # BM25 lexical search
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:top_k * 2]
        
        # Dense vector semantic search
        query_embedding = self.embedder.encode([query])
        semantic_scores = np.dot(self.embeddings, query_embedding.T).flatten()
        semantic_top_indices = np.argsort(semantic_scores)[::-1][:top_k * 2]
        
        # Prepare results for RRF
        lexical_results = [(i, bm25_scores[i]) for i in bm25_top_indices]
        semantic_results = [(i, semantic_scores[i]) for i in semantic_top_indices]
        
        # Apply Reciprocal Rank Fusion
        combined_results = self._reciprocal_rank_fusion(
            lexical_results, semantic_results, weights
        )
        
        return combined_results[:top_k]
    
    def _get_strategy_weights(self, strategy: str) -> Dict[str, float]:
        """Get retrieval weights based on strategy"""
        strategy_weights = {
            'lexical_heavy': {'lexical': 0.7, 'semantic': 0.3},
            'semantic_heavy': {'lexical': 0.3, 'semantic': 0.7},
            'balanced': {'lexical': 0.5, 'semantic': 0.5}
        }
        
        return strategy_weights.get(strategy, strategy_weights['balanced'])
    
    def _reciprocal_rank_fusion(self, lexical_results: List, semantic_results: List, 
                               weights: Dict) -> List[Dict]:
        """
        Reciprocal Rank Fusion 
        Reference: https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
        """
        k = 60  # RRF parameter
        doc_scores = {}
        
        # Add lexical scores with RRF
        for rank, (doc_idx, score) in enumerate(lexical_results):
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = {'lexical_rrf': 0, 'semantic_rrf': 0}
            doc_scores[doc_idx]['lexical_rrf'] = 1 / (k + rank + 1)
        
        # Add semantic scores with RRF
        for rank, (doc_idx, score) in enumerate(semantic_results):
            if doc_idx not in doc_scores:
                doc_scores[doc_idx] = {'lexical_rrf': 0, 'semantic_rrf': 0}
            doc_scores[doc_idx]['semantic_rrf'] = 1 / (k + rank + 1)
        
        # Combine with strategy weights
        final_results = []
        for doc_idx, scores in doc_scores.items():
            combined_score = (
                weights['lexical'] * scores['lexical_rrf'] + 
                weights['semantic'] * scores['semantic_rrf']
            )
            
            final_results.append({
                'chunk': self.chunks[doc_idx],
                'score': combined_score,
                'lexical_rrf': scores['lexical_rrf'],
                'semantic_rrf': scores['semantic_rrf']
            })
        
        # Sort by combined score
        return sorted(final_results, key=lambda x: x['score'], reverse=True)


class RerankingAgent:
    """Simple reranking agent using sentence similarity"""
    
    def __init__(self):
        print("🔧 Loading reranking model...")
        # Use the same embedding model for consistency and simplicity
        self.rerank_model = SentenceTransformer('all-mpnet-base-v2')
        print("✅ Reranking model loaded!")
    
    def rerank_documents(self, query: str, retrieved_docs: List[Dict], top_k: int = 10) -> List[Dict]:
        """Rerank using semantic similarity scoring"""
        
        if not retrieved_docs:
            return []
        
        # Extract texts for reranking
        doc_texts = []
        for doc_data in retrieved_docs:
            chunk = doc_data['chunk']
            doc_texts.append(chunk['content'][:512])  # Truncate for efficiency
        
        # Get query embedding
        query_embedding = self.rerank_model.encode([query])
        
        # Get document embeddings
        doc_embeddings = self.rerank_model.encode(doc_texts)
        
        # Calculate similarity scores
        similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
        
        # Create reranked results
        reranked_scores = []
        for i, doc_data in enumerate(retrieved_docs):
            reranked_scores.append({
                'chunk': doc_data['chunk'],
                'relevance_score': float(similarities[i]),
                'original_score': doc_data['score'],
                'lexical_rrf': doc_data.get('lexical_rrf', 0),
                'semantic_rrf': doc_data.get('semantic_rrf', 0)
            })
        
        # Sort by relevance score
        reranked_scores.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        return reranked_scores[:top_k]


class CitationAwareGenerator:
    """Citation-Aware Generation using Gemini LLM only"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')  # Fast and reliable
        
    def generate_response(self, query: str, reranked_docs: List[Dict]) -> Dict:
        """Generate response with proper citations using only Gemini"""
        
        if not reranked_docs:
            return {
                'answer': "I don't have enough information in the knowledge base to answer this question.",
                'sources': [],
                'context_used': 0
            }
        
        # Assemble context with source tracking
        context_parts = []
        source_urls = set()
        
        for i, doc_data in enumerate(reranked_docs):
            chunk = doc_data['chunk']
            source_url = chunk['metadata']['source_url']
            source_urls.add(source_url)
            
            context_parts.append(
                f"[Source {i+1}: {source_url}]\n{chunk['content']}\n"
            )
        
        context_text = "\n".join(context_parts)
        
        # Create deterministic prompt for factual accuracy
        prompt = f"""You are an expert Atlan customer support agent. Answer the user's question using the provided documentation context.

REQUIREMENTS:
1. Use ONLY information explicitly stated in the provided context
2. Include source citations for ALL factual claims using [Source X] format
3. Be helpful and actionable - provide next steps when possible
4. If specific details aren't available, acknowledge what IS known and suggest logical next steps
5. Never invent information, but be constructive rather than just saying "I don't know"
6. Maintain professional, helpful tone

RESPONSE STRUCTURE:
- Start with what you CAN help with based on the documentation
- Provide relevant information from the context with citations
- When information is missing, suggest practical next steps (contact support, check documentation sections, etc.)
- End with actionable guidance

Context from Atlan Documentation:
{context_text}

User Question: {query}

Provide a helpful response with proper source citations:"""

        try:
            # Generate with low temperature for deterministic behavior
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistency
                    max_output_tokens=800,
                    top_p=0.8
                )
            )
            
            answer = response.text
            
            return {
                'answer': answer,
                'sources': list(source_urls),
                'context_used': len(reranked_docs),
                'generation_successful': True
            }
            
        except Exception as e:
            return {
                'answer': f"I apologize, but I encountered an error while generating the response: {str(e)}",
                'sources': [],
                'context_used': 0,
                'generation_successful': False
            }


class AgenticHybridRAG:
    """Main orchestrator following your agentic hybrid RAG architecture"""
    
    def __init__(self, docs_file: str = 'atlan_knowledge_base.json'):
        print("🚀 Initializing Agentic Hybrid RAG System...")
        
        # Initialize all agents
        self.query_analyzer = QueryAnalysisAgent()
        self.retriever = HybridRetriever(docs_file)
        self.reranker = RerankingAgent()
        
        # Initialize generator with Gemini
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment")
        
        self.generator = CitationAwareGenerator(api_key)
        
        # Initialize classifier
        from pipeline.feature_extractor import GeminiFeatureExtractor
        self.classifier = GeminiFeatureExtractor(api_key)
        
        print("Agentic Hybrid RAG System ready!")
    
    def answer_question(self, query: str, classification: Dict = None) -> Dict:
        """Complete agentic RAG pipeline"""
        
        # Default classification if not provided
        if classification is None:
            classification = {'topic': 'How-to', 'priority': 'P1'}
        
        # Step 1: Query Analysis
        analysis = self.query_analyzer.analyze_query(query, classification)
        
        # Step 2: Multi-Modal Retrieval with strategy
        retrieved_docs = self.retriever.retrieve(
            query, 
            analysis['retrieval_strategy'], 
            top_k=20
        )
        
        if not retrieved_docs:
            return {
                'answer': "I don't have enough information in the knowledge base to answer this question.",
                'sources': [],
                'pipeline_info': {
                    'retrieval_strategy': analysis['retrieval_strategy'],
                    'documents_retrieved': 0,
                    'documents_reranked': 0
                }
            }
        
        # Step 3: Cross-Encoder Reranking
        reranked_docs = self.reranker.rerank_documents(query, retrieved_docs, top_k=10)
        
        # Step 4: Citation-Aware Generation
        result = self.generator.generate_response(query, reranked_docs)
        
        # Add pipeline information
        result['pipeline_info'] = {
            'retrieval_strategy': analysis['retrieval_strategy'],
            'query_type': analysis['query_type'],
            'complexity_score': analysis['complexity_score'],
            'documents_retrieved': len(retrieved_docs),
            'documents_reranked': len(reranked_docs),
            'key_entities': analysis['key_entities']
        }
        
        return result

def main():
    # Test the Agentic Hybrid RAG system
    rag = AgenticHybridRAG()
    
    test_questions = [
        "How do I connect Snowflake to Atlan?",
        "What permissions are needed for the API?", 
        "How do I create a custom connector?"
    ]
    
    for question in test_questions:
        print(f"\n❓ Question: {question}")
        result = rag.answer_question(question)
        print(f"Answer: {result['answer']}")
        print(f"Sources: {result['sources']}")
        print(f"Pipeline: {result['pipeline_info']}")

if __name__ == "__main__":
    main()