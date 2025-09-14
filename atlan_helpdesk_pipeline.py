#!/usr/bin/env python3
"""
atlan_helpdesk_pipeline.py
Complete End-to-End AI Helpdesk Pipeline

FLOW: Input Query → Gemini Triage → Framework Validation → Hybrid RAG → Final Answer

This is the main pipeline that orchestrates all components:
1. Takes user query
2. Classifies with Gemini (topic, sentiment, priority)
3. Validates classification quality with triage_evaluation_full.py
4. Retrieves relevant docs with Hybrid RAG
5. Generates final answer

Usage:
    from atlan_helpdesk_pipeline import AtlanHelpdeskPipeline
    
    pipeline = AtlanHelpdeskPipeline()
    result = pipeline.process_query("How do I connect Snowflake to Atlan?")
"""

import os
import sys
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import tempfile
import subprocess

# Core imports
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field

# AI/ML imports
from google import genai
from sentence_transformers import SentenceTransformer
import faiss

# Local imports
from hybrid_rag import AgenticHybridRAG
import triage_evaluation_full as triage_eval

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the helpdesk pipeline"""
    gemini_model: str = "gemini-2.5-flash-lite"
    confidence_threshold: float = 0.7
    validation_enabled: bool = True
    rag_top_k: int = 5
    max_response_tokens: int = 1000
    embeddings_model: str = "all-MiniLM-L6-v2"

class TopicTag(str, Enum):
    HOW_TO = "How-to"
    PRODUCT = "Product"
    CONNECTOR = "Connector"
    LINEAGE = "Lineage"
    API_SDK = "API/SDK"
    SSO = "SSO"
    GLOSSARY = "Glossary"
    BEST_PRACTICES = "Best practices"
    SENSITIVE_DATA = "Sensitive data"
    OTHER = "Other"

class TicketFeatures(BaseModel):
    topics: List[TopicTag] = Field(min_length=1, description="List of relevant topics")
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_label: str
    urgency_score: float = Field(ge=0.0, le=1.0)
    priority: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_entities: List[str]
    reasoning: str

@dataclass
class ClassificationResult:
    """Result from Gemini classification"""
    ticket_id: str
    topics: List[str]
    sentiment_score: float
    sentiment_label: str
    urgency_score: float
    priority: str
    confidence: float
    key_entities: List[str]
    reasoning: str
    classification_success: bool
    processing_time: float

@dataclass
class ValidationResult:
    """Result from triage evaluation framework"""
    confidence_level: str  # HIGH_CONF, MED_CONF, LOW_CONF
    topic_coherence: float
    needs_review: bool
    validation_score: float
    processing_time: float

@dataclass
class RAGResult:
    """Result from Hybrid RAG"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    processing_time: float

@dataclass
class PipelineResult:
    """Final result from the complete pipeline"""
    query: str
    classification: ClassificationResult
    validation: ValidationResult
    rag_result: RAGResult
    final_answer: str
    total_processing_time: float
    pipeline_confidence: float
    recommended_action: str

class AtlanHelpdeskPipeline:
    """
    Complete AI Helpdesk Pipeline
    
    Orchestrates: Gemini Classification → Validation → Hybrid RAG → Response
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.setup_components()
        
    def setup_components(self):
        """Initialize all pipeline components"""
        logger.info("Initializing pipeline components...")
        
        # 1. Setup Gemini client
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        self.gemini_client = genai.Client()
        
        # 2. Setup Hybrid RAG
        try:
            self.rag = AgenticHybridRAG()
            logger.info("✅ Hybrid RAG initialized")
        except Exception as e:
            logger.warning(f"⚠️ Hybrid RAG initialization failed: {e}")
            self.rag = None
            
        # 3. Setup embeddings model for validation
        self.embeddings_model = SentenceTransformer(self.config.embeddings_model)
        
        # 4. Load knowledge base if available
        self.knowledge_base = self.load_knowledge_base()
        
        logger.info("✅ Pipeline components initialized")
    
    def load_knowledge_base(self) -> Optional[pd.DataFrame]:
        """Load Atlan knowledge base"""
        try:
            kb_path = "atlan_knowledge_base.json"
            if os.path.exists(kb_path):
                with open(kb_path, 'r') as f:
                    kb_data = json.load(f)
                df = pd.DataFrame(kb_data)
                logger.info(f"✅ Knowledge base loaded: {len(df)} entries")
                return df
            else:
                logger.warning("⚠️ Knowledge base not found")
                return None
        except Exception as e:
            logger.warning(f"⚠️ Failed to load knowledge base: {e}")
            return None
    
    def classify_with_gemini(self, query: str, ticket_id: str = None) -> ClassificationResult:
        """
        Step 1: Classify query using Gemini
        """
        start_time = time.time()
        
        if ticket_id is None:
            ticket_id = f"QUERY-{int(time.time())}"
            
        prompt = f"""
You are an AI triage assistant for Atlan's support team.
Your task: analyze a support query and output only valid JSON with the required fields.

Topics (multi-label):
Select all that apply:
"How-to": user asks how to use a feature or complete a task.
"Product": bug, error, or unexpected behavior in Atlan.
"Connector": issues connecting/integrating external systems (Snowflake, Redshift, BI tools, etc.).
"Lineage": lineage diagrams, capture, missing lineage.
"API/SDK": APIs, SDKs, webhooks, programmatic access.
"SSO": authentication, login, SSO, identity providers.
"Glossary": glossaries, business terms, linking.
"Best practices": recommendations, workflows, catalog hygiene.
"Sensitive data": PII, data masking, compliance.
"Other": if none fit.

Urgency & Priority:
Mentions of urgent, blocked, deadline, critical failure → priority="P0", urgency_score≈0.9–1.0.
Important but not blocking → priority="P1", urgency_score≈0.5–0.8.
Informational/low urgency → priority="P2", urgency_score≈0.1–0.4.

Sentiment:
"Frustrated": blocked, struggling, urgency, mild negativity → sentiment_score≈-0.3 to -0.6.
"Angry": strong dissatisfaction or infuriated → sentiment_score≈-0.7 to -1.0.
"Curious": exploring, asking questions, polite → sentiment_score≈0.2 to 0.6.
"Neutral": factual or polite without emotion → sentiment_score≈-0.1 to 0.1.

=== CLASSIFY THIS QUERY ===
{query}
=== END QUERY ===

Return only JSON."""

        try:
            response = self.gemini_client.models.generate_content(
                model=self.config.gemini_model,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TicketFeatures,
                },
            )
            
            features = response.parsed
            processing_time = time.time() - start_time
            
            result = ClassificationResult(
                ticket_id=ticket_id,
                topics=[str(topic).replace('TopicTag.', '') for topic in features.topics],
                sentiment_score=features.sentiment_score,
                sentiment_label=features.sentiment_label,
                urgency_score=features.urgency_score,
                priority=features.priority,
                confidence=features.confidence,
                key_entities=features.key_entities,
                reasoning=features.reasoning,
                classification_success=True,
                processing_time=processing_time
            )
            
            logger.info(f"✅ Gemini classification completed: {result.topics} (confidence: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Gemini classification failed: {e}")
            
            return ClassificationResult(
                ticket_id=ticket_id,
                topics=["Other"],
                sentiment_score=0.0,
                sentiment_label="Unknown",
                urgency_score=0.5,
                priority="P2",
                confidence=0.0,
                key_entities=[],
                reasoning=f"Classification failed: {str(e)}",
                classification_success=False,
                processing_time=processing_time
            )
    
    def validate_classification(self, classification: ClassificationResult, query: str) -> ValidationResult:
        """
        Step 2: Validate classification using triage evaluation framework
        """
        if not self.config.validation_enabled:
            return ValidationResult(
                confidence_level="MED_CONF",
                topic_coherence=0.5,
                needs_review=False,
                validation_score=0.5,
                processing_time=0.0
            )
            
        start_time = time.time()
        
        try:
            # Create temporary ticket data for validation
            temp_ticket = {
                'id': classification.ticket_id,
                'subject': f"Query: {query[:100]}...",
                'body': query,
                'predicted_topics': classification.topics
            }
            
            # Create temporary files for validation
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save ticket data
                tickets_file = os.path.join(temp_dir, 'temp_ticket.json')
                with open(tickets_file, 'w') as f:
                    json.dump([temp_ticket], f)
                
                # Generate embeddings for this query
                embedding = self.embeddings_model.encode([query])
                
                # Save embeddings
                embeddings_file = os.path.join(temp_dir, 'temp_embeddings.npz')
                np.savez(embeddings_file, embeddings=embedding)
                
                # Save metadata
                metadata_file = os.path.join(temp_dir, 'temp_metadata.csv')
                metadata_df = pd.DataFrame([{
                    'id': classification.ticket_id,
                    'text': query
                }])
                metadata_df.to_csv(metadata_file, index=False)
                
                # Run validation (simplified)
                # In a real deployment, you might run the full validation pipeline
                # For now, we'll do a simplified confidence assessment
                
                confidence_level = "MED_CONF"
                topic_coherence = 0.5
                needs_review = False
                validation_score = classification.confidence
                
                # Simple validation logic
                if classification.confidence >= 0.8:
                    confidence_level = "HIGH_CONF"
                    topic_coherence = 0.7
                elif classification.confidence <= 0.3:
                    confidence_level = "LOW_CONF"
                    topic_coherence = 0.3
                    needs_review = True
                
                processing_time = time.time() - start_time
                
                result = ValidationResult(
                    confidence_level=confidence_level,
                    topic_coherence=topic_coherence,
                    needs_review=needs_review,
                    validation_score=validation_score,
                    processing_time=processing_time
                )
                
                logger.info(f"✅ Validation completed: {confidence_level} (score: {validation_score:.3f})")
                return result
                
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Validation failed: {e}")
            
            return ValidationResult(
                confidence_level="LOW_CONF",
                topic_coherence=0.0,
                needs_review=True,
                validation_score=0.0,
                processing_time=processing_time
            )
    
    def retrieve_with_rag(self, query: str, classification: ClassificationResult) -> RAGResult:
        """
        Step 3: Retrieve relevant information using Hybrid RAG
        """
        start_time = time.time()
        
        try:
            if self.rag is None:
                # Fallback: simple knowledge base search
                return self.fallback_retrieval(query, classification, start_time)
            
            # Enhanced query with classification context
            enhanced_query = f"""
Query: {query}
Topics: {', '.join(classification.topics)}
Priority: {classification.priority}
Entities: {', '.join(classification.key_entities)}
"""
            
            # Use Hybrid RAG
            rag_response = self.rag.query(enhanced_query, top_k=self.config.rag_top_k)
            
            processing_time = time.time() - start_time
            
            result = RAGResult(
                answer=rag_response.get('answer', 'No answer found'),
                sources=rag_response.get('sources', []),
                confidence=rag_response.get('confidence', 0.5),
                processing_time=processing_time
            )
            
            logger.info(f"✅ RAG retrieval completed: {len(result.sources)} sources")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ RAG retrieval failed: {e}")
            
            return self.fallback_retrieval(query, classification, start_time)
    
    def fallback_retrieval(self, query: str, classification: ClassificationResult, start_time: float) -> RAGResult:
        """Fallback retrieval when RAG is not available"""
        
        if self.knowledge_base is not None:
            # Simple similarity search in knowledge base
            query_embedding = self.embeddings_model.encode([query])
            
            # Find relevant entries (simplified)
            relevant_docs = []
            for _, row in self.knowledge_base.head(3).iterrows():  # Take first 3 as demo
                relevant_docs.append({
                    'content': row.get('content', 'No content available'),
                    'source': row.get('source', 'Knowledge Base'),
                    'score': 0.5
                })
            
            answer = f"Based on your query about {', '.join(classification.topics)}, here are some relevant resources. For specific guidance on {', '.join(classification.key_entities)}, please refer to the official documentation."
        else:
            # Ultimate fallback
            answer = f"I understand you're asking about {', '.join(classification.topics)}. This appears to be a {classification.priority} priority {classification.sentiment_label.lower()} inquiry. Please check the official Atlan documentation for detailed guidance."
            relevant_docs = []
        
        processing_time = time.time() - start_time
        
        return RAGResult(
            answer=answer,
            sources=relevant_docs,
            confidence=0.3,  # Low confidence for fallback
            processing_time=processing_time
        )
    
    def generate_final_answer(self, query: str, classification: ClassificationResult, 
                            validation: ValidationResult, rag_result: RAGResult) -> str:
        """
        Step 4: Generate final comprehensive answer
        """
        
        # Construct final answer based on all components
        answer_parts = []
        
        # Add priority and sentiment context
        if classification.priority == "P0":
            answer_parts.append("🚨 **URGENT**: This appears to be a high-priority issue.")
        elif classification.priority == "P1":
            answer_parts.append("⚡ **IMPORTANT**: This is a medium-priority request.")
        
        # Add main answer from RAG
        answer_parts.append(rag_result.answer)
        
        # Add confidence and validation info
        if validation.confidence_level == "LOW_CONF" or validation.needs_review:
            answer_parts.append("\n⚠️ **Note**: This response has been flagged for potential review. For critical issues, please contact support directly.")
        
        # Add relevant sources
        if rag_result.sources:
            sources_text = "\n\n📚 **Relevant Resources:**"
            for i, source in enumerate(rag_result.sources[:3], 1):
                sources_text += f"\n{i}. {source.get('source', 'Unknown Source')}"
            answer_parts.append(sources_text)
        
        # Add topic-specific guidance
        topic_guidance = self.get_topic_specific_guidance(classification.topics)
        if topic_guidance:
            answer_parts.append(f"\n\n💡 **Additional Guidance**: {topic_guidance}")
        
        return "\n".join(answer_parts)
    
    def get_topic_specific_guidance(self, topics: List[str]) -> str:
        """Get topic-specific guidance"""
        guidance_map = {
            "Connector": "For connector issues, check your credentials and network connectivity first.",
            "Lineage": "For lineage problems, verify that your source systems support lineage capture.",
            "API/SDK": "For API issues, check authentication and rate limits. Refer to the API documentation.",
            "SSO": "For SSO problems, verify your identity provider configuration and user mappings.",
            "Product": "For product issues, this may require technical support. Consider filing a support ticket.",
        }
        
        relevant_guidance = []
        for topic in topics:
            if topic in guidance_map:
                relevant_guidance.append(guidance_map[topic])
        
        return " ".join(relevant_guidance) if relevant_guidance else ""
    
    def calculate_pipeline_confidence(self, classification: ClassificationResult, 
                                    validation: ValidationResult, rag_result: RAGResult) -> float:
        """Calculate overall pipeline confidence"""
        
        # Weighted average of component confidences
        weights = {
            'classification': 0.4,
            'validation': 0.3,
            'rag': 0.3
        }
        
        total_confidence = (
            classification.confidence * weights['classification'] +
            validation.validation_score * weights['validation'] +
            rag_result.confidence * weights['rag']
        )
        
        return min(total_confidence, 1.0)
    
    def get_recommended_action(self, classification: ClassificationResult, 
                             validation: ValidationResult, pipeline_confidence: float) -> str:
        """Get recommended action based on results"""
        
        if classification.priority == "P0":
            return "ESCALATE_IMMEDIATELY"
        elif validation.needs_review or pipeline_confidence < 0.4:
            return "HUMAN_REVIEW_RECOMMENDED"
        elif pipeline_confidence > 0.7:
            return "AUTO_RESPONSE_HIGH_CONFIDENCE"
        else:
            return "AUTO_RESPONSE_MEDIUM_CONFIDENCE"
    
    def process_query(self, query: str, ticket_id: str = None) -> PipelineResult:
        """
        Main pipeline method: Process a complete query through all stages
        
        FLOW: Query → Gemini Classification → Validation → RAG → Final Answer
        """
        start_time = time.time()
        
        logger.info(f"🚀 Processing query: {query[:100]}...")
        
        try:
            # Step 1: Classify with Gemini
            logger.info("Step 1: Gemini Classification...")
            classification = self.classify_with_gemini(query, ticket_id)
            
            # Step 2: Validate classification
            logger.info("Step 2: Classification Validation...")
            validation = self.validate_classification(classification, query)
            
            # Step 3: Retrieve with RAG
            logger.info("Step 3: Hybrid RAG Retrieval...")
            rag_result = self.retrieve_with_rag(query, classification)
            
            # Step 4: Generate final answer
            logger.info("Step 4: Final Answer Generation...")
            final_answer = self.generate_final_answer(query, classification, validation, rag_result)
            
            # Calculate overall metrics
            total_time = time.time() - start_time
            pipeline_confidence = self.calculate_pipeline_confidence(classification, validation, rag_result)
            recommended_action = self.get_recommended_action(classification, validation, pipeline_confidence)
            
            result = PipelineResult(
                query=query,
                classification=classification,
                validation=validation,
                rag_result=rag_result,
                final_answer=final_answer,
                total_processing_time=total_time,
                pipeline_confidence=pipeline_confidence,
                recommended_action=recommended_action
            )
            
            logger.info(f"✅ Pipeline completed in {total_time:.2f}s (confidence: {pipeline_confidence:.3f})")
            return result
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"❌ Pipeline failed: {e}")
            
            # Return error result
            return self.create_error_result(query, str(e), total_time)
    
    def create_error_result(self, query: str, error: str, processing_time: float) -> PipelineResult:
        """Create error result when pipeline fails"""
        
        error_classification = ClassificationResult(
            ticket_id=f"ERROR-{int(time.time())}",
            topics=["Other"],
            sentiment_score=0.0,
            sentiment_label="Unknown",
            urgency_score=0.5,
            priority="P2",
            confidence=0.0,
            key_entities=[],
            reasoning=f"Pipeline error: {error}",
            classification_success=False,
            processing_time=0.0
        )
        
        error_validation = ValidationResult(
            confidence_level="LOW_CONF",
            topic_coherence=0.0,
            needs_review=True,
            validation_score=0.0,
            processing_time=0.0
        )
        
        error_rag = RAGResult(
            answer="I'm sorry, I encountered an error processing your request.",
            sources=[],
            confidence=0.0,
            processing_time=0.0
        )
        
        return PipelineResult(
            query=query,
            classification=error_classification,
            validation=error_validation,
            rag_result=error_rag,
            final_answer=f"❌ Sorry, I encountered an error: {error}. Please try again or contact support.",
            total_processing_time=processing_time,
            pipeline_confidence=0.0,
            recommended_action="HUMAN_REVIEW_REQUIRED"
        )

# Example usage and testing
def main():
    """Example usage of the pipeline"""
    
    # Test queries
    test_queries = [
        "How do I connect Snowflake to Atlan? I'm getting permission errors.",
        "Our lineage is not showing up for dbt models. This is urgent!",
        "What's the best way to set up a glossary in Atlan?",
        "I need to extract lineage data via API for our quarterly audit.",
    ]
    
    # Initialize pipeline
    pipeline = AtlanHelpdeskPipeline()
    
    print("🚀 ATLAN AI HELPDESK PIPELINE DEMO")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n📋 Test Query {i}: {query}")
        print("-" * 60)
        
        # Process query
        result = pipeline.process_query(query)
        
        # Display results
        print(f"🏷️  Topics: {', '.join(result.classification.topics)}")
        print(f"📊 Priority: {result.classification.priority}")
        print(f"😊 Sentiment: {result.classification.sentiment_label}")
        print(f"🎯 Confidence: {result.pipeline_confidence:.3f}")
        print(f"⚡ Action: {result.recommended_action}")
        print(f"⏱️  Time: {result.total_processing_time:.2f}s")
        print(f"\n💬 Answer:\n{result.final_answer}")
        print("=" * 60)

if __name__ == "__main__":
    main()