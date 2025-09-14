#!/usr/bin/env python3
"""
triage.py
Ticket Triage Classification Module

This module provides the core triage functionality extracted from triage.ipynb.
It includes Gemini-based classification and multi-label validation.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum

import pandas as pd
import numpy as np
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field
from google import genai

logger = logging.getLogger(__name__)

# Ensure NLTK punkt is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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
class TriageResult:
    """Result from ticket triage classification"""
    ticket_id: str
    subject: str
    body: str
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
    """Result from ticket validation"""
    ticket_id: str
    status: str  # confident, moderate, disagreement, isolated
    confidence: str  # high, medium, low
    action: str  # keep, review, manual_review
    agreement_score: float
    cluster_coherence: float
    needs_retriage: bool

class MultiLabelEvaluator:
    """Multi-label evaluator for clustering and consensus validation"""
    
    def __init__(self, sentence_model: str = 'all-MiniLM-L6-v2'):
        self.sentence_model = SentenceTransformer(sentence_model)
        
    def parse_topics(self, topics_field):
        """Parse topics from various formats"""
        if isinstance(topics_field, list):
            return [str(t).replace('TopicTag.', '') for t in topics_field]
        if isinstance(topics_field, str):
            if 'TopicTag.' in topics_field:
                import re
                pattern = r"'([^']+)'"
                matches = re.findall(pattern, topics_field)
                return matches
            return [t.strip() for t in topics_field.split(',') if t.strip()]
        return []
    
    def extract_salient_sentences(self, text: str, top_k: int = 2) -> List[str]:
        """Extract most salient sentences based on embedding similarity to full text"""
        sentences = sent_tokenize(text)
        if len(sentences) <= top_k:
            return sentences
        
        # Get embeddings
        full_text_emb = self.sentence_model.encode([text])
        sentence_embs = self.sentence_model.encode(sentences)
        
        # Calculate salience (similarity to full text)
        similarities = cosine_similarity(sentence_embs, full_text_emb).flatten()
        
        # Get top-k most salient sentences
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [sentences[i] for i in sorted(top_indices)]
    
    def evaluate_tickets(self, tickets_df: pd.DataFrame) -> Tuple[Dict, List[str], np.ndarray]:
        """Evaluate tickets using clustering and consensus analysis"""
        
        # Parse topics
        tickets_df = tickets_df.copy()
        tickets_df['topics_parsed'] = tickets_df['topics'].apply(self.parse_topics)
        
        # Extract salient sentences for each ticket
        logger.info("Extracting salient sentences...")
        salient_texts = []
        for text in tickets_df['body']:
            salient_sentences = self.extract_salient_sentences(text, top_k=2)
            salient_texts.append(' '.join(salient_sentences))
        
        # Get embeddings of salient text only
        logger.info("Computing embeddings...")
        text_embeddings = self.sentence_model.encode(salient_texts)
        
        # Build similarity graph
        similarity_matrix = cosine_similarity(text_embeddings)
        G = nx.Graph()
        similarity_threshold = 0.3  
        
        # Add nodes
        for i in range(len(tickets_df)):
            G.add_node(i, ticket_id=tickets_df.iloc[i]['ticket_id'])
        
        # Add edges for similar tickets
        for i in range(len(tickets_df)):
            for j in range(i+1, len(tickets_df)):
                if similarity_matrix[i][j] > similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # Find clusters
        clusters = list(nx.connected_components(G))
        cluster_analysis = {}
        outliers = []
        
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) == 1:
                node = list(cluster)[0]
                outliers.append(tickets_df.iloc[node]['ticket_id'])
                continue
                
            # Analyze cluster topics
            cluster_topics = []
            for node in cluster:
                cluster_topics.extend(tickets_df.iloc[node]['topics_parsed'])
            
            topic_counter = Counter(cluster_topics)
            
            # Calculate coherence using salient embeddings
            cluster_embeddings = text_embeddings[list(cluster)]
            cluster_sim_matrix = cosine_similarity(cluster_embeddings)
            upper_triangle = cluster_sim_matrix[np.triu_indices_from(cluster_sim_matrix, k=1)]
            avg_coherence = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0.0
            
            cluster_analysis[cluster_id] = {
                'size': len(cluster),
                'coherence': avg_coherence,
                'dominant_topics': topic_counter.most_common(3),
                'tickets': [tickets_df.iloc[node]['ticket_id'] for node in cluster]
            }
        
        return cluster_analysis, outliers, similarity_matrix

class TriageClassifier:
    """Main triage classifier using Gemini API"""
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.client = genai.Client()
        self.evaluator = MultiLabelEvaluator()
        
    def classify_ticket(self, ticket: Dict[str, Any]) -> TriageResult:
        """
        Classify a single ticket using Gemini API
        
        Args:
            ticket: Dictionary containing ticket data with 'id', 'subject', and 'body'
        
        Returns:
            TriageResult containing classification results
        """
        start_time = time.time()
        
        prompt = f"""
You are an AI triage assistant for a data platform's support team.
Your task: analyze a support ticket and output only valid JSON with the required fields.

Rules

Topics (multi-label):
Select all that apply from this list:
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

Other fields:
key_entities: short technical terms (e.g., "Snowflake", "dbt", "Okta"), not full sentences.
reasoning: 1–2 sentences explaining why you chose these labels.
confidence: 0.0–1.0 (higher if ticket is clear).

=== CLASSIFY THIS TICKET ===
{ticket['body']}
=== END TICKET ===

Return only JSON."""

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    'response_mime_type': 'application/json',
                    'response_schema': TicketFeatures,
                },
            )
            
            features = response.parsed
            processing_time = time.time() - start_time
            
            result = TriageResult(
                ticket_id=ticket['id'],
                subject=ticket.get('subject', ''),
                body=ticket['body'],
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
            
            logger.info(f"✅ Classified ticket {ticket['id']}: {result.topics} (confidence: {result.confidence:.3f})")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Error classifying ticket {ticket['id']}: {str(e)}")
            
            return TriageResult(
                ticket_id=ticket['id'],
                subject=ticket.get('subject', ''),
                body=ticket['body'],
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
    
    def validate_classification(self, triage_result: TriageResult, historical_results: List[TriageResult] = None) -> ValidationResult:
        """
        Validate a single ticket classification using consensus validation
        
        Args:
            triage_result: The triage result to validate
            historical_results: Optional list of historical results for context
        
        Returns:
            ValidationResult containing validation metrics
        """
        
        # If no historical context, return medium confidence validation
        if not historical_results or len(historical_results) < 5:
            confidence_level = "medium" if triage_result.confidence >= 0.6 else "low"
            needs_retriage = triage_result.confidence < 0.4
            
            return ValidationResult(
                ticket_id=triage_result.ticket_id,
                status="isolated",
                confidence=confidence_level,
                action="keep" if not needs_retriage else "review",
                agreement_score=triage_result.confidence,
                cluster_coherence=0.0,
                needs_retriage=needs_retriage
            )
        
        # Create DataFrame for evaluation
        all_results = historical_results + [triage_result]
        df_data = []
        for result in all_results:
            df_data.append({
                'ticket_id': result.ticket_id,
                'body': result.body,
                'topics': result.topics,
                'confidence': result.confidence
            })
        
        df = pd.DataFrame(df_data)
        
        try:
            # Run cluster analysis
            cluster_analysis, outliers, similarity_matrix = self.evaluator.evaluate_tickets(df)
            
            # Validate the specific ticket
            ticket_id = triage_result.ticket_id
            current_topics = set(triage_result.topics)
            
            if ticket_id in outliers:
                # Isolated ticket - trust original classification if confident
                confidence_level = "medium" if triage_result.confidence >= 0.6 else "low"
                needs_retriage = triage_result.confidence < 0.5
                
                return ValidationResult(
                    ticket_id=ticket_id,
                    status="isolated",
                    confidence=confidence_level,
                    action="keep" if not needs_retriage else "review",
                    agreement_score=triage_result.confidence,
                    cluster_coherence=0.0,
                    needs_retriage=needs_retriage
                )
            
            # Find cluster for this ticket
            ticket_cluster = None
            for cluster_id, data in cluster_analysis.items():
                if ticket_id in data['tickets']:
                    ticket_cluster = data
                    break
            
            if not ticket_cluster:
                # No cluster found - treat as isolated
                return ValidationResult(
                    ticket_id=ticket_id,
                    status="isolated",
                    confidence="medium",
                    action="keep",
                    agreement_score=triage_result.confidence,
                    cluster_coherence=0.0,
                    needs_retriage=False
                )
            
            # Calculate topic agreement with cluster
            cluster_topics = dict(ticket_cluster['dominant_topics'])
            cluster_topic_set = set(cluster_topics.keys())
            
            # Jaccard similarity
            intersection = len(current_topics & cluster_topic_set)
            union = len(current_topics | cluster_topic_set)
            agreement = intersection / union if union > 0 else 0
            
            # Determine validation result
            if agreement >= 0.5:
                status = 'confident'
                confidence = 'high'
                action = 'keep'
                needs_retriage = False
            elif agreement >= 0.25:
                status = 'moderate'
                confidence = 'medium'
                action = 'keep'
                needs_retriage = False
            else:
                status = 'disagreement'
                confidence = 'low'
                action = 'review'
                needs_retriage = True
            
            return ValidationResult(
                ticket_id=ticket_id,
                status=status,
                confidence=confidence,
                action=action,
                agreement_score=agreement,
                cluster_coherence=ticket_cluster['coherence'],
                needs_retriage=needs_retriage
            )
            
        except Exception as e:
            logger.error(f"❌ Validation failed for ticket {ticket_id}: {e}")
            
            # Fallback validation based on confidence
            confidence_level = "high" if triage_result.confidence >= 0.8 else "medium" if triage_result.confidence >= 0.5 else "low"
            needs_retriage = triage_result.confidence < 0.4
            
            return ValidationResult(
                ticket_id=ticket_id,
                status="error",
                confidence=confidence_level,
                action="keep" if not needs_retriage else "review",
                agreement_score=triage_result.confidence,
                cluster_coherence=0.0,
                needs_retriage=needs_retriage
            )

def process_tickets_batch(tickets: List[Dict], classifier: TriageClassifier, 
                         batch_size: int = 15, pause_seconds: int = 60) -> List[TriageResult]:
    """
    Process tickets in batches to avoid rate limits
    
    Args:
        tickets: List of ticket dictionaries
        classifier: TriageClassifier instance
        batch_size: Number of tickets per batch
        pause_seconds: Seconds to pause between batches
    
    Returns:
        List of TriageResult objects
    """
    results = []
    total = len(tickets)
    
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logger.info(f"Processing tickets {start+1} to {end} of {total}")
        
        for i in range(start, end):
            result = classifier.classify_ticket(tickets[i])
            results.append(result)
        
        if end < total:
            logger.info(f"Batch complete. Waiting {pause_seconds} seconds before next batch...")
            time.sleep(pause_seconds)
    
    logger.info(f"Successfully processed {len(results)} tickets")
    return results