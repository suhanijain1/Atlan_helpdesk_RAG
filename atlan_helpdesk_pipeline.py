# chatbot_pipeline.py

import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any

# Import the core modular components
from triage import TriageClassifier, TriageResult, ValidationResult
from hybrid_rag import AgenticHybridRAG

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class PipelineResult:
    """A structured class for the final output of the pipeline."""
    query: str
    triage: TriageResult
    validation: ValidationResult
    rag_response: Dict[str, Any]
    final_answer: str
    processing_time: float

class ChatbotPipeline:
    """
    A modular, end-to-end pipeline for the Atlan customer support chatbot.
    
    This pipeline orchestrates three distinct stages:
    1. Triage: Classifies the incoming query.
    2. Validate: Checks the classification confidence against historical data.
    3. Respond: Generates a comprehensive answer using the Hybrid RAG system.
    """

    def __init__(self, historical_data: List[Dict] = None):
        """
        Initializes the pipeline with all necessary components.
        
        Args:
            historical_data (List[Dict], optional): A list of past tickets for the
                                                    validation stage. Defaults to None.
        """
        logging.info("Initializing Chatbot Pipeline...")
        self.triage_classifier = TriageClassifier()
        self.rag_system = AgenticHybridRAG()
        
        # Load historical data into TriageResult objects for the validator
        self.historical_results = self._load_historical_data(historical_data or [])
        logging.info(f"Pipeline initialized with {len(self.historical_results)} historical records for validation.")

    def _load_historical_data(self, ticket_data: List[Dict]) -> List[TriageResult]:
        """Helper to convert raw historical ticket dicts into TriageResult objects."""
        # This is a simplified placeholder. In production, you'd load pre-classified results.
        # For this demo, we'll create placeholder results.
        return [
            TriageResult(
                ticket_id=ticket.get('id'),
                subject=ticket.get('subject', ''),
                body=ticket.get('body', ''),
                topics=ticket.get('topics', ['Other']),
                sentiment_score=0.0,
                sentiment_label='Neutral',
                urgency_score=0.5,
                priority='P2',
                confidence=0.8, # Assume historical data is reasonably confident
                key_entities=[],
                reasoning='',
                classification_success=True,
                processing_time=0
            ) for ticket in ticket_data
        ]

    # Use this method in your atlan_helpdesk_pipeline.py
    def run(self, query: str, ticket_id: str = None) -> PipelineResult:
        """
        Executes the full pipeline, sending all tickets to the RAG system.
        """
        start_time = time.time()
        if not ticket_id:
            ticket_id = f"TICKET-{int(time.time())}"
        
        logging.info(f"Processing query for ticket [{ticket_id}]...")
        
        triage_result = self._stage_triage(query, ticket_id)
        validation_result = self._stage_validate(triage_result)
        
        # This simplified logic sends all tickets to RAG for intelligent handling.
        logging.info(f"[{ticket_id}] All topics are sent to RAG for intelligent handling.")
        rag_response = self._stage_respond(query, triage_result)
        final_answer = self._format_final_answer(triage_result, validation_result, rag_response)

        total_time = time.time() - start_time
        logging.info(f"Pipeline finished for ticket [{ticket_id}] in {total_time:.2f}s.")

        return PipelineResult(
            query=query, triage=triage_result, validation=validation_result,
            rag_response=rag_response, final_answer=final_answer,
            processing_time=total_time
        )
    
    def _stage_triage(self, query: str, ticket_id: str) -> TriageResult:
        """Calls the TriageClassifier module."""
        logging.info(f"[{ticket_id}] Stage 1: Triage Classification...")
        ticket_data = {'id': ticket_id, 'subject': query[:80], 'body': query}
        result = self.triage_classifier.classify_ticket(ticket_data)
        logging.info(f"[{ticket_id}] Triage complete. Topics: {result.topics}, Priority: {result.priority}")
        return result

    def _stage_validate(self, triage_result: TriageResult) -> ValidationResult:
        """Calls the validation logic within the TriageClassifier module."""
        logging.info(f"[{triage_result.ticket_id}] Stage 2: Classification Validation...")
        result = self.triage_classifier.validate_classification(triage_result, self.historical_results)
        logging.info(f"[{triage_result.ticket_id}] Validation complete. Status: {result.status}, Confidence: {result.confidence}")
        return result
        
    def _stage_respond(self, query: str, triage_result: TriageResult) -> Dict[str, Any]:
        """Calls the AgenticHybridRAG module with the correct context."""
        logging.info(f"[{triage_result.ticket_id}] Stage 3: Response Generation...")
        
        try:
            classification_context = {
                "topic": triage_result.topics[0] if triage_result.topics else "Other",
                "all_topics": triage_result.topics,
                "priority": triage_result.priority
            }
            result = self.rag_system.answer_question(query, classification=classification_context)
            
            # Ensure result is a valid dictionary
            if not isinstance(result, dict):
                logging.error(f"[{triage_result.ticket_id}] RAG system returned invalid result type: {type(result)}")
                return {'answer': "I encountered an issue generating a response.", 'sources': []}
            
            # Ensure required keys exist
            if 'answer' not in result:
                result['answer'] = "I couldn't find a specific answer to your question."
            if 'sources' not in result:
                result['sources'] = []
                
            logging.info(f"[{triage_result.ticket_id}] Response generated with {len(result.get('sources', []))} sources.")
            return result
            
        except Exception as e:
            logging.error(f"[{triage_result.ticket_id}] Error in response generation: {e}")
            return {'answer': "I encountered an issue generating a response.", 'sources': []}

    def _format_final_answer(
        self, 
        triage: TriageResult, 
        validation: ValidationResult, 
        rag: Dict[str, Any]
    ) -> str:
        """Creates a final, user-friendly response string from all pipeline stages."""
        try:
            answer_parts = []

            # 1. Add priority context
            if triage.priority == "P0":
                answer_parts.append("**URGENT ISSUE DETECTED**")
            elif triage.priority == "P1":
                answer_parts.append("⚡ **High Priority Request**")

            # 2. Add the main answer from the RAG system
            rag_answer = rag.get('answer') if rag else None
            if rag_answer and str(rag_answer).strip():  # Only add if not None or empty
                answer_parts.append(str(rag_answer))
            else:
                answer_parts.append("I couldn't find a specific answer, but I can offer some general guidance.")

            # 3. Add sources if available
            sources = rag.get('sources', []) if rag else []
            if sources:
                valid_sources = [str(src) for src in sources if src and str(src).strip()]
                if valid_sources:
                    source_list = "\n".join([f"- {src}" for src in valid_sources])
                    answer_parts.append(f"\n📚 **Here are some relevant documents:**\n{source_list}")
            
            # 4. Add a confidence note if validation was low
            if validation and (validation.confidence == 'low' or validation.status == 'disagreement'):
                answer_parts.append("\n⚠️ *Our system had low confidence in categorizing this query, so the information above might be general. If this doesn't solve your issue, please provide more details.*")

            # Filter out any None or empty strings before joining
            valid_answer_parts = []
            for part in answer_parts:
                if part is not None:
                    part_str = str(part).strip()
                    if part_str:
                        valid_answer_parts.append(part_str)
            
            if not valid_answer_parts:
                return "I apologize, but I encountered an issue generating a response. Please try again or contact support."
            
            return "\n\n".join(valid_answer_parts)
            
        except Exception as e:
            logging.error(f"Error formatting final answer: {e}")
            return "I apologize, but I encountered an issue generating a response. Please try again or contact support."


if __name__ == '__main__':
    # --- DEMO ---
    
    # In a real app, you'd load this from a database or file.
    sample_historical_tickets = [
        {'id': 'hist-001', 'body': 'how to setup sso with okta', 'topics': ['SSO']},
        {'id': 'hist-002', 'body': 'api is giving 403 error', 'topics': ['API/SDK']},
        {'id': 'hist-003', 'body': 'cannot connect to snowflake warehouse', 'topics': ['Connector']}
    ]

    # Initialize the pipeline
    pipeline = ChatbotPipeline(historical_data=sample_historical_tickets)

    # Define some test queries
    test_queries = [
        "How do I set up SSO with Azure AD? I'm completely blocked.",
        "What is the best practice for organizing our data glossary?",
        "My Snowflake connection keeps failing with an authentication error."
    ]
    
    # Run the pipeline for each query
    for i, query in enumerate(test_queries):
        print(f"\n{'='*25} RUNNING TEST QUERY #{i+1} {'='*25}")
        print(f"Query: {query}\n")
        
        result = pipeline.run(query)
        
        print(f"FINAL ANSWER:\n\n{result.final_answer}")
        print(f"\n--- PIPELINE METADATA ---")
        print(f"  - Triage Topics: {result.triage.topics}")
        print(f"  - Triage Priority: {result.triage.priority}")
        print(f"  - Validation Status: {result.validation.status} ({result.validation.confidence} confidence)")
        print(f"  - RAG Sources Found: {len(result.rag_response.get('sources', []))}")
        print(f"  - Total Time: {result.processing_time:.2f} seconds")
        print(f"{'='*70}\n")