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

    def run(self, query: str, ticket_id: str = None) -> PipelineResult:
        """
        Executes the full chatbot pipeline for a given query, now with conditional RAG logic.
        """
        start_time = time.time()
        if not ticket_id:
            ticket_id = f"TICKET-{int(time.time())}"
        
        logging.info(f"Processing query for ticket [{ticket_id}]...")
        
        # Stage 1 & 2: Triage and Validate (These always run)
        triage_result = self._stage_triage(query, ticket_id)
        validation_result = self._stage_validate(triage_result)
        
        # Define topics that should trigger a full RAG response
        RAG_TOPICS = {"How-to", "Product", "Best practices", "API/SDK", "SSO"}
        
        # NEW LOGIC: Check if the primary topic requires a RAG response
        primary_topic = triage_result.topics[0] if triage_result.topics else "Other"
        
        if primary_topic in RAG_TOPICS:
            logging.info(f"[{ticket_id}] Topic '{primary_topic}' requires RAG. Generating direct answer.")
            # Stage 3: Generate a response using the RAG system
            rag_response = self._stage_respond(query, triage_result)
            # Stage 4: Format the final, user-facing answer
            final_answer = self._format_final_answer(triage_result, validation_result, rag_response)
        else:
            logging.info(f"[{ticket_id}] Topic '{primary_topic}' does not require RAG. Routing ticket.")
            # Create a simple routing message for non-RAG topics
            rag_response = {} # No RAG response
            final_answer = (
                f"Thank you for your query. Your ticket has been classified as a "
                f"'{primary_topic}' issue and has been routed to the appropriate team for review."
            )

        total_time = time.time() - start_time
        logging.info(f"Pipeline finished for ticket [{ticket_id}] in {total_time:.2f}s.")

        return PipelineResult(
            query=query,
            triage=triage_result,
            validation=validation_result,
            rag_response=rag_response,
            final_answer=final_answer,
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
        """Calls the AgenticHybridRAG module."""
        logging.info(f"[{triage_result.ticket_id}] Stage 3: Response Generation...")
        # The RAG system can use the classification as context
        classification_context = {
            "topic": triage_result.topics[0] if triage_result.topics else "Other",
            "priority": triage_result.priority
        }
        result = self.rag_system.answer_question(query, classification=classification_context)
        logging.info(f"[{triage_result.ticket_id}] Response generated with {len(result.get('sources', []))} sources.")
        return result

    def _format_final_answer(
        self, 
        triage: TriageResult, 
        validation: ValidationResult, 
        rag: Dict[str, Any]
    ) -> str:
        """Creates a final, user-friendly response string from all pipeline stages."""
        answer_parts = []

        # 1. Add priority context
        if triage.priority == "P0":
            answer_parts.append("**URGENT ISSUE DETECTED**")
        elif triage.priority == "P1":
            answer_parts.append("⚡ **High Priority Request**")

        # 2. Add the main answer from the RAG system
        answer_parts.append(rag.get('answer', "I couldn't find a specific answer, but I can offer some general guidance."))

        # 3. Add sources if available
        sources = rag.get('sources', [])
        if sources:
            source_list = "\n".join([f"- {src}" for src in sources])
            answer_parts.append(f"\n📚 **Here are some relevant documents:**\n{source_list}")
        
        # 4. Add a confidence note if validation was low
        if validation.confidence == 'low' or validation.status == 'disagreement':
            answer_parts.append("\n⚠️ *Our system had low confidence in categorizing this query, so the information above might be general. If this doesn't solve your issue, please provide more details.*")

        return "\n\n".join(answer_parts)


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
        print(f"❓ Query: {query}\n")
        
        result = pipeline.run(query)
        
        print(f"✅ FINAL ANSWER:\n\n{result.final_answer}")
        print(f"\n--- PIPELINE METADATA ---")
        print(f"  - Triage Topics: {result.triage.topics}")
        print(f"  - Triage Priority: {result.triage.priority}")
        print(f"  - Validation Status: {result.validation.status} ({result.validation.confidence} confidence)")
        print(f"  - RAG Sources Found: {len(result.rag_response.get('sources', []))}")
        print(f"  - Total Time: {result.processing_time:.2f} seconds")
        print(f"{'='*70}\n")