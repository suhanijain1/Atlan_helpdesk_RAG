# run_batch_evaluation.py

import os
import pandas as pd
import logging
from triage_evaluation_full import (
    load_embeddings, 
    l2norm_rows,
    compute_topic_centroids_from_preds, 
    compute_combo_centroids,
    build_topic_and_combo_metrics,
    compute_ticket_proxies,
    tag_confidence,
    get_top_terms_per_topic,
    produce_sme_review_csv,
    save_plots
)
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_evaluation(tickets_df: pd.DataFrame, output_dir: str = "triage_results"):
    """
    Runs a full batch evaluation on a set of tickets with pre-assigned topics.

    This function uses the logic from `triage_evaluation_full.py` to:
    1. Generate text embeddings for the tickets.
    2. Analyze the coherence and separation of the predicted topics.
    3. Calculate confidence scores for each ticket's classification.
    4. Save detailed reports and plots to the output directory.

    Args:
        tickets_df (pd.DataFrame): DataFrame with 'body' and 'topics' columns.
        output_dir (str): Directory to save the evaluation artifacts.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Created output directory: {output_dir}")

    # --- 1. Generate Embeddings ---
    logging.info("Generating text embeddings for tickets...")
    texts = tickets_df['body'].tolist()
    # Using the same model as in the triage module for consistency
    embedder = SentenceTransformer('all-MiniLM-L6-v2') 
    embeddings = embedder.encode(texts, show_progress_bar=True)
    emb_norm = l2norm_rows(embeddings)
    logging.info(f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}.")

    # The evaluation script expects a column named 'pred_topics' as tuples
    df = tickets_df.copy()
    df['pred_topics'] = df['topics'].apply(lambda x: tuple(x) if isinstance(x, list) else tuple())
    df['text'] = df['body'] # The script uses a 'text' column

    # --- 2. Compute Centroids and Metrics ---
    logging.info("Computing topic and combo centroids...")
    topic_to_idxs, centroids = compute_topic_centroids_from_preds(df, emb_norm)
    combo_centroids = compute_combo_centroids(df, emb_norm)
    
    logging.info("Building topic and combo metrics (coherence, separation)...")
    topic_summary, combo_summary = build_topic_and_combo_metrics(df, emb_norm, centroids, combo_centroids)

    # --- 3. Compute Per-Ticket Confidence ---
    logging.info("Computing per-ticket confidence scores...")
    ticket_df_metrics = compute_ticket_proxies(df, emb_norm, centroids, combo_centroids)
    ticket_df_metrics['conf_label'] = tag_confidence(ticket_df_metrics)

    # --- 4. Generate Reports and Save Artifacts ---
    logging.info("Generating and saving reports...")
    
    # Merge metrics back into the main dataframe
    df_out = df.join(ticket_df_metrics)
    
    # Define output paths
    tickets_out_path = os.path.join(output_dir, "tickets_with_metrics.csv")
    topic_summary_path = os.path.join(output_dir, "topic_summary.csv")
    sme_review_path = os.path.join(output_dir, "sme_review.csv")
    
    df_out.to_csv(tickets_out_path, index=False)
    topic_summary.to_csv(topic_summary_path, index=False)

    # Generate SME review file with top terms and examples
    topic_terms = get_top_terms_per_topic(df)
    produce_sme_review_csv(df, topic_summary, topic_terms, sme_review_path)

    # Generate plots
    save_plots(topic_summary, emb_norm, df, output_dir)
    
    logging.info(f"Evaluation complete. Reports saved to '{output_dir}'.")
    print("\n--- Evaluation Summary ---")
    print("Top Topics by Coherence:")
    print(topic_summary.sort_values('coherence', ascending=False).head(5).to_string(index=False))
    print("\nLow Confidence Tickets:")
    low_conf_tickets = df_out[df_out['conf_label'] == "LOW_CONF"]
    print(f"Found {len(low_conf_tickets)} low-confidence tickets.")
    print(low_conf_tickets[['body', 'topics', 'margin']].head().to_string())
    print("--------------------------")


if __name__ == '__main__':
    # Load your classified sample tickets. This would typically come from
    # running the TriageClassifier on a batch of unlabeled tickets.
    # For this demo, we'll use a pre-labeled JSON file.
    try:
        # Assuming you have a 'sample_tickets.json' from the project
        # with a structure like: [{"id": "...", "body": "...", "topics": ["..."]}]
        tickets_with_predictions = pd.read_json('sample_tickets.json')
        # The JSON might have a different key for text, let's standardize to 'body'
        if 'text' in tickets_with_predictions.columns and 'body' not in tickets_with_predictions.columns:
            tickets_with_predictions.rename(columns={'text': 'body'}, inplace=True)
            
    except FileNotFoundError:
        print("file not found. Please provide a valid 'sample_tickets.json' file.")
    # Run the full evaluation
    run_evaluation(tickets_with_predictions, output_dir="triage_evaluation_results")