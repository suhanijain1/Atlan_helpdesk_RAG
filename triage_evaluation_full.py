#!/usr/bin/env python3
"""
triage_evaluation_full.py
Full triage evaluation pipeline (production-ready).

Usage:
    python triage_evaluation_full.py --embeddings embeddings.npz --metadata metadata.csv --tickets sample_tickets.json

The script will:
 - load embeddings (npy / npz) aligned with metadata CSV (id,text)
 - if Gemini predictions exist in a provided JSON or metadata contains 'predicted_topics', use those as evaluation groups (multi-label)
 - otherwise infer clusters using HDBSCAN (if installed) or KMeans fallback
 - compute centroids for topics and frequent combos (exact multi-label sets)
 - compute per-topic & per-combo metrics: coherence, outlier_frac, sep_ratio
 - compute per-ticket proxies: score_assigned, score_best_other, margin, recall_proxy
 - flag tickets/topics with low confidence via thresholds
 - save CSVs and PNGs to same directory, and print a concise report to stdout

Outputs:
 - tickets_with_metrics.csv
 - topic_summary.csv
 - combo_summary.csv
 - sme_review.csv (topic -> example texts + top terms)
 - ticket_metrics.csv
 - plots: coherence_by_topic.png, global_pca.png
"""
import os
import sys
import argparse
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from textwrap import shorten

plt.rcParams["figure.dpi"] = 120

# ---------------- CONFIG (tweakable) ----------------
MIN_TOPIC_SIZE = 5        # min size to call a topic "valid" for certain metrics
MIN_COMBO_SIZE = 5        # min size to compute combo-centroid checks
OUTLIER_Z = 3.5
MARGIN_LOW_THRESH = 0.10
MARGIN_HIGH_THRESH = 0.30
TFIDF_MAX_FEATURES = 2000
SVD_COMPONENTS = 50
# ---------------------------------------------------

def load_embeddings(path_emb):
    if path_emb is None:
        return None
    if path_emb.endswith('.npz'):
        arr = np.load(path_emb, allow_pickle=True)
        if 'embeddings' in arr:
            emb = arr['embeddings']
        else:
            keys = [k for k in arr.files]
            emb = arr[keys[0]]
    else:
        emb = np.load(path_emb)
    return emb

def l2norm_rows(X, eps=1e-12):
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = eps
    return X / norms

def mean_offdiag_cosine_from_array(embs):
    if embs.shape[0] < 2:
        return float('nan')
    sims = cosine_similarity(embs)
    iu = np.triu_indices(sims.shape[0], k=1)
    return float(sims[iu].mean())

def detect_outliers_mad_from_array(embs, z_thresh=OUTLIER_Z):
    if embs.shape[0] < 3:
        return []
    centroid = embs.mean(axis=0)
    dists = np.linalg.norm(embs - centroid, axis=1)
    med = np.median(dists)
    mad = median_abs_deviation(dists, scale='normal')
    if mad == 0:
        return []
    z = (dists - med) / (mad + 1e-12)
    return [i for i, v in enumerate(z) if v > z_thresh]

def compute_topic_centroids_from_preds(df, emb_norm):
    topic_to_idxs = defaultdict(list)
    for i, tlist in enumerate(df['pred_topics']):
        for t in tlist:
            topic_to_idxs[t].append(i)
    centroids = {}
    for t, idxs in topic_to_idxs.items():
        if len(idxs) == 0:
            continue
        c = emb_norm[idxs].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        centroids[t] = c
    return topic_to_idxs, centroids

def compute_combo_centroids(df, emb_norm, min_combo_size=MIN_COMBO_SIZE):
    combo_counts = Counter(df['pred_topics'])
    combo_centroids = {}
    for combo, cnt in combo_counts.items():
        if cnt >= min_combo_size and combo != tuple():
            idxs = [i for i, c in enumerate(df['pred_topics']) if c == combo]
            if len(idxs) >= min_combo_size:
                cc = emb_norm[idxs].mean(axis=0)
                combo_centroids[combo] = cc / (np.linalg.norm(cc) + 1e-12)
    return combo_centroids

def infer_clusters_if_needed(df, emb_norm):
    has_preds = df['pred_topics'].apply(lambda x: len(x) > 0).any()
    if has_preds:
        print("Found predicted topics in data; skipping cluster inference.")
        return df, True, None, None

    # Try HDBSCAN first (if available)
    try:
        import hdbscan
        # For small datasets (< 50), use smaller min_cluster_size
        if len(df) < 50:
            min_cluster = max(2, len(df) // 10)  # More permissive for small datasets
        else:
            min_cluster = max(3, min(8, max(3, len(df) // 6)))
        
        print(f"Using HDBSCAN with min_cluster_size={min_cluster} for {len(df)} tickets")
        clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean', cluster_selection_method='eom')
        labs = clusterer.fit_predict(emb_norm)
        df['pred_topics'] = df.index.map(lambda i: (f"cluster_{int(labs[i])}",))
        df['inferred_label'] = labs
        n_clusters = len(set(labs)) - (1 if -1 in labs else 0)
        print(f"HDBSCAN produced {n_clusters} clusters; noise labeled as -1.")
        
        # If HDBSCAN still produces no clusters, fall back to KMeans
        if n_clusters == 0:
            print("HDBSCAN found no clusters, falling back to KMeans.")
            raise Exception("No clusters found")
            
        return df, False, labs, "hdbscan"
    except Exception:
        print("HDBSCAN not available or failed; falling back to KMeans.")

    # KMeans fallback: choose k by silhouette
    best_k = None
    best_score = -999
    max_k = min(12, max(2, len(df) - 1))
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=0, n_init=10)
        labs_try = km.fit_predict(emb_norm)
        try:
            s = silhouette_score(emb_norm, labs_try, metric='cosine')
        except Exception:
            s = silhouette_score(emb_norm, labs_try, metric='euclidean')
        if s > best_score + 1e-9:
            best_score = s
            best_k = k
    km = KMeans(n_clusters=best_k, random_state=0, n_init=20)
    labs = km.fit_predict(emb_norm)
    df['pred_topics'] = df.index.map(lambda i: (f"cluster_{int(labs[i])}",))
    df['inferred_label'] = labs
    print(f"KMeans fallback produced k={best_k} clusters (silhouette={best_score:.3f}).")
    return df, False, labs, f"kmeans_k{best_k}"

def build_topic_and_combo_metrics(df, emb_norm, centroids, combo_centroids):
    rows = []
    topic_list = list(centroids.keys())
    for t, c in centroids.items():
        idxs = [i for i, preds in enumerate(df['pred_topics']) if t in preds]
        size = len(idxs)
        if size < 2:
            coh = float('nan')
            out_frac = float('nan')
            sep = float('nan')
        else:
            sub = emb_norm[idxs]
            coh = mean_offdiag_cosine_from_array(sub)
            outs = detect_outliers_mad_from_array(sub)
            out_frac = len(outs) / size
            other_centroids = [centroids[u] for u in topic_list if u != t]
            if len(other_centroids) == 0:
                sep = float('nan')
            else:
                other_centroids = np.vstack(other_centroids)
                sims_to_others = sub.dot(other_centroids.T).max(axis=1)
                mean_inter = float(sims_to_others.mean())
                sep = float(coh / (mean_inter + 1e-12))
        rows.append({'topic': t, 'size': size, 'coherence': coh, 'outlier_frac': out_frac, 'sep_ratio': sep})
    topic_summary = pd.DataFrame(rows).sort_values(['size', 'coherence'], ascending=[False, False])

    combo_rows = []
    combo_counts = Counter(df['pred_topics'])
    for combo, cnt in combo_counts.items():
        idxs = [i for i, c in enumerate(df['pred_topics']) if c == combo]
        if len(idxs) < MIN_COMBO_SIZE:
            combo_rows.append({'combo': str(combo), 'size': len(idxs), 'coherence': float('nan'), 'outlier_frac': float('nan')})
            continue
        sub = emb_norm[idxs]
        coh = mean_offdiag_cosine_from_array(sub)
        outs = detect_outliers_mad_from_array(sub)
        combo_rows.append({'combo': str(combo), 'size': len(idxs), 'coherence': coh, 'outlier_frac': len(outs) / len(idxs)})
    combo_summary = pd.DataFrame(combo_rows).sort_values(['size', 'coherence'], ascending=[False, False])

    return topic_summary, combo_summary

def compute_ticket_proxies(df, emb_norm, centroids, combo_centroids):
    topic_list = list(centroids.keys())
    if len(topic_list) > 0:
        Cmat = np.vstack([centroids[t] for t in topic_list])
        S = emb_norm.dot(Cmat.T)
    else:
        S = np.zeros((len(df), 0))
    ticket_rows = []
    for i, preds in enumerate(df['pred_topics']):
        if (preds == tuple()) or (len(preds) == 0):
            ticket_rows.append({'idx': i, 'assigned': (), 'score_assigned': float('nan'),
                                'score_best_other': float('nan'), 'margin': float('nan'),
                                'recall_proxy': float('nan'), 'used_combo_centroid': False})
            continue
        if preds in combo_centroids:
            sc = float(emb_norm[i].dot(combo_centroids[preds]))
            best_other = float(S[i].max()) if S.shape[1] > 0 else -1.0
            margin = sc - best_other
            ticket_rows.append({'idx': i, 'assigned': preds, 'score_assigned': sc,
                                'score_best_other': best_other, 'margin': margin,
                                'recall_proxy': float('nan'), 'used_combo_centroid': True})
            continue
        assigned_idxs = [topic_list.index(t) for t in preds if t in topic_list]
        if len(assigned_idxs) == 0:
            ticket_rows.append({'idx': i, 'assigned': preds, 'score_assigned': float('nan'),
                                'score_best_other': float('nan'), 'margin': float('nan'),
                                'recall_proxy': float('nan'), 'used_combo_centroid': False})
            continue
        score_assigned = float(S[i, assigned_idxs].mean())
        other_mask = np.ones(S.shape[1], dtype=bool)
        other_mask[assigned_idxs] = False
        score_best_other = float(S[i, other_mask].max()) if other_mask.any() else -1.0
        margin = float(score_assigned - score_best_other)
        k = max(1, len(assigned_idxs))
        topk = np.argsort(S[i])[-k:]
        recall = len(set(topk).intersection(set(assigned_idxs))) / float(k)
        ticket_rows.append({'idx': i, 'assigned': preds, 'score_assigned': score_assigned,
                            'score_best_other': score_best_other, 'margin': margin,
                            'recall_proxy': recall, 'used_combo_centroid': False})
    ticket_df = pd.DataFrame(ticket_rows).set_index('idx')
    return ticket_df

def tag_confidence(ticket_df, low=MARGIN_LOW_THRESH, high=MARGIN_HIGH_THRESH):
    def lab(m):
        if pd.isna(m):
            return "unknown"
        if m <= low:
            return "LOW_CONF"
        if m >= high:
            return "HIGH_CONF"
        return "MED_CONF"
    return ticket_df['margin'].apply(lab)

def get_top_terms_per_topic(df, top_n=6):
    texts = df['text'].fillna("").tolist()
    vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
    X = vec.fit_transform(texts).toarray()
    terms = vec.get_feature_names_out()
    topic_terms = {}
    topic_to_idxs = defaultdict(list)
    for i, preds in enumerate(df['pred_topics']):
        for t in preds:
            topic_to_idxs[t].append(i)
    for t, idxs in topic_to_idxs.items():
        if len(idxs) == 0:
            topic_terms[t] = []
            continue
        centroid = X[idxs].mean(axis=0)
        top_idx = centroid.argsort()[::-1][:top_n]
        topic_terms[t] = [terms[i] for i in top_idx if centroid[i] > 0][:top_n]
    return topic_terms

def produce_sme_review_csv(df, topic_summary, topic_terms, out_path):
    rows = []
    for _, r in topic_summary.iterrows():
        t = r['topic']
        idxs = [i for i, preds in enumerate(df['pred_topics']) if t in preds]
        examples = [shorten(df.loc[i, 'text'], width=240) for i in idxs[:5]]
        rows.append({'topic': t, 'size': int(r['size']), 'coherence': r['coherence'],
                     'outlier_frac': r['outlier_frac'], 'top_terms': ", ".join(topic_terms.get(t, [])),
                     'examples': " ||| ".join(examples)})
    pd.DataFrame(rows).to_csv(out_path, index=False)

def save_plots(topic_summary, emb_norm, df, out_dir):
    dfc = topic_summary.dropna(subset=['coherence']).sort_values('coherence', ascending=False).head(30)
    if not dfc.empty:
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(dfc)), dfc['coherence'].values)
        plt.xticks(range(len(dfc)), [shorten(str(x), 20) for x in dfc['topic']], rotation=90)
        plt.title('Topic coherence (top topics)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'coherence_by_topic.png'))
        plt.close()
    try:
        pca = PCA(n_components=2)
        pts = pca.fit_transform(emb_norm)
        plt.figure(figsize=(7, 5))
        topics = list(topic_summary['topic'].head(8))
        for t in topics:
            idxs = [i for i, preds in enumerate(df['pred_topics']) if t in preds]
            if len(idxs) == 0:
                continue
            plt.scatter(pts[idxs, 0], pts[idxs, 1], label=f"{t}({len(idxs)})", s=40, alpha=0.8)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
        plt.title('Global PCA of embeddings (sample topics colored)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'global_pca.png'))
        plt.close()
    except Exception as e:
        print("PCA plot failed:", e)

def main(args):
    out_dir = args.outdir or os.path.dirname(args.embeddings) or "/mnt/data"
    os.makedirs(out_dir, exist_ok=True)

    emb = load_embeddings(args.embeddings) if args.embeddings else None
    meta = pd.read_csv(args.metadata) if args.metadata else None

    if emb is None and args.tickets is None:
        print("ERROR: No embeddings provided and no tickets JSON provided.")
        sys.exit(1)

    if emb is None:
        print("No embeddings provided. Building TF-IDF+SVD embeddings from ticket text...")
        tickets = pd.read_json(args.tickets) if args.tickets else pd.read_json(args.metadata)
        texts = tickets['text'].fillna("").tolist()
        vec = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES, stop_words='english')
        X = vec.fit_transform(texts).toarray()
        n_comp = min(SVD_COMPONENTS, max(1, X.shape[1] - 1))
        if n_comp >= 1:
            svd = TruncatedSVD(n_components=n_comp, random_state=0)
            emb = svd.fit_transform(X)
        else:
            emb = X
        meta = tickets[['id', 'text']].copy().reset_index(drop=True)
    else:
        if meta is None and args.tickets:
            meta = pd.read_json(args.tickets)[['id', 'text']].copy().reset_index(drop=True)
        if meta is None:
            print("ERROR: embeddings provided but no metadata given.")
            sys.exit(1)
        if len(meta) != emb.shape[0]:
            print("ERROR: metadata rows != embeddings rows.")
            sys.exit(1)

    # standardize predicted topics column if present
    if 'predicted_topics' in meta.columns:
        meta['pred_topics'] = meta['predicted_topics'].apply(lambda v: tuple(v) if isinstance(v, (list, tuple)) else tuple())
    if 'pred_topics' not in meta.columns:
        if args.tickets:
            tickets = pd.read_json(args.tickets)
            if 'predicted_topics' in tickets.columns:
                meta['pred_topics'] = tickets['predicted_topics'].apply(lambda v: tuple(v) if isinstance(v, (list, tuple)) else tuple())
            else:
                meta['pred_topics'] = meta.apply(lambda r: tuple(), axis=1)
        else:
            meta['pred_topics'] = meta.apply(lambda r: tuple(), axis=1)

    emb_norm = l2norm_rows(emb)
    df = meta.copy().reset_index(drop=True)
    if 'text' not in df.columns:
        df['text'] = df.apply(lambda r: str(r.get('subject', '')) + ' ' + str(r.get('body', '')), axis=1)

    df, had_preds, inferred_labels, clusterer_name = infer_clusters_if_needed(df, emb_norm)

    topic_to_idxs, centroids = compute_topic_centroids_from_preds(df, emb_norm)
    combo_centroids = compute_combo_centroids(df, emb_norm)

    topic_summary, combo_summary = build_topic_and_combo_metrics(df, emb_norm, centroids, combo_centroids)
    ticket_df = compute_ticket_proxies(df, emb_norm, centroids, combo_centroids)
    ticket_df['conf_label'] = tag_confidence(ticket_df)

    topic_terms = get_top_terms_per_topic(df, top_n=6)
    produce_sme_review_csv(df, topic_summary, topic_terms, os.path.join(out_dir, "sme_review.csv"))

    df_out = df.copy()
    df_out = df_out.join(ticket_df.reset_index(), how='left')
    df_out.to_csv(os.path.join(out_dir, "tickets_with_metrics.csv"), index=False)
    topic_summary.to_csv(os.path.join(out_dir, "topic_summary.csv"), index=False)
    combo_summary.to_csv(os.path.join(out_dir, "combo_summary.csv"), index=False)
    ticket_df.to_csv(os.path.join(out_dir, "ticket_metrics.csv"), index=False)

    save_plots(topic_summary, emb_norm, df, out_dir)

    # Print concise report
    print("\n--- Triage Evaluation Report ---\n")
    print("Inputs: embeddings:", args.embeddings, "metadata:", args.metadata, "tickets_json:", args.tickets)
    print("Used clusterer:", clusterer_name, "Had pre-existing preds?:", had_preds)
    print("\nTop topics by size & coherence:\n")
    if not topic_summary.empty:
        print(topic_summary.head(12).to_string(index=False))
    else:
        print("  <no topics found>")

    print("\nTop combos (>=MIN_COMBO_SIZE):\n")
    if not combo_summary.empty:
        print(combo_summary.head(12).to_string(index=False))
    else:
        print("  <no combos found or all combos smaller than MIN_COMBO_SIZE>")

    print("\nTickets saved to:", os.path.join(out_dir, "tickets_with_metrics.csv"))
    print("Topic summary saved to:", os.path.join(out_dir, "topic_summary.csv"))
    print("SME review CSV saved to:", os.path.join(out_dir, "sme_review.csv"))
    print("Ticket metrics saved to:", os.path.join(out_dir, "ticket_metrics.csv"))
    print("Plots saved (if generated): coherence_by_topic.png, global_pca.png\n")

    print("Low-confidence tickets (example):\n")
    if 'conf_label' in df_out.columns:
        low_conf = df_out[df_out['conf_label'] == "LOW_CONF"].copy()
        if low_conf.shape[0] == 0:
            print(" - None (no low-confidence tickets found with current thresholds).")
        else:
            for i, r in low_conf.head(10).iterrows():
                print(" - idx={} assigned={} margin={} text={}".format(i, r['pred_topics'], r.get('margin', None), shorten(r['text'], 160)))
    else:
        print(" - No confidence labels computed.")

    print("\n--- End Report ---\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triage evaluation script")
    parser.add_argument('--embeddings', default='/mnt/data/embeddings.npz', help='Path to embeddings (.npz or .npy)')
    parser.add_argument('--metadata', default='/mnt/data/metadata.csv', help='Metadata CSV with id,text and optional predicted_topics')
    parser.add_argument('--tickets', default='/mnt/data/sample_tickets.json', help='Optional tickets JSON with predicted_topics')
    parser.add_argument('--outdir', default='/mnt/data', help='Output directory')
    args = parser.parse_args()
    main(args)
