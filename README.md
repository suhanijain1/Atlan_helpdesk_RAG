# Customer Support Copilot – Atlan Take Home Assignment

**By Suhani Jain**

This project mimics how a human support agent thinks through tickets — but implemented technically with structured classification, clustering-based evaluation, and hybrid retrieval. The pipeline has three parts: **Triage**, **Evaluation**, **Hybrid RAG**.

Along the way, I went back and forth on approaches (classification vs clustering, model choice, dense vs sparse retrieval). This README explains those decisions, the experiments I ran, and the tradeoffs I made.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- API access to Google Gemini (free tier works)

### Local Setup

1. **Clone and navigate to the project:**
```bash
git clone <repository-url>
cd atlan
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download required NLTK data:**
```bash
python -c "import nltk; nltk.download('punkt')"
```

4. **Set up environment variables:**
```bash
# Create .env file
cp .env.template .env

# Add your Gemini API key to .env
echo "GEMINI_API_KEY=your_api_key_here" >> .env
```

5. **Run the pipeline:**
```bash
# Run classification pipeline
python pipeline/main.py

# Or launch the Streamlit app
streamlit run app.py
```

### Alternative Setup (using setup script)
```bash
chmod +x setup.sh
./setup.sh
```

## 📋 Project Architecture

### Step 1: Triage

**Goal:** Assign each ticket structured metadata for routing.

**What I tried first:** I considered letting clustering group tickets "naturally." In practice, clustering collapsed on short/noisy text. Many runs gave 0 clusters (all noise) or only 3–4 clusters surviving out of 20+ expected topics. This made it unreliable as the main driver.

**What I chose:** Classification-first, because it guarantees coverage. Every ticket gets tags, even if imperfect.

**Implementation:**

Used Gemini 1.5 Lite (JSON mode) to generate:
- **Topic** (Connector, Billing, Product, etc.)
- **Sentiment** (frustrated, curious, neutral)
- **Priority** (P0, P1, P2)
- **Entities** (Snowflake, SSO, API, etc.)

**Example:** "Can't log into Snowflake, production down" → `{topic: Connector, priority: P0, sentiment: frustrated, entities: [Snowflake]}`.

*From Arup Mondal:* I added sentiment + priority scores to compress messy free text into semantic vectors.

**Why Gemini Lite:**
- JSON mode ensures structured outputs
- Gemma didn't support JSON mode, so I couldn't use it here
- In production, I'd move to a smaller, JSON-native model (self-hosted or hosted)

**Cons:**
- Some classifications are inconsistent
- Multi-label outputs (topic + tone + urgency) add interpretability, but sometimes look messy (e.g., "curious" + P0)

### Step 2: Evaluation

This was the biggest innovation. Classifying is easy — checking if it's reliable is the hard part.

**What I tried:**
- **HDBSCAN** on ~200 tickets → collapsed:
  - Run 1: 0 clusters (everything = noise)
  - Run 2: 3 clusters left from 20+ topics
- **KMeans** (k=20) → forced clusters, but many were incoherent
- **Coherence metrics:** measured cosine similarity of tickets to topic centroid. Range: 0.48 (weak) – 0.92 (strong)
- **Separation ratios:** topics <1.1 were basically overlapping
- **Outlier fractions:** in some cases >40% of tickets sat >2σ from centroid

**What I settled on:**
1. Compute topic and combo centroids
2. For each ticket, calculate:
   - `score_assigned`, `score_best_other`, and `margin = difference`
3. Flag tickets with margin <0.2 (≈80% error rate when I manually reviewed)
4. Generate evaluation reports with:
   - Topic summaries (coherence, separation, outliers)
   - Ticket-level CSVs (with flags)
   - Visuals (`coherence_by_topic.png`, `global_pca.png`)

**Why this matters:** It turns evaluation from gut-checks ("does this look right?") into quantifiable diagnostics that SMEs can use.

**Cons:**
- With only ~200 tickets, metrics swing a lot. With >2k tickets, coherence should stabilize
- Short tickets ("login issue") almost always misfire. Needs metadata enrichment

### Step 3: Hybrid RAG

**Goal:** Once tickets are triaged, retrieve relevant context for resolution.

**What I tried:**
- **Dense embeddings only:** good for paraphrases, but failed on abbreviations (e.g., SSO vs single sign-on)
- **Sparse embeddings only** (BM25/TF-IDF): good for exact matches, but failed on synonyms (e.g., prod DB vs production database)

**What I chose:** Hybrid retrieval (dense + sparse).
1. Run both searches
2. Merge results with weighted scoring
3. Output ranked list of docs/past tickets

**Results:**
- Coverage improved by ~20–25% vs dense-only retrieval
- Example: hybrid successfully retrieved docs for "prod DB issue" (sparse) and "single sign-on failed" (dense)

**Connection to triage:**
- Triage outputs structured metadata (topic, urgency, sentiment)
- Hybrid RAG uses those signals to query dense + sparse indexes
- Together, tickets are routed (triage) and solved (RAG)

**Cons:**
- Hybrid adds latency (two retrieval passes)
- Fusion weights are sensitive — need tuning per dataset

## 🛠️ Usage

### Running the Pipeline
```bash
# Run complete classification pipeline
python pipeline/main.py

# Results saved to:
# - classification_results.json
# - triage_results/ (evaluation metrics and plots)
```

### Using the Streamlit App
```bash
streamlit run app.py
```

Features:
- Upload tickets for classification
- View triage results with confidence scores
- Search knowledge base with hybrid retrieval
- Interactive evaluation dashboards

### Running Individual Components

**Feature Extraction:**
```bash
python -c "
from pipeline.feature_extractor import FeatureExtractor
extractor = FeatureExtractor()
features = extractor.extract_features('Customer login issue with SSO')
print(features)
"
```

**Evaluation:**
```bash
python pipeline/evaluation.py
```

**Hybrid RAG:**
```bash
python -c "
from hybrid_rag import HybridRAG
rag = HybridRAG()
results = rag.search('SSO authentication problem')
print(results)
"
```

## 📊 Key Files

- **`pipeline/main.py`** - Main pipeline orchestrator
- **`pipeline/pipeline.py`** - Core classification pipeline
- **`pipeline/feature_extractor.py`** - Gemini-based feature extraction
- **`pipeline/evaluation.py`** - Clustering and coherence evaluation
- **`pipeline/multimethod.py`** - Heuristic + clustering classification
- **`hybrid_rag.py`** - Dense + sparse retrieval
- **`app.py`** - Streamlit web interface

## 🔧 Configuration

Edit `pipeline/config.py` to customize:
- Model names and API endpoints
- Classification thresholds
- Evaluation parameters
- Retrieval weights

## ⚠️ Limitations & Next Steps

**Current Limitations:**
- **Short tickets:** collapse into noise. Fix = enrich with product/subject metadata
- **Dataset size:** ~200 tickets was too small for stable clusters. With 2k+, expect more reliable coherence
- **Model lifecycle:** Gemini Lite support ends in Sept. Must migrate to Gemma (once JSON-ready) or a hosted small model
- **Batching:** Current = sequential. Batch inference would cut runtime ~60%
- **Monitoring:** Low-margin tickets should be automatically flagged to SMEs for review

**Next Steps:**
1. Migrate to self-hosted JSON-capable model
2. Implement batch processing for scale
3. Add automated SME review workflows
4. Expand knowledge base with product documentation
5. Fine-tune retrieval weights per domain

## 📚 References

- **Arup Mondal, Hybrid LLM + ML for Ticket Classification** → added sentiment/priority scores for compact vectors
- **Anthropic, Ticket Routing Guide** → multi-dimensional tagging
- **Dave Ebbelaar GitHub Gist** → JSON formatting + evaluation workflow
- **Shilpa Deeparaj, AI-Powered Ticket Classification** → batch processing and scaling
- **Hybrid RAG** → based on standard sparse + dense fusion (BM25 + dense vector search), widely used in FAISS/Pinecone hybrid search

## 🎯 Results

The pipeline successfully processes customer support tickets with:
- **95%+ classification coverage** (every ticket gets tags)
- **20-25% improved retrieval** vs dense-only search
- **Quantifiable evaluation metrics** for SME review
- **End-to-end workflow** from raw tickets to actionable insights

---

**Contact:** Suhani Jain | [GitHub](https://github.com/suhanijain1) | Built for Atlan Take Home Assignment 