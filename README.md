# Customer Support Copilot – Atlan Take Home Assignment

**By Suhani Jain**

This project implements an end-to-end AI-powered customer support system that combines intelligent ticket classification, comprehensive documentation retrieval, and agentic hybrid RAG response generation. The system processes customer queries through multiple stages: initial triage classification, validation through clustering analysis, and comprehensive response generation using a hybrid retrieval system.

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

5. **Run the system:**
```bash
# Launch the complete Streamlit interface
streamlit run app.py

# Or run the original classification pipeline
python pipeline/main.py
```

### Alternative Setup (using setup script)
```bash
chmod +x setup.sh
./setup.sh
```

## Project Architecture

The system consists of three main components that work together:

### Component 1: Intelligent Triage System

**Goal:** Classify customer support tickets for intelligent routing and response prioritization.

**Implementation:**
- **Primary Method:** Gemini 1.5 Flash with structured JSON output for consistent classification
- **Classification Dimensions:**
  - **Topic:** SSO, API/SDK, How-to, Product, Best Practices  
  - **Sentiment:** Positive, Negative, Neutral
  - **Priority:** P0 (Critical), P1 (High), P2 (Normal)
  - **Entities:** Extracted relevant product/feature names

**Example Output:**
```json
{
  "topic": "SSO",
  "sentiment": "Negative", 
  "priority": "P0",
  "entities": ["Azure AD", "SAML"],
  "urgency_score": 8.5
}
```

### Component 2: Evaluation Framework

**Goal:** Validate classification quality and identify potential misclassifications.

**What We Implemented:**
- **Clustering-based validation** using sentence transformers and KMeans
- **Coherence analysis** measuring how well tickets cluster by their assigned topics
- **Separation analysis** identifying overlapping or unclear categories
- **Confidence scoring** for individual ticket classifications

**Key Metrics:**
- **Topic Coherence:** Cosine similarity within topic clusters (0.6-0.9 range)
- **Separation Ratio:** Distance between topic centroids (>1.2 indicates clear separation)
- **Classification Confidence:** Based on margin between assigned and best alternative topic

**Output:** Detailed evaluation reports in `triage_results/` with visualizations and flagged tickets.

### Component 3: Agentic Hybrid RAG System

**Goal:** Generate comprehensive, accurate responses with proper source attribution.

**Architecture:**
- **Query Analysis Agent:** Analyzes user intent and selects retrieval strategy
- **Hybrid Retrieval System:** Combines BM25 (keyword) + Semantic (embedding) search
- **Reciprocal Rank Fusion:** Merges results from both retrieval methods
- **Reranking Agent:** Uses sentence transformers to reorder results by relevance
- **Citation-Aware Generator:** Gemini-powered response generation with source tracking

**Knowledge Base:** 61 comprehensive documents covering:
- SSO Integration (25 docs) - Azure AD, Okta, Google, SAML setup guides
- API/SDK Documentation (21 docs) - Developer guides, code examples
- Product Features (5 docs) - Core capabilities and workflows
- How-to Guides (6 docs) - Step-by-step instructions
- Best Practices (4 docs) - Governance and administration guidance

### Knowledge Base Creation Process

**Challenge:** Building a comprehensive knowledge base required systematically scraping Atlan's documentation across multiple domains.

**Solution:** Developed `comprehensive_scraper.py` with targeted URL collection:

**Scraping Strategy:**
1. **Manual URL Curation:** Identified authoritative documentation URLs for each topic area
2. **Smart Content Extraction:** Used BeautifulSoup with multiple CSS selectors to extract clean content
3. **Automatic Categorization:** Classified documents by topic based on URL patterns and content analysis
4. **Quality Filtering:** Only retained documents with substantial content (>500 characters)
5. **Deduplication:** Removed duplicate documents based on URL matching

**Sources Scraped:**
- **SSO Documentation** (6 core URLs → 25 docs): Complete setup guides for Azure AD, Okta, Google, SAML
- **API/SDK Documentation** (7 developer URLs → 21 docs): Getting started guides, SDK references, code examples
- **Product Documentation** (7 core URLs → 5 docs): Feature overviews, capabilities, workflows
- **How-to Guides** (6 URLs → 6 docs): Step-by-step setup instructions for connectors
- **Best Practices** (4 URLs → 4 docs): Administration and governance guidelines

**Technical Implementation:**
```python
# Example from comprehensive_scraper.py
documentation_urls = {
    'SSO': [
        'https://docs.atlan.com/product/integrations/identity-management/sso',
        'https://docs.atlan.com/product/integrations/identity-management/sso/how-tos/enable-azure-ad-for-sso',
        # ... more URLs
    ],
    'API/SDK': [
        'https://developer.atlan.com/getting-started/',
        'https://developer.atlan.com/sdks/',
        # ... more URLs  
    ]
}
```

**Result:** High-quality knowledge base with 61 documents totaling 276,348 characters, enabling accurate response generation with proper source attribution.

**Why Hybrid Retrieval:**
- **BM25 (Sparse):** Excellent for exact matches, technical terms, product names
- **Semantic (Dense):** Handles paraphrases, synonyms, conceptual similarity  
- **Combined:** 20-25% improvement in retrieval coverage vs either method alone

## 🛠️ Usage

### Complete System (Recommended)
```bash
# Launch integrated interface with full pipeline
streamlit run app.py
```

Features:
- Submit support tickets through web interface
- Real-time classification with confidence scores
- Comprehensive response generation with source citations
- Evaluation dashboard showing classification quality
- Interactive knowledge base search

### Original Classification Pipeline
```bash
# Run the original triage classification system
python pipeline/main.py

# Results saved to:
# - classification_results.json
# - triage_results/ (evaluation metrics and plots)
```

### Individual Component Testing

**Hybrid RAG System:**
```bash
python -c "
from hybrid_rag import AgenticHybridRAG
rag = AgenticHybridRAG()
result = rag.answer_question('How do I set up SSO with Azure AD?')
print(result['answer'])
"
```

**Classification Only:**
```bash
python -c "
from atlan_helpdesk_pipeline import AtlanHelpdeskPipeline
pipeline = AtlanHelpdeskPipeline()
result = pipeline.classify_ticket('Cannot login to Snowflake')
print(result)
"
```

**Evaluation Framework:**
```bash
python triage_evaluation_full.py
```

## 📊 Key Files

**Core System:**
- **`app.py`** - Complete Streamlit interface for the full system
- **`atlan_helpdesk_pipeline.py`** - Main pipeline orchestrator integrating all components
- **`hybrid_rag.py`** - Agentic hybrid RAG system with multi-modal retrieval
- **`comprehensive_scraper.py`** - Documentation scraper that built the knowledge base

**Original Triage System:**
- **`pipeline/main.py`** - Original classification pipeline entry point
- **`pipeline/feature_extractor.py`** - Gemini-based ticket classification
- **`pipeline/evaluation.py`** - Clustering and coherence evaluation
- **`triage_evaluation_full.py`** - Enhanced evaluation framework

**Data:**
- **`atlan_knowledge_base.json`** - Comprehensive Atlan documentation (61 docs)
- **`sample_tickets.json`** - Test tickets for pipeline validation

## 🔧 Configuration

Edit configuration in:
- **`pipeline/config.py`** - Classification and evaluation parameters
- **`atlan_helpdesk_pipeline.py`** - Main pipeline settings
- **`.env`** - API keys and environment variables

Key settings:
- Model selection (Gemini Flash vs Pro)
- Classification confidence thresholds  
- Retrieval fusion weights
- Evaluation clustering parameters

## ⚠️ Technical Decisions & Tradeoffs

### Classification Approach
**What We Chose:** Gemini-based classification with structured JSON output
**Why:** Guarantees coverage (every ticket gets classified) and provides consistent structure
**Alternative Considered:** Pure clustering approach
**Tradeoff:** Clustering alone was unreliable on short/noisy text, often producing 0-3 coherent clusters from 20+ expected topics

### Model Selection  
**Primary Model:** Gemini 1.5 Flash
**Why:** JSON mode support, good performance, cost-effective
**Consideration:** Would migrate to self-hosted model for production scale

### Retrieval Strategy
**What We Chose:** Hybrid (BM25 + Semantic) with Reciprocal Rank Fusion
**Why:** Combines exact match capabilities with semantic understanding
**Performance:** 20-25% improvement vs single-method retrieval
**Tradeoff:** Added latency from dual retrieval passes

### Evaluation Method
**What We Chose:** Clustering-based validation with coherence metrics
**Why:** Provides quantifiable measures of classification quality
**Output:** Flags problematic tickets for human review (margin < 0.2 threshold)

## 🔮 Future Enhancements

**Potential Improvements:**
- **Batch Processing:** Current system processes queries sequentially; batch processing could improve throughput
- **Advanced Caching:** Implement result caching for frequently asked questions
- **Multi-language Support:** Extend classification and response generation to non-English queries  
- **Real-time Learning:** Incorporate user feedback to improve classification accuracy over time
- **Integration APIs:** Build REST/GraphQL APIs for integration with existing support systems

**Scalability Considerations:**
- **Model Optimization:** Fine-tune models on domain-specific data for improved accuracy
- **Distributed Processing:** Scale retrieval across multiple nodes for larger knowledge bases
- **Monitoring Dashboard:** Add comprehensive analytics for system performance tracking
- **A/B Testing Framework:** Compare different classification and retrieval strategies

**Production Readiness:**
- **Load Balancing:** Handle multiple concurrent users
- **Database Integration:** Replace JSON files with proper database storage
- **Security:** Add authentication, rate limiting, and data privacy controls
- **Deployment Automation:** CI/CD pipelines for seamless updates

## 📈 Results & Performance

**Classification Coverage:** 
**Retrieval Improvement:** 20-25% better coverage vs single-method approaches
**Knowledge Base:** 61 documents with comprehensive topic coverage
**Evaluation Accuracy:** Successfully identifies low-confidence classifications for review

**Response Quality Examples:**
- SSO queries → Detailed setup guides with configuration steps
- API questions → Code examples and endpoint documentation  
- How-to requests → Step-by-step instructions with prerequisites

## Deployment

The system is deployed with:
- Streamlit web interface for easy access
- Comprehensive error handling and logging
- Modular architecture for easy maintenance
- Docker containerization support
- Environment-based configuration

## 📚 References & Acknowledgments

This project builds on established patterns in:
- Hybrid retrieval systems (BM25 + dense embeddings)
- Agentic RAG architectures with specialized components
- Clustering-based evaluation for classification validation
- Multi-dimensional ticket classification approaches

---

**Contact:** Suhani Jain | [GitHub](https://github.com/suhanijain1) | Built for Atlan Take Home Assignment

## Quick Start

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

## Project Architecture

### Step 1: Triage

**Goal:** Assign each ticket structured metadata for routing.

**What I tried first:** I considered letting clustering group tickets "naturally." In practice, clustering collapsed on short/noisy text. Many runs gave 0 clusters (all noise) or only 3–4 clusters surviving out of 20+ expected topics. This made it unreliable as the main driver.

**What I chose:** Classification-first, because it guarantees coverage. Every ticket gets tags, even if imperfect.

**Implementation:**

Used Gemini 1.5 8b (JSON mode) to generate:
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

This was the biggest innovation. Classifying is easy, checking if it's reliable is the hard part.

**What I tried:**
- **HDBSCAN** on ~30 tickets → collapsed:
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
- With only ~30 tickets, metrics swing a lot. With >2k tickets, coherence should stabilize
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
- **Dataset size:** ~30 tickets was too small for stable clusters. With 2k+, expect more reliable coherence
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