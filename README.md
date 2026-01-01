# Task Cascades

Code for **"Task Cascades for Efficient Unstructured Data Processing"** (SIGMOD 2026).

Shreya Shankar, Sepanta Zeighami, Aditya G. Parameswaran

[[Paper]](https://www.sh-reya.com/task_cascades_preprint.pdf)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone https://github.com/ucbepic/task-cascades.git
cd task-cascades
pip install -r requirements.txt
```

### 2. Download Data (Git LFS)

The datasets are stored with Git LFS. After cloning:

```bash
git lfs install
git lfs pull
```

Verify the data files exist:
```bash
ls expt_data/
# Should show: agnews_test.csv, court_opinions.csv, enron.csv, etc.
```

### 3. Set Up API Keys

Create a `.env` file in the project root:
```bash
OPENAI_API_KEY=your_key_here
```

## Quick Start

```bash
# Run a single experiment
python task_cascades/experiments/run_experiments.py --task game_review

# Run specific methods only
python task_cascades/experiments/run_experiments.py --task legal_doc --methods baseline task_cascades

# Use config file to select methods
python task_cascades/experiments/run_experiments.py --task enron --methods_config task_cascades/config/methods_config.yaml
```

## Reproducing Paper Experiments

### Table 2: Main Results

```bash
bash scripts/run_all_experiments.sh
```

### Figure 5: Varying Target Accuracy

```bash
bash scripts/run_varying_target.sh
```

### Table 3: Repeated Trials

```bash
bash scripts/run_all_repeated_trials.sh
```

## Methods

### Baselines

| Method | Key | Description |
|--------|-----|-------------|
| Oracle Only | `oracle` | All documents sent to GPT-4o |
| 2-Model Cascade | `baseline` | GPT-4o-mini → GPT-4o |
| 2-Model Cascade (+G) | `baseline_guaranteed` | With accuracy guarantees |

### Task Cascades

| Method | Key | Description |
|--------|-----|-------------|
| **Task Cascades** | `task_cascades` | Full pipeline (3 iter × 5 surrogates) |
| **Task Cascades (+G)** | `task_cascades_guaranteed` | With accuracy guarantees |
| **Task Cascades (Lite)** | `task_cascades_lite` | 1 iteration, 8 surrogates |

### Variants

| Method | Key | Description |
|--------|-----|-------------|
| No Surrogates | `no_surrogates` | Learned filtering only |
| Single-Iteration | `single_iteration` | All 15 surrogates in one iteration |
| No Filtering | `no_filtering` | Surrogates on full documents |
| Naive RAG Filter | `naive_rag_filter` | Cosine similarity filtering |
| Selectivity Ordering | `selectivity_ordering` | Selectivity-based cascade ordering |
| Restructure (Top-25%) | `restructure_top25` | Keep top-25% relevant chunks |
| RAG + NoSur | `rag_no_surrogates` | Similarity filtering, no surrogates |

## Tasks

| Dataset | Key | Description |
|---------|-----|-------------|
| AGNEWS | `ag_news` | Classify news article summaries into one of four topics: *World*, *Sports*, *Business*, or *Science/Tech* |
| COURT | `court_opinion` | Determine if a U.S. Supreme Court opinion reverses the lower-court ruling |
| ENRON | `enron` | Identify emails sent by C-suite or VP-level executives in the Enron corpus |
| FEVER | `fever` | Decide whether a natural-language claim is supported by the provided evidence snippets |
| GAMES | `game_review` | Determine whether a review praises a different game more than the one being reviewed |
| LEGAL | `legal_doc` | Detect covenants not to sue or IP no-challenge clauses in license agreements |
| PUBMED | `pubmed` | Classify biomedical articles into one of six study types: *RCT*, *Observational*, *Meta-analysis*, *Bench/Lab*, *Computational*, or *Review* |
| WIKI_TALK | `wiki_talk` | Predict whether a Wikipedia Talk-page discussion culminates in an edit revert |

## Configuration

Edit `task_cascades/config/methods_config.yaml`:

```yaml
methods:
  # Baselines
  oracle: true
  baseline: true
  baseline_guaranteed: true

  # Task Cascades
  task_cascades: true
  task_cascades_guaranteed: true
  task_cascades_lite: true

  # Variants
  no_surrogates: true
  single_iteration: true
  no_filtering: true
  naive_rag_filter: true
  selectivity_ordering: true
  restructure_top25: true
  rag_no_surrogates: true
```

## Project Structure

```
task-cascades/
├── task_cascades/           # Main package
│   ├── config/              # Configuration and method settings
│   ├── data/                # Dataset loading
│   ├── filtering/           # Document filtering
│   ├── cascade/             # Cascade design and surrogate discovery
│   ├── predictors/          # LLM wrappers
│   ├── baselines/           # LOTUS baseline
│   └── experiments/         # Experiment runners
├── scripts/                 # Shell scripts
├── analysis/                # Result analysis
├── expt_data/               # Datasets (Git LFS)
└── results/                 # Output
```

## Cost Warning

Full experiments cost ~$1,000 in OpenAI API calls. Start small:

```bash
python task_cascades/experiments/run_experiments.py --task game_review --sample_size 100
```
