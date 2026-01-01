# Task Cascades

Code for **"Task Cascades for Efficient Unstructured Data Processing"** (SIGMOD 2026).

Shreya Shankar, Sepanta Zeighami, Aditya G. Parameswaran

[[Paper]](https://www.sh-reya.com/task_cascades_preprint.pdf)

## Setup

### 1. Clone and Install Dependencies

```bash
git clone <repo-url>
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
python task_cascades/experiments/example_run.py --task=game_review
```

## Reproducing Paper Experiments

### Table 2: Main Results

```bash
bash scripts/run_all_full_experiments.sh
```

### Figure 5: Varying Target Accuracy

```bash
bash scripts/run_varying_target.sh
```

### Table 3: Repeated Trials

```bash
bash scripts/run_all_repeated_trials.sh
```

## Project Structure

```
task-cascades/
├── task_cascades/           # Main package
│   ├── config/              # Configuration
│   ├── data/                # Dataset loading
│   ├── filtering/           # Document filtering
│   ├── cascade/             # Cascade design
│   ├── predictors/          # LLM wrappers
│   ├── baselines/           # LOTUS baseline
│   └── experiments/         # Experiment runners
├── scripts/                 # Shell scripts
├── analysis/                # Result analysis
├── expt_data/               # Datasets (Git LFS)
└── results/                 # Output
```

## Methods

| Method | Description |
|--------|-------------|
| **Task Cascades** | Full pipeline: surrogates + learned filtering |
| **Task Cascades (Guaranteed)** | With statistical accuracy guarantees |
| **Task Cascades Lite** | Lightweight: 1 iteration, 8 surrogates |
| No Filtering | Variant: surrogates only |
| No Surrogates | Variant: filtering only |
| 2-Model Baseline | GPT-4o-mini → GPT-4o cascade |
| Oracle | All documents to GPT-4o |

## Tasks

| Task | Type | Description |
|------|------|-------------|
| `game_review` | Binary | Review mentions other games positively |
| `legal_doc` | Binary | Covenant not to sue detection |
| `enron` | Binary | Senior executive email |
| `wiki_talk` | Binary | Discussion resulted in reversion |
| `court_opinion` | Binary | Court reverses ruling |
| `fever` | Binary | Claim supported by evidence |
| `ag_news` | 4-class | News classification |
| `pubmed` | 6-class | Study type classification |

## Configuration

Edit `task_cascades/config/methods_config.yaml` to enable/disable methods:

```yaml
methods:
  task_cascades: true
  task_cascades_guaranteed: true
  no_filtering: true
  no_surrogates: true
  baseline: true
  oracle: true
```

## Cost Warning

Full experiments cost ~$1,000 in OpenAI API calls. Start with a smaller sample:

```bash
python task_cascades/experiments/example_run.py --task=game_review --sample_size=100
```
