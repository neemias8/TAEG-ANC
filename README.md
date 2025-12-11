# TAEG-ANC: Temporal Anchoring Event Graph for Abstractive Narrative Consolidation

## Overview

**TAEG-ANC** is a Multi-Document Summarization (MDS) framework designed to consolidate multiple gospel narratives (Matthew, Mark, Luke, John) into a single, cohesive chronological summary. 

Unlike traditional methods that simply concatenate texts or use extractive ranking, TAEG-ANC employs a **Temporal Anchoring Event Graph (TAEG)** to align events across different documents chronologically. It then applies both Extractive and Abstractive summarization strategies to these aligned events to produce a final narrative.

This repository includes implementations for:
- **Graph Construction**: Intelligent parsing and linking of gospel verses.
- **Extractive Summarization**: LexRank.
- **Abstractive Summarization**: BART, PEGASUS, PRIMERA and GEMMA.

## The Challenge: Narrative Consolidation vs. Summarization

A key distinction of this project is its focus on **Narrative Consolidation** rather than traditional Multi-Document Summarization (MDS).

Standard MDS algorithms are typically designed for **brevity** and **conciseness** (e.g., generating a short news headline from multiple articles). They explicitly try to remove redundancy and shorten the text.

**Narrative Consolidation**, however, opposes this goal. In domains such as **Biblical Harmonization** or **Criminal Witness Testimonies**, the objectives are:
1.  **Completeness**: Preserving unique details from *every* source. If Witness A saw a red car and Witness B saw the license plate, the consolidation must include *both*.
2.  **Detail**: Retaining the richness of the original text rather than compressing it.
3.  **Temporal Fluidity**: Ensuring the combined story flows logically and chronologically, rather than being a disjointed collection of facts.

TAEG-ANC addresses this by using the Temporal Graph to "anchor" events, allowing the system to "zip" multiple perspectives into a single, high-resolution narrative stream, prioritizing **recall** and **coherence** over compression.

## Methodology

The core innovation is the **Temporal Graph**, which breaks down the four gospels into discrete "events" anchored in a timeline.

1.  **Graph Construction**: 
    - Vertices represent specific events in specific gospels (e.g., "The Baptism of Jesus" in Matthew).
    - Edges represent temporal relationships (`BEFORE`, `SAME_EVENT`).
    - The graph is built using heuristics and specific verse references.

2.  **Consolidation**:
    - **TAEG-Guided**: The system traverses the chronological list of events. For each event, it gathers all relevant texts (e.g., Matthew 3 + Mark 1 + Luke 3) and generates a local summary using the chosen strategy. These local summaries are then concatenated to form the global narrative.
    - **Pure Global (Baseline)**: All texts are concatenated into one massive document and fed to the summarizer (truncated to model limits).

## Strategies & Models

The system implements the **Strategy Pattern** for easy switching between summarization methods:

### Extractive
- **LexRank-TA**: A graph-based ranking algorithm that selects the most salient sentences from the inputs.
  > **Note**: LexRank runs via the legacy `run_taeg.py` script to preserve the original extractive algorithm logic.

### Abstractive
- **BART** (`facebook/bart-large-cnn`): Excellent for coherent rewriting of short-to-medium length contexts.
- **PEGASUS** (`google/pegasus-cnn_dailymail`): While `multi_news` is fine-tuned for MDS, we observed excessive hallucination of external news entities (e.g. "The New York Times") when applied to Biblical text. We switched to the `cnn_dailymail` checkpoint, which demonstrated significantly better robustness and fidelity to the source material for this domain.
- **PRIMERA** (`allenai/PRIMERA`): A model designed for efficient long-document MDS usage (pyramid-based pre-training).
- **GEMMA 3** (`google/gemma3:4b` via Ollama): A modern, instruction-tuned Large Language Model (LLM). We use it to demonstrate how a general-purpose LLM performs when guided by the temporal graph versus "raw" processing.

> **Note**: All models are configured to run on **CPU**. Gemma runs via the **Ollama** local inference server.

## Installation

1.  **Clone the repository**.
2.  **Create a virtual environment**:
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```
3.  **Install dependencies**:
    ```powershell
    pip install -r requirements.txt
    pip install -r requirements_abstractive.txt
    ```
4.  **Install Ollama (for Gemma 3)**:
    - Download and install [Ollama](https://ollama.com).
    - Pull the required model:
      ```powershell
      ollama pull gemma3:4b
      ```
    - Verify setup:
      ```powershell
      python check_ollama.py
      ```

## Usage

### 1. Run Complete Experiment (Recommended)
To run all methods (LexRank, BART, PEGASUS, PRIMERA, GEMMA) in both TAEG-Guided and Pure modes, and generate a comparison table:

```powershell
python run_complete_experiment.py
```

This script handles:
- Graph construction (and caching).
- Automated execution of all summarizers.
- Evaluation using ROUGE (Token overlap), METEOR (Semantic match), BERTScore (Neural similarity), and Kendall's Tau (Temporal ordering).
- Comparison results saved to `outputs/comparison_results.csv`.

### 2. Run Individual Components

**TAEG-Guided Summarization**:
```powershell
python run_taeg_abstractive.py --method gemma
# Options: bart, pegasus, primera, gemma
```

**Pure (Global) Summarization**:
```powershell
python run_pure_abstractive.py --method gemma
```

**Evaluation**:
```powershell
python compare_methods.py
```

## Project Structure

```
TAEG-ANC/
├── src/
│   ├── consolidators.py       # Strategy pattern implementations (Models)
│   ├── main.py                # Legacy pipeline entry point
│   ├── summarizer.py          # Legacy LexRank logic
│   ├── evaluator.py           # Metrics (ROUGE, METEOR, etc.)
│   └── data_loader.py         # Gospel text loading logic
├── data/                      # XML chronology and gospel texts
├── outputs/                   # Generated summaries and reports
├── improved_graph_builder.py  # Core graph construction logic
├── run_complete_experiment.py # Master orchestration script
├── run_taeg_abstractive.py    # TAEG-based runner
├── run_pure_abstractive.py    # Baseline runner
├── compare_methods.py         # Evaluation script
└── requirements_abstractive.txt
```

## Authors
[Your Name/Team] - [University/Institution]

## Results

The following table summarizes the performance of different methods evaluated on the Four Gospels dataset. The **TAEG-Guided** methods (using the proposed Temporal Graph) significantly outperform the **Pure** (Baseline) abstractive models in this domain, particularly in maintaining temporal coherence (Kendall's Tau).

| Method | Length (chars) | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | METEOR | Kendall Tau |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **TAEG-LexRank** | 59408 | 0.8037 | 0.6551 | 0.2065 | 0.8351 | 0.3613 | 0.3048 |
| **TAEG-LexRank-TA** (Oracle) | 87528 | 0.9805 | 0.9548 | 0.9655 | 0.9863 | 0.7345 | 1.0000 |
| **TAEG-BART** | 42220 | 0.6271 | 0.5163 | 0.4844 | 0.8874 | 0.2633 | 1.0000 |
| **TAEG-PEGASUS** | 33740 | 0.5337 | 0.4286 | 0.4074 | 0.8640 | 0.2161 | 1.0000 |
| **TAEG-PRIMERA** | 83811 | 0.8846 | 0.7085 | 0.6200 | 0.9094 | 0.4801 | 1.0000 |
| **TAEG-GEMMA** | 101395 | 0.8883 | 0.7168 | 0.7036 | 0.8931 | 0.4926 | 1.0000 |
| Pure-BART | 525 | 0.0113 | 0.0092 | 0.0098 | 0.8240 | 0.0033 | 0.5842 |
| Pure-PEGASUS | 462 | 0.0110 | 0.0093 | 0.0099 | 0.8086 | 0.0032 | -0.0454 |
| Pure-PRIMERA | 4352 | 0.0959 | 0.0783 | 0.0678 | 0.8434 | 0.0273 | 0.4471 |
| Pure-GEMMA | 3300 | 0.0451 | 0.0167 | 0.0289 | 0.8094 | 0.0141 | 0.3919 |

> **Key Takeaway**: TAEG-ANC succeeds in "zipping" the narratives together. **TAEG-GEMMA** achieved the best overall performance, demonstrating that combining the structural guidance of the Temporal Graph with the linguistic capability of modern LLMs yields the highest quality consolidation. Conversely, even powerful LLMs (Pure-GEMMA) fail to capture the full scope of the narrative when attempting to summarize the entire corpus globally without structural guidance.