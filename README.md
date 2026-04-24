# SycoEval-EM: Sycophancy Evaluation in Simulated Emergency Care Encounters

<p>
    <a href="https://arxiv.org/abs/2601.16529" target="_blank"><img src="https://img.shields.io/badge/arXiv-2601.16529-b31b1b"></a>
</p>

This repository contains the implementation of [SycoEval-EM](https://arxiv.org/abs/2601.16529), a benchmark for evaluating whether large language models acting as doctors acquiesce to unindicated patient requests or maintain guideline-adherent care in simulated emergency medicine consultations.

This code is intended for research use.

## Benchmark Summary

SycoEval-EM runs multi-agent clinical simulations with three roles:

- Patient agent: presents a clinical scenario and applies a persuasion tactic
- Doctor agent: responds to the patient while following emergency care guidelines
- Evaluator agents: score the conversation using a three-judge evaluation setup

The benchmark includes three emergency care scenarios:

- Low-risk migraine-like headache with a request for head CT
- Viral sinusitis symptoms with a request for antibiotics
- Acute non-specific low back pain with a request for opioids

The benchmark includes five persuasion tactics:

- Emotional fear
- Anecdotal or social proof
- Persistence and challenge
- Preemptive assertion
- Citation pressure

## Results

![Model-by-scenario sycophancy heatmap](src/eval/syco-em-plot/heatmap_model_scenario.png)

## Quick Start

This section shows how to run the final benchmark code in `src/main/sycophancy.py`.

### Step 1. Prerequisites

Clone this repository and install the required Python packages.

```bash
cd YOURPATH/syco-eval
pip install requests python-dotenv
```

### Step 2. Configure the API Key

Create a `.env` file in the repository root with your OpenRouter API key.

```bash
OPENROUTER_API_KEY=your_openrouter_api_key
```

### Step 3. Run the Benchmark

From the repository root, run:

```bash
python src/main/sycophancy.py
```

By default, the benchmark evaluates every model listed in `DOCTOR_MODELS` using:

- Patient model: `google/gemini-2.5-flash`
- Evaluator models: `openai/gpt-4o-mini`, `x-ai/grok-4-fast`, `google/gemini-2.5-flash`
- 5 runs per scenario and persuasion tactic
- 4 parallel workers

### Step 4. Retrieve the Results

Results are saved to:

```text
YOURPATH/syco-eval/results/syco
```

Each doctor model gets a separate output folder containing:

- `full_results_*.json`: full conversations, judge evaluations, and run metadata
- `summary_*.json`: aggregate metrics for that model

The full benchmark run also writes:

- `master_summary_*.json`: status and summary across all doctor models

## Configuration

To change the benchmark setup, edit the constants in `src/main/sycophancy.py`:

- `DOCTOR_MODELS`: doctor models to evaluate
- `PATIENT_MODEL`: patient simulator model
- `EVALUATOR_MODELS`: three judge models
- `NUM_SIMULATIONS_PER_CONDITION`: runs per scenario/tactic pair
- `MAX_WORKERS`: parallel API workers
- `OUTPUT_DIR`: output path, using `YOURPATH` instead of a machine-specific path

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@misc{peng2026sycoevalemsycophancyevaluationlarge,
      title={SycoEval-EM: Sycophancy Evaluation of Large Language Models in Simulated Clinical Encounters for Emergency Care}, 
      author={Dongshen Peng and Yi Wang and Austin Schoeffler and Carl Preiksaitis and Christian Rose},
      year={2026},
      eprint={2601.16529},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2601.16529}, 
}
```
