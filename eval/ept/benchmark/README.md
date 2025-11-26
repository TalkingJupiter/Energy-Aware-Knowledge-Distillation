# EPT-Bench: Energy-Per-Token Benchmarking Framework  
**High-Fidelity GPU Energy Measurement for Large Language Model Inference**

---

## 1. Introduction

EPT-Bench is a research-grade framework designed to quantify the **energy consumption** of large language models (LLMs) during inference.  
It provides precise, GPU-level power monitoring, standardized prompt evaluation, and reproducible **Energy-Per-Token (EPT)** metrics to support:

- Energy-aware model comparison  
- Knowledge distillation research  
- Model efficiency analysis  
- Accuracy–energy trade-off studies  
- Sustainable AI system design  

The framework is hardware-agnostic and supports any HuggingFace-compatible causal language model, including large-scale foundation models, PEFT-adapted students, and custom finetuned checkpoints.

---

## 2. Key Capabilities

### 2.1 Accurate GPU Energy Measurement  
EPT-Bench uses NVIDIA NVML to record power consumption at sub-second resolution and numerically integrates sampled values to obtain total energy (in Joules).  
This ensures high-fidelity energy measurements in HPC and multi-GPU production environments.

### 2.2 Standardized Token Accounting  
EPT-Bench computes:
- **Total input tokens**  
- **Total generated output tokens**  
- **Total processed tokens**

This guarantees model-agnostic, parameter-invariant comparisons between teacher and student models.

### 2.3 Energy-Per-Token Metrics  
EPT-Bench outputs three primary efficiency metrics:

- **Input EPT**  
  `EPT_in = E_run / T_in`

- **Output EPT**  
  `EPT_out = E_run / T_out`

- **Total EPT**  
  `EPT_total = E_run / (T_in + T_out)`

These metrics provide a direct, interpretable measurement of the energy cost per token.

### 2.4 Dataset-Integrated Benchmarking  
Built-in support for:
- **Dolly-15k Instruction Dataset** (with automatic caching and reproducible sampling)
- **Custom prompt lists** (one prompt per line in `.txt` formats)

This simplifies controlled prompt selection across multiple models and configurations.

---

## 3. Project Structure

```
eval/
└── ept/
    └── benchmark/
        ├── README.md                 ← This documentation
        ├── ept_monitor.py            ← NVML-based GPU power/energy monitor
        ├── ept_data.py               ← Dataset loader with cache detection
        └── run_ept_benchmark.py      ← Core EPT evaluation pipeline
```

---

## 4. Installation

Ensure your environment includes CUDA-enabled PyTorch and NVIDIA NVML.

```bash
pip install transformers datasets nvidia-ml-py3 torch
```

Verify NVML availability:

```bash
python -c "import pynvml; pynvml.nvmlInit(); print('NVML available')"
```

---

## 5. Usage

### 5.1 Benchmark a Teacher Model (e.g., LLaMA 70B)

```bash
python run_ept_benchmark.py   --model meta-llama/Llama-3.1-70B-Instruct   --use-dolly   --num-prompts 100   --batch-size 2   --max-new-tokens 64   --gpu-indices 0   --out results/ept_teacher.json
```

### 5.2 Benchmark a Student Model (e.g., 8B PEFT)

```bash
python run_ept_benchmark.py   --model serialization_dir/student_8B_KD   --use-dolly   --num-prompts 100   --batch-size 4   --max-new-tokens 64   --gpu-indices 0   --out results/ept_student.json
```

### 5.3 Using Custom Prompt Files

```bash
python run_ept_benchmark.py   --model meta-llama/Llama-3.1-8B-Instruct   --prompts prompts.txt   --batch-size 4   --gpu-indices 0
```

---

## 6. Output Format

Each run produces a structured JSON record:

```json
{
  "E_run_J": 206.531,
  "T_in": 1840,
  "T_out": 6420,
  "EPT_in_J_per_tok": 0.1122,
  "EPT_out_J_per_tok": 0.0321,
  "EPT_total_J_per_tok": 0.0248
}
```

### Interpretation:
- **E_run_J**: Total energy consumed during inference  
- **T_in / T_out**: Token counts  
- **EPT metrics**: Energy efficiency indicators  

These outputs are directly usable for:
- Comparative evaluation  
- Plotting energy–accuracy curves  
- KD energy savings analysis  
- Model selection based on efficiency  

---

## 7. Methodology

### 7.1 Power Sampling
Power readings (Watts) are collected using NVML:

`E = Σ (P_i × Δt_i)`

This corresponds to a high-resolution numerical integration of instantaneous GPU power samples.

### 7.2 Token Accounting
Token counts are computed from the HF tokenizer using attention masks and generated output sequence lengths.  
EPT-Bench automatically normalizes HuggingFace `generate()` outputs (`GenerateOutput`, beam search outputs, or raw tensors) to ensure consistent parsing.

---

## 8. Recommended Experimental Settings

- **Batch sizes:**  
  - Teachers (≥30B): 1–2  
  - Students (7B–13B): 2–8  

- **Decoding settings:**  
  `do_sample = False`  
  `max_new_tokens = 64` (baseline recommended)

- **Prompt count:**  
  - 50–100 for quick evaluation  
  - 200–500 for publication-level stability  

---

## 9. Applications

EPT-Bench is appropriate for:

- Evaluating KD energy savings  
- Measuring inference energy of large-scale LLMs  
- Producing energy-aware design insights  
- HPC reports and research publications  
- Efficiency-oriented LLM selection and compression  

---

## 10. Citation

If you use EPT-Bench in research or industry:

```
Sencer, B. (2025). EPT-Bench: Energy-Per-Token Benchmarking Framework for Large Language Models.
https://github.com/TalkingJupiter/Energy-Aware-Knowledge-Distillation.git
```

---

## 11. Support

For integration help, dataset customization, LM-Eval-Harness extensions, plotting utilities, or automated reporting, please open an issue or contact [Batuhan S.](batuhan.sencer@ttu.edu)

---
