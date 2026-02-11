<h1 align="left">LLM-in-Sandbox</h1>

<p align="left">
  <a href="https://llm-in-sandbox.github.io">🌐 Project Page</a> •
  <a href="https://arxiv.org/abs/2601.16206">📄 Paper</a> •
  <a href="https://huggingface.co/papers/2601.16206">🤗 Huggingface</a>
</p>

Enabling LLMs to explore within a code sandbox (i.e., a virtual computer) to elicit general agentic intelligence.

<p align="left">
  <img src="https://llm-in-sandbox.github.io/assets/intro.png" alt="Experiment Results" width="600">
</p>

<p align="left">
  <a href="https://www.youtube.com/watch?v=Ols-XrOwIHo&t=1s">
    <img src="https://img.youtube.com/vi/Ols-XrOwIHo/maxresdefault.jpg" alt="Demo Video" width="600">
  </a>
  <br>
  <em>▶️ Click to watch the demo video</em>
</p>

**Features:**
- 🌍 General-purpose: works beyond coding—scientific reasoning, long-context understanding, video production, travel planning, and more
- 🐳 Isolated execution environment via Docker containers
- 🔌 Compatible with OpenAI, Anthropic, and self-hosted servers (vLLM, SGLang, etc.)
- 📁 Flexible I/O: mount any input files, export any output files

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [More Examples](#more-examples)
- [Benchmark and Reproduction](#benchmark-and-reproduction)
- [Citation](#citation)

## Installation

**Requirements:** Python 3.10+, [Docker](https://docs.docker.com/engine/install/)

```bash
pip install llm-in-sandbox
```

Or install from source:

```bash
git clone https://github.com/llm-in-sandbox/llm-in-sandbox.git
cd llm-in-sandbox
pip install -e .
```

**Docker Image**

The default Docker image (`cdx123/llm-in-sandbox:v0.1`) will be automatically pulled when you first run the agent. The first run may take a minute to download the image (~400MB), but subsequent runs will start instantly. 

<details>
<summary>Advanced: Build your own image</summary>

Modify [Dockerfile](./docker/Dockerfile) and build your own image:

```bash
llm-in-sandbox build
```

</details>

## Quick Start

LLM-in-Sandbox works with various LLM providers including OpenAI, Anthropic, and self-hosted servers (vLLM, SGLang, etc.).

### Option 1: Cloud / API Services

```bash
llm-in-sandbox run \
    --query "write a hello world in python" \
    --llm_name "openai/gpt-5" \
    --llm_base_url "http://your-api-server/v1" \
    --api_key "your-api-key"
```

### Option 2: Self-Hosted Models

<details>
<summary>Using local vLLM server for Qwen3-Coder-30B-A3B-Instruct</summary>

**1. Start vLLM server:**
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --served-model-name qwen3_coder \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --tensor-parallel-size 8  \
    --enable-prefix-caching
```

**2. Run agent (in a new terminal once server is ready):**
```bash
llm-in-sandbox run \
    --query "write a hello world in python" \
    --llm_name qwen3_coder \
    --llm_base_url "http://localhost:8000/v1"  \
    --temperature 0.7
```

</details>

<details>
<summary>Using local SGLang server for DeepSeek-V3.2-Thinking</summary>

**1. Start sgLang server:**
```bash
python3 -m sglang.launch_server \
    --model-path "deepseek-ai/DeepSeek-V3.2" \
    --served-model-name "DeepSeek-V3.2" \
    --trust-remote-code \
    --tp-size 8 \
    --tool-call-parser deepseekv32 \
    --reasoning-parser deepseek-v3 \
    --host 0.0.0.0 \
    --port 5678
```

**2. Run agent (in a new terminal once server is ready):**
```bash
llm-in-sandbox run \
    --query "write a hello world in python" \
    --llm_name DeepSeek-V3.2 \
    --llm_base_url "http://0.0.0.0:5678/v1" \
    --extra_body '{"chat_template_kwargs": {"thinking": True}}'
```

</details>

### Parameters (Common)

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--query` | Task for the agent | *required* |
| `--llm_name` | Model name | *required* |
| `--llm_base_url` | API endpoint URL | *from LLM_BASE_URL env var* |
| `--api_key` | API key (not needed for local server) | *from OPENAI_API_KEY env var* |
| `--input_dir` | Input files folder to mount (Optional) | *None* |
| `--output_dir` | Output folder for results | `./output` |
| `--docker_image` | Docker image to use | `cdx123/llm-in-sandbox:v0.1` |
| `--prompt_config` | Path to prompt template | `./config/general.yaml` |
| `--temperature` | Sampling temperature | `1.0` |
| `--max_steps` | Max conversation turns | `100` |
| `--extra_body` | Extra JSON body for LLM API calls | *None* |

Run `llm-in-sandbox run --help` for all available parameters.

### Output

Each run creates a timestamped folder:

```
output/2026-01-16_14-30-00/
├── files/
│   ├── answer.txt      # Final answer
│   └── hello_world.py  # Output file
└── trajectory.json     # Execution history
```

## More Examples

We provide examples across diverse non-coding domains: scientific reasoning, long-context understanding, instruction following, travel planning, video production, music composition, poster design, and more.

👉 See [examples/README.md](./examples/README.md) for the full list.

## Benchmark and Reproduction

Reproduce our paper results, evaluate any LLM in the sandbox, or add your own tasks.

👉 See [llm_in_sandbox/benchmark/README.md](./llm_in_sandbox/benchmark/README.md)


## Contact Us

Daixuan Cheng: daixuancheng6@gmail.com  
Shaohan Huang: shaohanh@microsoft.com  

## Acknowledgment

We learned the design and reused code from [R2E-Gym](https://github.com/R2E-Gym/R2E-Gym). Thanks for the great work!

## Citation
If you find our work helpful, please cite us:
```bibtex
@article{cheng2026llm,
  title={LLM-in-Sandbox Elicits General Agentic Intelligence},
  author={Cheng, Daixuan and Huang, Shaohan and Gu, Yuxian and Song, Huatong and Chen, Guoxin and Dong, Li and Zhao, Wayne Xin and Wen, Ji-Rong and Wei, Furu},
  journal={arXiv preprint arXiv:2601.16206},
  year={2026}
}
```
