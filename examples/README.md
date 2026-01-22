# LLM-in-Sandbox Examples

This directory contains examples demonstrating the general-purpose capabilities of the Agent.

## Example List

| Example | Description | Input Files | Special Config | Expected Output |
|---------|-------------|-------------|-----------------|-----------------|
| [01_travel_planning](01_travel_planning/) | Travel itinerary planning | None | No | Detailed itinerary + budget |
| [02_poster_design](02_poster_design/) | Poster design | Event info | No | Poster image |
| [03_video_creation](03_video_creation/) | Video production | Theme info | No | MP4 video |
| [04_music_composition](04_music_composition/) | Music composition | None | No | MIDI/audio |
| [05_chemical_analysis](05_chemical_analysis/) | Chemical Analysis | None | Yes | Answer: "A" |
| [06_biomed_qa](06_biomed_qa/) | Biomed QA | None | Yes | Answer: "A" |
| [07_physics_reasoning](07_physics_reasoning/) | Physics Reasoning | None | Yes | Answer: "\boxed{\frac{1}{k} m g \sin \alpha + d}" |
| [08_math_problem](08_math_problem/) | Math Problem | None | Yes | Answer: "\boxed{588}" |
| [09_long_context](09_long_context/) | Long-Context Reasoning | Academic Documents | Yes | Answer: "1. Airline Industry (12)\\n2. Accommodation Industry (4)" |
| [10_instruct_follow](10_instruct_follow/) | Instruction Following | None | Yes | Open-ended Answer |

## Quick Start

```bash
# Navigate to an example directory
cd examples/01_travel_planning

# Run the example (requires LLM service to be configured)
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --llm_name "your-model" \
    --llm_base_url "http://your-api-server/v1" \
    --api_key "your-api-key" \
    --extra_body `[OPTIONAL] Extra JSON body for LLM API calls`

# If the example requires input files
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --input_dir ./input \
    --llm_name "your-model" \
    --llm_base_url "http://your-api-server/v1" \
    --api_key "your-api-key" \
    --extra_body `[OPTIONAL] Extra JSON body for LLM API calls`

# If the example requires special prompt config (e.g., 05-10)
cd examples/05_chemical_analysis
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --prompt_config ./prompt_config.yaml \
    --llm_name "your-model" \
    --llm_base_url "http://your-api-server/v1" \
    --api_key "your-api-key" \
    --extra_body `[OPTIONAL] Extra JSON body for LLM API calls`
```

<details>
<summary>Example Usage with Deepseek-V3.2-Thinking</summary>

**1. Start server:**
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
# Navigate to an example directory
cd examples/04_music_composition

# Run the example
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --llm_name DeepSeek-V3.2 \
    --llm_base_url "http://0.0.0.0:5678/v1" \
    --extra_body '{"chat_template_kwargs": {"thinking": True}}'

# If the example requires input files
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --input_dir ./input \
    --llm_name DeepSeek-V3.2 \
    --llm_base_url "http://0.0.0.0:5678/v1" \
    --extra_body '{"chat_template_kwargs": {"thinking": True}}'

# If the example requires special prompt config (e.g., 05-10)
cd examples/05_chemical_analysis
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --prompt_config ./prompt_config.yaml \
    --llm_name DeepSeek-V3.2 \
    --llm_base_url "http://0.0.0.0:5678/v1" \
    --extra_body '{"chat_template_kwargs": {"thinking": True}}'

cd examples/09_long_context
llm-in-sandbox run \
    --query "$(cat prompt.txt)" \
    --input_dir ./input \
    --prompt_config ./prompt_config.yaml \
    --llm_name DeepSeek-V3.2 \
    --llm_base_url "http://0.0.0.0:5678/v1" \
    --extra_body '{"chat_template_kwargs": {"thinking": True}}'
```
</details>
