# SatQuery AI

### An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis

**Smart India Hackathon 2026 -- SIH26167**  
**Team:** PINNACLE  
**Theme:** Space Technology

SatQuery AI is an agentic, multimodal remote-sensing analysis system
that lets users ask natural-language questions about satellite imagery.
Instead of forcing users to select a separate model for every task, the
system interprets the query, validates the available imagery, routes the
request to the appropriate specialist model, and returns an
evidence-grounded result.

The system is designed for single-image analysis, bi-temporal change
analysis, optical–SAR reasoning, and spatial grounding.

------------------------------------------------------------------------

## Why SatQuery AI?

Remote-sensing analysis is often fragmented across task-specific tools.
A user may need different models and workflows for VQA, grounding, SAR
analysis, and change detection.

SatQuery AI addresses this workflow problem with an agentic controller:

``` text
Natural-language query
        |
        v
Query understanding + input validation
        |
        v
Agentic routing
        |
        +------------------+------------------+------------------+
        |                  |                  |                  |
        v                  v                  v                  v
  RS-adapted VLM      Grounding Agent     Change Agent       SAR / Fusion
        |                  |                  |                  |
        +------------------+------------------+------------------+
                                   |
                                   v
                         Result fusion + evidence
                                   |
                                   v
                      Answer + visual evidence
                      + confidence + execution trace
```

------------------------------------------------------------------------

## Key Capabilities

| Capability              | Purpose                                               |
|-------------------------|-------------------------------------------------------|
| Remote-sensing VLM      | Single-image VQA, scene understanding and description |
| Spatial Grounding       | Locate queried objects or regions                     |
| Bi-temporal Change      | Analyze and describe changes between two dates        |
| SAR Analysis            | Extract information from SAR imagery                  |
| Optical + SAR reasoning | Combine complementary sensor information              |
| Agentic routing         | Select the appropriate specialist workflow            |
| Evidence output         | Return visual/spatial evidence with the answer        |
| Execution trace         | Show selected task, model and important parameters    |

### Supported input concepts

- Single optical/multispectral image
- Single SAR image
- Co-registered optical + SAR pair
- Bi-temporal image pair
- GeoTIFF/TIFF for geospatial imagery
- Natural-language analysis queries

------------------------------------------------------------------------

# System Architecture

``` mermaid
flowchart TB
    U["User<br/>Natural Language Query + Image(s)"]
    API["FastAPI Backend<br/>POST /analyze"]
    V["Input Validation<br/>Format • Count • Modality • Metadata"]
    G["LangGraph Agentic Controller<br/>Query Understanding + Routing"]

    VLM["RS-Adapted Multispectral VLM<br/>Qwen3.5-2B + LoRA + Spectral Adapter"]
    GR["Spatial Grounding Agent"]
    CH["Bi-temporal Change Agent"]
    SAR["SAR / Optical-SAR Agent"]

    F["Result Fusion"]
    E["Evidence + Confidence"]
    T["Auditable Execution Summary"]
    OUT["Final Response<br/>Text + Spatial/Visual Evidence"]

    U --> API --> V --> G
    G --> VLM
    G --> GR
    G --> CH
    G --> SAR
    VLM --> F
    GR --> F
    CH --> F
    SAR --> F
    F --> E
    F --> T
    E --> OUT
    T --> OUT
```

------------------------------------------------------------------------

# Agentic Workflow

``` mermaid
flowchart LR
    A["User Query"] --> B["Understand Intent"]
    B --> C["Validate Inputs"]
    C --> D{"Select Workflow"}

    D -->|Single Image VQA / Caption| E["Multispectral VLM"]
    D -->|Object / Region Location| F["Grounding Agent"]
    D -->|Two Dates| G["Change Agent"]
    D -->|SAR / Optical + SAR| H["SAR / Fusion Agent"]

    E --> I["Combine Results"]
    F --> I
    G --> I
    H --> I

    I --> J["Evidence + Confidence"]
    J --> K["Answer + Execution Summary"]
```

The controller follows:

``` text
Query
  ↓
Understand
  ↓
Validate
  ↓
Select specialist
  ↓
Execute
  ↓
Fuse
  ↓
Explain
```

------------------------------------------------------------------------

# Multispectral VLM

The remote-sensing VLM is adapted from Qwen3.5-2B using the project’s
multispectral training pipeline.

``` text
Sentinel-2 / Multispectral Input
            |
            v
      12-band patch
            |
            v
   Spectral Adapter
       12 -> 3
            |
            v
     Qwen Vision Encoder
            |
            v
       Qwen Language Model
            |
            v
      Natural-language answer
```

The trained component consists of:

``` text
Qwen3.5-2B
    +
LoRA adapter
    +
12 -> 3 Spectral Adapter
```

The spectral adapter is required during inference because the trained
model accepts the project’s 12-band multispectral representation rather
than ordinary RGB input.

------------------------------------------------------------------------

# Repository Structure

``` text
satquery-ai/
│
├── agent/
│   ├── graph.py
│   ├── router_node.py
│   └── state.py
│
├── specialist_models/
│   ├── change_agent.py
│   ├── grounding_agent.py
│   ├── multispec_vlm_agent.py
│   ├── sar_agent.py
│   └── sar_class.py
│
├── backend/
├── frontend/
├── models/
│   └── qwen35_multispectral_6000/
├── docs/
│   └── images/
├── scripts/
└── README.md
```

Keep large model weights outside Git when possible. Use Git LFS or a
model registry for large checkpoints.

------------------------------------------------------------------------

# Running Locally

## 1. Clone

``` bash
git clone <YOUR_REPOSITORY_URL>
cd satquery-ai
```

## 2. Create an environment

``` bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Install the versions compatible with your GPU, Qwen/Unsloth setup and
other specialist models.

## 4. Configure Hugging Face

Create `.env`:

``` env
HF_TOKEN=your_huggingface_token
```

Do not commit `.env`.

## 5. Place the trained VLM

``` text
models/
└── qwen35_multispectral_6000/
    ├── adapter_config.json
    ├── adapter_model.safetensors
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── spectral_adapter.pt
```

The exact exported files can vary slightly by checkpoint.

## 6. Test the VLM

``` bash
python multispec_test.py
```

Example structured output:

``` text
{
    "answer": "grassland",
    "agent": "multispectral_vlm",
    "model": "Qwen3.5-2B-SatQuery-6K",
    "task": "remote_sensing_vqa",
    "confidence": null
}
```

------------------------------------------------------------------------

# Using Hugging Face

For models available through the Hugging Face Hub or an inference
endpoint:

``` bash
pip install huggingface_hub
export HF_TOKEN="your_huggingface_token"
```

Example:

``` python
import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    token=os.environ["HF_TOKEN"]
)

response = client.chat.completions.create(
    model="YOUR_HF_MODEL_ID",
    messages=[
        {
            "role": "user",
            "content": "Describe the remote-sensing scene."
        }
    ],
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### Important: custom multispectral VLM

The SatQuery multispectral VLM is not a standard RGB VLM. Its complete
inference path is:

``` text
12-band preprocessing
       +
trained spectral adapter
       +
Qwen3.5-2B
       +
LoRA adapter
```

Therefore, a generic Hugging Face chat call does not automatically
reproduce the complete SatQuery multispectral pipeline.

For this custom component:

1.  Run the complete model locally with `multispec_vlm_agent.py`, or
2.  Deploy the complete inference pipeline behind a custom Hugging Face
    Inference Endpoint, or
3.  Deploy the complete pipeline as a Hugging Face Space/API service.

The client should call that service rather than duplicating
model-loading logic.

------------------------------------------------------------------------

# Backend API

``` text
Frontend
   |
   | query + image(s)
   v
FastAPI
   |
   v
Input Validation
   |
   v
LangGraph
   |
   v
Specialist Agents
   |
   v
Result Fusion
   |
   v
JSON Response
```

Representative endpoint:

``` http
POST /analyze
Content-Type: multipart/form-data
```

Conceptual request:

``` text
query = "What land cover is present in this image?"
image = <GeoTIFF>
```

Representative response:

``` json
{
  "answer": "The image contains predominantly grassland.",
  "task": "remote_sensing_vqa",
  "models": ["Qwen3.5-2B-SatQuery-6K"],
  "evidence": {},
  "confidence": null
}
```

The exact API schema follows the current backend implementation.

------------------------------------------------------------------------

# Visual Evidence

SatQuery AI is designed to return more than plain text:

``` text
                    Query
                      |
                      v
               Specialist Agent
                      |
          +-----------+-----------+
          |                       |
          v                       v
     Text Answer             Spatial Evidence
          |                       |
          +-----------+-----------+
                      |
                      v
              Confidence / Trace
```

Possible evidence includes:

- highlighted regions
- bounding boxes
- change maps
- optical/SAR comparison views
- selected model/task information

------------------------------------------------------------------------

# Problem → Solution

| Remote-sensing problem                                | SatQuery AI approach                            |
|-------------------------------------------------------|-------------------------------------------------|
| Task-specific AI tools                                | One agentic interface over multiple specialists |
| Difficult model selection                             | Automatic query-driven routing                  |
| Multisensor imagery is difficult to interpret jointly | Optical/SAR specialist workflow                 |
| Changes require multiple observations                 | Bi-temporal change agent                        |
| Users need spatial answers, not only text             | Grounding and visual evidence                   |
| Generic VLMs lack remote-sensing adaptation           | Remote-sensing adapted VLM                      |
| Analysis can be difficult to audit                    | Execution summary and evidence                  |
| Complex workflows for non-experts                     | Natural-language interaction                    |

------------------------------------------------------------------------

# Example Queries

``` text
"Describe the land cover and major objects visible in this image."

"What type of vegetation is present?"

"Highlight the water body referred to in the query."

"What changed between these two dates?"

"Where did the change occur?"

"Use the optical and SAR images together to identify built-up regions."

"Has the built-up area increased or decreased?"
```

------------------------------------------------------------------------

# Design Principles

### Specialist models

Different remote-sensing tasks use different specialist capabilities
rather than forcing one VLM to perform every task.

### Query-driven orchestration

The user describes the required analysis in natural language and the
controller determines the relevant workflow.

### Multimodal reasoning

Optical/multispectral, SAR and temporal information can be used
according to the input configuration and query.

### Evidence-grounded output

The system is designed to expose spatial/visual evidence alongside the
textual response.

### Auditable execution

The response can expose the selected task, model/tool and important
execution parameters without exposing internal reasoning.

------------------------------------------------------------------------

# Development Roadmap

``` text
[Current]
Multispectral VLM
      |
      v
LangGraph Integration
      |
      v
Unified Agent Router
      |
      +---- Grounding
      +---- Change
      +---- SAR / Fusion
      |
      v
Result Fusion
      |
      v
Evidence + Confidence
      |
      v
FastAPI
      |
      v
Web Interface
```

------------------------------------------------------------------------

# Developers

**Team PINNACLE -- Smart India Hackathon 2026**

**SIH26167:** SatQuery AI — An Interactive Vision-Language Assistant for
Multimodal Remote Sensing Image Analysis through Text Queries
