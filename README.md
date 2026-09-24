## SatQuery-AI SIH 2026 

### Team Pinnacle

## *__SatQueryAI --- WORKFLOW__*

                         ┌─────────────────┐
                         │    Frontend     │
                         │ Upload + Query  │
                         └────────┬────────┘
                                  │
                                  ▼
                         ┌─────────────────┐
                         │     FastAPI     │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │       LangGraph          │
                    │    Agentic Controller    │
                    └────────────┬─────────────┘
                                 │
             ┌───────────────────┼───────────────────┐
             │                   │                   │
             ▼                   ▼                   ▼
       Image Validation      Task Router       Metadata Check
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                  ┌──────────────┼───────────────┐
                  │              │               │
                  ▼              ▼               ▼
              RGB VLM      Multispectral     Change Model
              + LoRA         VLM + Adapter       
                  │              │               │
                  └──────────────┼───────────────┘
                                 ▼
                         Evidence Aggregation
                                 │
                                 ▼
                      Confidence + Explanation
                                 │
                                 ▼
                        Execution Summary
                                 │
                                 ▼
                                User


                         SATQUERY AI
                              │
                              ▼
                       ┌──────────────┐
                       │  LangGraph   │
                       │    Agent     │
                       └──────┬───────┘
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
          ▼                   ▼                    ▼
   MULTISPECTRAL             SAR             BI-TEMPORAL
      VLM                SPECIALIST          SPECIALIST
          │                   │                    │
          │                   │                    │
    YOUR MODEL           Open-source          Open-source
    fine-tuned           pretrained           pretrained
    / adapted              model                model
          │                   │                    │
          ▼                   ▼                    ▼
      VQA /              SAR analysis        Change detection
     Captioning                              Change VQA /
                                              description
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
                              ▼
                       Grounding specialist
                              │
                              ▼
                     ┌─────────────────┐
                     │ Result Fusion   │
                     │ + Evidence      │
                     │ + Confidence    │
                     └─────────────────┘
                     