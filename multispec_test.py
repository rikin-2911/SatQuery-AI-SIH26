import torch

from specialist_models.multispec_vlm_agent import (
    MultispectralVLM
)


MODEL_PATH = (
    "/home/rikin/satquery-ai/checkpoint_6000"
)

SPECTRAL_PATH = (
    "/home/rikin/satquery-ai/checkpoint_6000/spectral_adapter.pt"
)


def main():

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------

    agent = MultispectralVLM(
        model_path=MODEL_PATH,
        spectral_adapter_path=SPECTRAL_PATH,
    )

    # --------------------------------------------------
    # Temporary test input
    #
    # IMPORTANT:
    # Replace this with a REAL 12-band
    # training-normalized patch.
    # --------------------------------------------------

    patch = torch.randn(
        12,
        256,
        256,
    )

    # --------------------------------------------------
    # Run
    # --------------------------------------------------

    result = agent.run(
        image=patch,
        query=(
            "What type of land and explain it?"
        ),
    )

    print("\nRESULT")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()