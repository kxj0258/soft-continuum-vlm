"""Scaffold for later adapter training."""


def main() -> None:
    print(
        "Deferred adapter training. Expected input: scripted demonstration dataset "
        "and deterministic feature tensors. Expected output: SoftEmbodimentAdapter "
        "checkpoint and metrics. Integration path: add a torch training loop after "
        "Milestone 5 without downloading external VLM or VLA weights."
    )


if __name__ == "__main__":
    main()
