"""
Front‑end dispatcher that forwards CLI args to driver scripts.

It checks the driver's `main` signature and calls it with or without the argv list
"""
from __future__ import annotations
import sys, inspect

def _usage() -> None:
    print(
        "Usage:\n"
        "  amd_dca <command> [options]\n\n"
        "Commands:\n"
        "  preprocess   run data preprocessing (pass --combat, etc.)\n"
        "  train        train the autoencoder\n"
        "  evaluate     generate denoised matrix + PCA/UMAP\n"
        "  dge          run DESeq2 / edgeR comparisons\n",
        file=sys.stderr,
    )
    sys.exit(1)

def main() -> None:
    if len(sys.argv) < 2:
        _usage()

    cmd, *cmd_args = sys.argv[1:]

    module_lookup = {
        "preprocess": "amd_dca.scripts.run_preprocessing",
        "train":      "amd_dca.scripts.run_training",
        "evaluate":   "amd_dca.scripts.run_evaluation",
        "dge":        "amd_dca.scripts.run_dge",
    }
    if cmd not in module_lookup:
        print(f"Unknown command: {cmd!r}", file=sys.stderr)
        _usage()

    mod = __import__(module_lookup[cmd], fromlist=["main"])

    # Call driver.main with or without argv depending on its signature
    sig = inspect.signature(mod.main)
    if len(sig.parameters) == 0:
        mod.main()
    else:
        mod.main(cmd_args)

if __name__ == "__main__":
    main()