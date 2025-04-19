"""
Front-end dispatcher for all pipeline commands.
"""
import sys, inspect

def _usage():
    print(
        "Usage:\n"
        "  amd_dca <command> [options]\n\n"
        "Commands:\n"
        "  run_preprocessing      Load raw counts + metadata + mapping, filter, split, save processed arrays.\n"
        "  ml_baseline_1          Run PCA -> KNN-denoiser baseline.\n"
        "  ml_baseline_2          Run RandomForest regressor baseline.\n"
        "  train_ae               Train Negative-Binomial Autoencoder.\n"
        "  train_vae              Train VAE.\n"
        "  infer_ae               Generate AE‑denoised test matrix.\n"
        "  infer_vae              Generate VAE‑denoised test matrix.\n"
        "  run_dge                Differential expression (DESeq2) on chosen input.\n"
        "  run_evaluation         Generate PCA/UMAP plots for raw vs denoised.\n",
        file=sys.stderr,
    )
    sys.exit(1)

def main():
    if len(sys.argv) < 2:
        _usage()

    cmd, *args = sys.argv[1:]
    modules = {
        "run_preprocessing": "amd_dca.scripts.run_preprocessing",
        "ml_baseline_1":     "amd_dca.scripts.run_ml_baseline_1",
        "ml_baseline_2":     "amd_dca.scripts.run_ml_baseline_2",
        "train_ae":          "amd_dca.scripts.run_dl_autoencoder",
        "train_vae":         "amd_dca.scripts.run_dl_vae",
        "infer_ae":          "amd_dca.scripts.run_infer_ae",
        "infer_vae":         "amd_dca.scripts.run_infer_vae",
        "run_dge":           "amd_dca.scripts.run_dge",
        "run_evaluation":    "amd_dca.scripts.run_evaluation",
    }

    if cmd not in modules:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        _usage()

    mod = __import__(modules[cmd], fromlist=["main"])
    sig = inspect.signature(mod.main)
    if len(sig.parameters) == 0:
        mod.main()
    else:
        mod.main(args)

if __name__ == "__main__":
    main()