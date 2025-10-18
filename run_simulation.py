from analyze import run_all
import pandas as pd
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results.pkl", help="output .pkl path")
    args = ap.parse_args()

    df = run_all()
    df = df.rename(columns={
        "n_sim": "nsim",
        "aspect_ratio": "gamma",
        "MSE": "MSE_hat",
        "MCSE": "se",
    })[["nsim", "gamma", "MSE_hat", "se"]].copy()
    df.loc[df["nsim"] == 1, "se"] = 0.0

    df.to_pickle(args.out)
    print(f"Saved {args.out} with columns: {list(df.columns)}")

if __name__ == "__main__":
    main()
