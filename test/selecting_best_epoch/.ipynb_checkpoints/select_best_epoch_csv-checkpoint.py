import re
import pandas as pd

REQUIRED = ["delta_e", "uv_ssim", "uv_psnr"]

# Fallback for plain-Python use (Jupyter's display() doesn't exist there)
try:
    display
except NameError:
    def display(x):
        print(x.to_string())


def load_csv(path_or_df, model=None):
    df = pd.read_csv(path_or_df) if isinstance(path_or_df, str) else path_or_df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED + ["epoch"] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    if "checkpoint" in df.columns:
        df["model"] = df["checkpoint"].str.replace(
            r"[_-]?epoch[_-]?\d+$", "", regex=True, flags=re.IGNORECASE
        )
        if model is not None:
            df = df[df["model"] == model]
            if df.empty:
                raise ValueError(f"No rows for model {model!r}")
        elif df["model"].nunique() > 1:
            raise ValueError(
                f"CSV contains multiple models {sorted(df['model'].unique())} — "
                f"pass model=... or use select_for_all()."
            )

    df = df.set_index("epoch").sort_index()
    df.index.name = "epoch"
    return df.dropna(subset=REQUIRED)


def select_best_epoch(path_or_df, model=None, tol_de=0.1, tol_ssim=0.002,
                      weights=(0.5, 0.3, 0.2), top_n=10, verbose=True):
    df = load_csv(path_or_df, model=model) \
        if isinstance(path_or_df, str) or "epoch" in getattr(path_or_df, "columns", []) \
        else path_or_df
    if df.empty:
        raise ValueError("No complete epoch metrics found — check the CSV.")

    # Rule 1: lexicographic with tolerance (Delta E > UV_SSIM > UV_PSNR)
    c = df[df.delta_e <= df.delta_e.min() + tol_de]
    c2 = c[c.uv_ssim >= c.uv_ssim.max() - tol_ssim]
    best_lex = c2.uv_psnr.idxmax()

    # Rule 2: rank-weighted score
    w_de, w_ssim, w_psnr = weights
    score = (w_de * df.delta_e.rank(ascending=True)
             + w_ssim * df.uv_ssim.rank(ascending=False)
             + w_psnr * df.uv_psnr.rank(ascending=False))
    best_rank = score.idxmin()

    top = df.loc[score.nsmallest(top_n).index, REQUIRED].copy()
    top["rank_score"] = score

    if verbose:
        cands = sorted({df.delta_e.idxmin(), df.uv_ssim.idxmax(), df.uv_psnr.idxmax()})
        print(f"Parsed {len(df)} epochs")
        print("\nPer-metric bests:")
        print(f"  Delta E : {df.delta_e.min():.4f} @ {df.delta_e.idxmin()}")
        print(f"  UV_SSIM : {df.uv_ssim.max():.4f} @ {df.uv_ssim.idxmax()}")
        print(f"  UV_PSNR : {df.uv_psnr.max():.4f} @ {df.uv_psnr.idxmax()}")
        print("\nCandidates:")
        display(df.loc[cands, REQUIRED])
        print(f"Lexicographic pick : {best_lex}")
        print(f"Rank-weighted pick : {best_rank}")
        print(f"\nTop {top_n} by rank score:")
        display(top)
        if best_lex == best_rank:
            print(f">>> SELECTED EPOCH: {best_lex} (both rules agree)")
        else:
            print(f">>> Rules disagree — using lexicographic pick: {best_lex} "
                  f"(rank-weighted was {best_rank})")
    return best_lex, df, top


if __name__ == "__main__":
    best, df, top = select_best_epoch("validation_checkpoint_metrics.csv")
