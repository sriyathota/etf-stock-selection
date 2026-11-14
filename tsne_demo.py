import os, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE, trustworthiness
from tsne_features import build_feature_rows
from sklearn.manifold import TSNE
import inspect

# Optional: import your analyzer to fetch the dict
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

try:
    from src.pca.etf_analyzer import ETFAnalyzer
except Exception as e:
    print(f"Import Error: {e}")
    ETFAnalyzer = None

def _tsne_with_iters(n_components=2, perplexity=30, learning_rate="auto",
                     init="pca", metric="cosine", random_state=42,
                     early_exaggeration=12, n_iter=900):
    # Build kwargs that match the installed sklearn version
    kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        metric=metric,
        random_state=random_state,
        early_exaggeration=early_exaggeration,
    )
    sig = inspect.signature(TSNE.__init__)
    if "n_iter" in sig.parameters:
        kwargs["n_iter"] = n_iter
    elif "max_iter" in sig.parameters:
        kwargs["max_iter"] = n_iter
    # else: leave iterations at the library default
    return TSNE(**kwargs)

def compute_tsne(data_dict, perplexity=30, metric="cosine", seed=42):
    # Clean problematic tickers (e.g., BRK/B, --)
    for etf, holdings in list(data_dict.items()):
        for t in list(holdings.keys()):
            if t == "--" or "/" in t:
                holdings.pop(t, None)
    X, meta = build_feature_rows(data_dict)
    Xz = StandardScaler().fit_transform(X)
    tsne = _tsne_with_iters(perplexity=perplexity, learning_rate="auto",
                        init="pca", metric=metric, random_state=seed,
                        early_exaggeration=12, n_iter=900)
    Y = tsne.fit_transform(Xz)
    tw = trustworthiness(Xz, Y, n_neighbors=min(30, max(5, Xz.shape[0]//50)))
    meta = meta.reset_index(drop=True)
    emb = pd.DataFrame({"x": Y[:,0], "y": Y[:,1]}).join(meta)
    return emb, float(tw)

def scatter_by(emb, by="sector", title="t-SNE of ETF Holdings", out="src/tsne/tsne_results/tsne_scatter.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(figsize=(8,6))
    for v in sorted(emb[by].unique()):
        m = emb[by]==v
        plt.scatter(emb.loc[m,"x"], emb.loc[m,"y"], s=18, alpha=0.85, label=v)
    plt.legend(title=by, bbox_to_anchor=(1.02,1), loc="upper left")
    plt.xlabel("tSNE-1"); plt.ylabel("tSNE-2")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    print(f"saved: {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--metric", default="cosine", choices=["cosine","euclidean"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--colorby", default="sector", choices=["sector","etf","industry"])
    ap.add_argument("--etfs", nargs="+", default=["SPY","VTI","QQQ","VOO"])
    args = ap.parse_args()

    if ETFAnalyzer:
        analyzer = ETFAnalyzer()
        # This mirrors your PCA demo pattern: assemble the dict via analyzer
        data_dict = analyzer.run_complete_analysis(args.etfs)
    else:
        raise SystemExit("ETFAnalyzer not importable here; pass a prepared dict instead.")

    emb, tw = compute_tsne(data_dict, perplexity=args.perplexity, metric=args.metric, seed=args.seed)
    print(f"Trustworthiness: {tw:.3f}")
    scatter_by(emb, by=args.colorby, title=f"t-SNE ({args.metric}, perp={args.perplexity})", out=f"src/tsne/tsne_results/tsne_{args.colorby}_{args.metric}_p{args.perplexity}.png")
    emb.to_csv("src/tsne/tsne_results/tsne_embedding.csv", index=False)
    print("saved: src/tsne/tsne_results/tsne_embedding.csv")

if __name__ == "__main__":
    main()
