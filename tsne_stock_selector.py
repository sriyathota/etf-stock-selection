import os, argparse
import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from tsne_features import build_feature_rows
import inspect

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

def tsne_embed(X, perplexity=30, metric="cosine", seed=42):
    Xz = StandardScaler().fit_transform(X)
    Y = _tsne_with_iters(perplexity=perplexity, learning_rate="auto",
                     init="pca", metric=metric, random_state=seed,
                     early_exaggeration=12, n_iter=900).fit_transform(Xz)
    return Y

def select_k_from_embedding(emb_df, k=5, prefer_column=None):
    """
    emb_df columns: ['x','y','ticker','etf',...]
    prefer_column can be 'weight' if you add it later; otherwise picks centroid-closest
    """
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(emb_df[["x","y"]].to_numpy())
    emb_df = emb_df.assign(cluster=labels)
    picks = []
    for c in range(k):
        g = emb_df[emb_df.cluster==c].copy()
        if prefer_column and prefer_column in g.columns:
            # highest weight inside cluster
            picks.append(g.sort_values(prefer_column, ascending=False).iloc[0]["ticker"])
        else:
            ctr = km.cluster_centers_[c]
            idx = np.argmin(np.sum((g[["x","y"]].to_numpy()-ctr)**2, axis=1))
            picks.append(g.iloc[idx]["ticker"])
    return picks, emb_df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--etf", default="SPY")
    ap.add_argument("--perplexity", type=int, default=30)
    ap.add_argument("--metric", default="cosine")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

    # Build dict using your analyzer (same pattern as demo)
    from src.pca.etf_analyzer import ETFAnalyzer
    analyzer = ETFAnalyzer()
    data_dict = analyzer.run_complete_analysis([args.etf])

    X, meta = build_feature_rows(data_dict)
    Y = tsne_embed(X, perplexity=args.perplexity, metric=args.metric)
    emb = meta.reset_index(drop=True).copy()
    emb["x"], emb["y"] = Y[:,0], Y[:,1]

    picks, emb_labeled = select_k_from_embedding(emb, k=args.k, prefer_column=None)
    print(f"{args.etf} t-SNE representatives ({args.k}): {', '.join(picks)}")

    os.makedirs("src/tsne/tsne_results", exist_ok=True)
    emb_labeled.to_csv(f"src/tsne/tsne_results/{args.etf}_tsne_embedding_labeled.csv", index=False)
    with open(f"src/tsne/tsne_results/{args.etf}_tsne_top_{args.k}.txt","w") as f:
        f.write("\n".join(map(str,picks)))
    print("saved: labeled embedding + picks under src/tsne/tsne_results/")

if __name__ == "__main__":
    main()
