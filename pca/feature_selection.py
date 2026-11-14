import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#df = the dataset
#stocks_selected = the number of stocks we would like to select from PCA (Default = 5)
def get_pca_feature_indices(df, stocks_selected=5):
    # Standardize return matrix for PCA
    X = StandardScaler().fit_transform(df)

    # Performing PCA
    pca = PCA()
    pca.fit(X)

    #Parametrizing the number of n components to consider for the feature selection
    #This is supposed be a hyperparameter since the number of components to consider will change the extent of the original dataset's variance we are considering
    components_to_consider = 3
    loadings = pca.components_[:components_to_consider, :]
    importance_scores = np.sum(np.square(loadings), axis=0) #Sum of importance scores in squares

    # Get indices of features with highest importance
    indices = np.argsort(importance_scores)[::-1][:stocks_selected] #Array manipulation to get stocks which socred the highest weights
    return indices

#Example usage
if __name__ == "__main__":
    # Relevant dataset:
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AVGO', 'COST', 'ADBE']

    # Fetch and process data
    raw_data = yf.download(tickers, start="2023-01-01", end="2024-01-01")['Close']
    returns_df = raw_data.pct_change().dropna()

    # Execution
    selected_idx = get_pca_feature_indices(returns_df, stocks_selected=5)

    print(f"Indices of Selected Features: {selected_idx}")
    print(f"Top 5 Tickers identified: {returns_df.columns[selected_idx].tolist()}")
