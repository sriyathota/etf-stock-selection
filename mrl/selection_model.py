import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. IMPORT SHARED ARCHITECTURE
# ==========================================
from matryoshka_utils import Matryoshka_Linear_Layer

# ==========================================
# 2. DATA COLLECTION & PREP
# ==========================================
tickers_v2 = [
    # --- TOP 150 REPRESENTATIVES ---
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'BRK-B', 'LLY',
    'JPM', 'UNH', 'V', 'MA', 'XOM', 'HD', 'PG', 'JNJ', 'COST', 'ORCL', 'ABBV', 'CVX',
    'ADBE', 'BAC', 'CRM', 'KO', 'PEP', 'AMD', 'TMO', 'NFLX', 'WMT', 'ACN', 'LIN', 'PM',
    'DIS', 'CSCO', 'TXN', 'ABT', 'INTC', 'DHR', 'PFE', 'IBM', 'VZ', 'QCOM', 'CAT', 'NEU',
    'AMGN', 'INTU', 'HON', 'NOW', 'GE', 'LOW', 'AXP', 'SPGI', 'AMAT', 'COP', 'UNP', 'GS',
    'SYK', 'ISRG', 'TJX', 'NKE', 'ELV', 'MS', 'RTX', 'LMT', 'BLK', 'BKNG', 'MDLZ', 'TJX',
    'PGR', 'ADI', 'ADP', 'C', 'MMC', 'GILD', 'CI', 'PLD', 'DE', 'AMT', 'LRCX', 'T', 'BSX',
    'VRTX', 'ZTS', 'ETN', 'CB', 'MU', 'MCD', 'REGN', 'PANW', 'SNPS', 'CDNS', 'WM', 'KLAC',
    'MO', 'EOG', 'ITW', 'HCA', 'ICE', 'MCK', 'MDT', 'SHW', 'ORLY', 'TGT', 'BDX', 'APH',
    'SLB', 'MAR', 'CMCSA', 'PH', 'ABNB', 'CRWD', 'AIG', 'TRV', 'AON', 'WELL', 'DLR', 'ROP',
    'ADSK', 'A', 'CTAS', 'EMR', 'PCAR', 'MPC', 'O', 'GD', 'TDG', 'BKR', 'KMB', 'STZ', 'D',
    'PAYX', 'CSX', 'FIS', 'NSC', 'NOC', 'ED', 'ECL', 'SO', 'DDUK', 'MET', 'PSX', 'EXC',

    # --- 100 MID-TIER / SECTOR DIVERSITY ---
    'KMI', 'DAL', 'UAL', 'AAL', 'MAR', 'HLT', 'RCL', 'CCL', 'MGM', 'WYNN', 'LUV', 'F',
    'GM', 'STLA', 'NCLH', 'BK', 'TFC', 'USB', 'PNC', 'COF', 'FITB', 'MTB', 'HBAN', 'KEY',
    'DFS', 'SYF', 'STT', 'WFC', 'RF', 'CFG', 'KEY', 'ALLY', 'SCHW', 'RJF', 'LPLA', 'AMP',
    'PRU', 'MET', 'AFL', 'PFG', 'L', 'HIG', 'CNA', 'TRV', 'RE', 'ACGL', 'WRB', 'EG',
    'DOW', 'LYB', 'CE', 'EMN', 'ALB', 'FMC', 'VMC', 'MLM', 'SUM', 'EXP', 'NEM', 'FCX',
    'AA', 'NUE', 'STLD', 'RS', 'CLF', 'X', 'DVN', 'FANG', 'MRO', 'APA', 'OXY', 'HAL',
    'SLB', 'BKR', 'NOV', 'FTI', 'OII', 'PXD', 'HES', 'VLO', 'MPC', 'PSX', 'PBF', 'DK',
    'HOG', 'PII', 'DOV', 'XYL', 'GWW', 'FAST', 'URI', 'HWM', 'TT', 'CARR', 'OTIS', 'JCI'
]

print(f"Downloading data for {len(tickers_v2)} stocks...")
raw_data_v2 = yf.download(tickers_v2, start="2019-01-01", end="2024-01-01", auto_adjust=True)['Close']
raw_data_v2 = raw_data_v2.dropna(axis=1, thresh=int(len(raw_data_v2) * 0.9))
raw_data_v2 = raw_data_v2.ffill().bfill()

returns_df_v2 = raw_data_v2.pct_change().dropna()
final_tickers_v2 = returns_df_v2.columns.tolist()

market_mean = returns_df_v2.mean(axis=1)
target_v2 = (market_mean > 0).astype(int)

scaler_v2 = StandardScaler()
X_scaled_v2 = scaler_v2.fit_transform(returns_df_v2.values)

X_tensor_v2 = torch.tensor(X_scaled_v2, dtype=torch.float32)
y_tensor_v2 = torch.tensor(target_v2.values, dtype=torch.long)

dataset_v2 = TensorDataset(X_tensor_v2, y_tensor_v2)
train_loader_v2 = DataLoader(dataset_v2, batch_size=64, shuffle=True)

print("-" * 30)
print(f"v2 Data Ready: {X_tensor_v2.shape[0]} days, {len(final_tickers_v2)} stocks.")
print(f"Input Dimension for model: {len(final_tickers_v2)}")

# ==========================================
# 3. MODEL DEFINITION
# ==========================================
class Stock_Matryoshka_Model_V2(nn.Module):
    def __init__(self, input_dim, nesting_list):
        super(Stock_Matryoshka_Model_V2, self).__init__()
        
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128), 
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU()
        )

        # FIXED: Arguments match matryoshka_utils.py
        self.mrl_layer = Matryoshka_Linear_Layer(
            input_dim=128,
            output_dim=2,
            nesting_list=nesting_list,
            efficient=True
        )

    def forward(self, x):
        z = self.backbone(x)
        return self.mrl_layer(z)

stock_nesting_list = [2, 4, 8, 16, 32, 64, 128]
input_dim_v2 = len(final_tickers_v2)

model_v2 = Stock_Matryoshka_Model_V2(
    input_dim=input_dim_v2,
    nesting_list=stock_nesting_list
)

print(f"✅ Model v2 initialized with input dimension: {input_dim_v2}")
print(f"Nesting levels: {stock_nesting_list}")

# ==========================================
# 4. TRAINING LOOP
# ==========================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_v2.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

print(f"Starting MRL v2 Training on {len(final_tickers_v2)} Stocks...")

num_epochs = 60  
model_v2.train()

for epoch in range(num_epochs):
    total_epoch_loss = 0

    for returns, labels in train_loader_v2:
        optimizer.zero_grad()
        nesting_logits = model_v2(returns)
        
        loss = 0
        for logits in nesting_logits:
            loss += criterion(logits, labels)

        loss.backward()
        optimizer.step()
        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader_v2)
    scheduler.step(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Combined MRL Loss: {avg_loss:.4f}")

print("✅ Training Complete! Model v2 is now optimized for hierarchical stock selection.")

# ==========================================
# 5. BENCHMARKING & SELECTION
# ==========================================
test_indices = {
    "S&P 100": "^OEX",
    "Nasdaq-100": "^NDX",
    "Dow Jones": "^DJI",
    "S&P 500 Equal Weight": "^SPXEW"
}

def evaluate_reconstruction_v3(index_name, target_ticker, n_representative=10):
    print(f"\n🚀 Testing {index_name} ({target_ticker})...")

    idx_df = yf.download(target_ticker, start="2019-01-01", end="2024-01-01", auto_adjust=True)['Close']

    if isinstance(idx_df, pd.DataFrame):
        idx_series = idx_df.iloc[:, 0]
    else:
        idx_series = idx_df

    benchmark_returns = idx_series.pct_change().dropna()
    common_dates = returns_df_v2.index.intersection(benchmark_returns.index)
    benchmark_returns = benchmark_returns.loc[common_dates]
    full_universe_subset = returns_df_v2.loc[common_dates]

    model_v2.eval()
    with torch.no_grad():
        weights = model_v2.backbone[0].weight.t().cpu().numpy()
        latent_dna = weights[:, :16]

    kmeans = KMeans(n_clusters=n_representative, n_init=10, random_state=42)
    clusters = kmeans.fit_predict(latent_dna)
    centroids = kmeans.cluster_centers_

    selected_indices = []
    for i in range(n_representative):
        cluster_pts = np.where(clusters == i)[0]
        dist = np.linalg.norm(latent_dna[cluster_pts] - centroids[i], axis=1)
        selected_indices.append(cluster_pts[np.argmin(dist)])

    selected_tickers = [final_tickers_v2[i] for i in selected_indices]
    portfolio_returns = full_universe_subset[selected_tickers].mean(axis=1)

    np.random.seed(42)
    random_indices = np.random.choice(len(final_tickers_v2), n_representative, replace=False)
    random_tickers = [final_tickers_v2[i] for i in random_indices]
    random_returns = full_universe_subset[random_tickers].mean(axis=1)

    def get_stats(target, pred):
        X = target.values.reshape(-1, 1)
        y = pred.values
        reg = LinearRegression().fit(X, y)
        r2 = reg.score(X, y)
        te = (pred - target).std() * np.sqrt(252)
        corr = np.corrcoef(target, pred)[0, 1]
        return r2, te, corr

    m_r2, m_te, m_corr = get_stats(benchmark_returns, portfolio_returns)
    r_r2, r_te, r_corr = get_stats(benchmark_returns, random_returns)

    print(f"✅ Matryoshka Tickers: {selected_tickers}")
    print(f"📊 Fidelity (R²): {m_r2:.4f} (Random: {r_r2:.4f})")
    print(f"📉 Tracking Error: {m_te:.2%} (Random: {r_te:.2%})")

    return {
        "Index": index_name,
        "Model_R2": m_r2,
        "Random_R2": r_r2,
        "Model_TE": m_te,
        "Corr": m_corr
    }

results = []
for name, ticker in test_indices.items():
    try:
        results.append(evaluate_reconstruction_v3(name, ticker))
    except Exception as e:
        print(f"❌ Error testing {name}: {e}")

summary_df = pd.DataFrame(results)
print("\n" + "="*60)
print("FINAL PERFORMANCE BENCHMARK: MATRYOSHKA vs. RANDOM")
print("="*60)
print(summary_df[["Index", "Model_R2", "Random_R2", "Model_TE", "Corr"]].to_string(index=False))
