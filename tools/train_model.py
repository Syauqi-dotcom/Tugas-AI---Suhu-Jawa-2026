"""
tools/train_model.py
====================
Regresi Multivariat dari Scratch menggunakan NumPy + Pandas.
Optimizer: EXDM (Exponential Decay Momentum)

EXDM Optimizer:
  Menggabungkan Momentum, Adaptive Learning Rate (seperti RMSProp),
  dan Exponential Decay untuk konvergensi stabil pada data iklim
  yang bersifat non-stasioner.

  Update rule:
    v_t  = β₁·v_{t-1} + (1-β₁)·∇L          (momentum)
    s_t  = β₂·s_{t-1} + (1-β₂)·∇L²         (squared gradient)
    η_t  = η₀ · exp(-λ·t)                    (exponential decay lr)
    w_t  = w_{t-1} - η_t · v_t / (√s_t + ε) (update)

Referensi arsitektur:
  - Climate-Invariant ML: arxiv.org/abs/2112.08440 (Beucler et al.)
  - Multivariate Regression CORDEX: Ghaemi et al. (2023)
  - EDCDF + Regression baseline: arxiv.org/html/2504.19145v2
"""

import argparse
import os
import numpy as np
import pandas as pd
import json


# ═══════════════════════════════════════════════════════════════
#  EXDM OPTIMIZER
# ═══════════════════════════════════════════════════════════════

class EXDMOptimizer:
    """
    EXDM: Exponential Decay Momentum Optimizer
    Dirancang untuk konvergensi stabil pada data iklim musiman.

    Parameter:
      lr      : learning rate awal (η₀)
      beta1   : koefisien momentum (β₁), default 0.9
      beta2   : koefisien adaptive (β₂), default 0.999
      decay   : laju decay eksponensial (λ), default 1e-4
      epsilon : numerator kecil (ε), default 1e-8
    """

    def __init__(self, lr: float = 0.01, beta1: float = 0.9,
                 beta2: float = 0.999, decay: float = 1e-4,
                 epsilon: float = 1e-8):
        self.lr      = lr
        self.beta1   = beta1
        self.beta2   = beta2
        self.decay   = decay
        self.epsilon = epsilon
        self._step   = 0
        self._v      = None  # momentum
        self._s      = None  # squared gradient accumulator

    def init(self, shape):
        self._v = np.zeros(shape)
        self._s = np.zeros(shape)

    def update(self, weights: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Satu langkah update EXDM."""
        if self._v is None:
            self.init(weights.shape)

        self._step += 1
        t = self._step

        # Exponential decay learning rate
        lr_t = self.lr * np.exp(-self.decay * t)

        # Momentum update
        self._v = self.beta1 * self._v + (1 - self.beta1) * gradient

        # Squared gradient accumulator
        self._s = self.beta2 * self._s + (1 - self.beta2) * gradient**2

        # Bias correction
        v_hat = self._v / (1 - self.beta1**t)
        s_hat = self._s / (1 - self.beta2**t)

        # Weight update
        weights_new = weights - lr_t * v_hat / (np.sqrt(s_hat) + self.epsilon)
        return weights_new

    def reset(self):
        self._step = 0
        self._v    = None
        self._s    = None


# ═══════════════════════════════════════════════════════════════
#  MULTIVARIATE REGRESSION MODEL
# ═══════════════════════════════════════════════════════════════

class MultivariateRegressionModel:
    """
    Regresi Linear Multivariat dengan EXDM Optimizer.
    Implementasi dari scratch menggunakan NumPy.

    Model: y = X·w + b
    Loss : MSE = (1/n) Σ (y_pred - y_true)²
    """

    def __init__(self, optimizer: EXDMOptimizer, l2_lambda: float = 1e-4):
        self.optimizer = optimizer
        self.l2_lambda = l2_lambda  # Regularisasi L2 (Ridge)
        self.weights   = None
        self.bias      = None
        self.loss_history = []
        self.val_loss_history = []

    def _init_weights(self, n_features: int):
        """Inisialisasi bobot dengan He initialization."""
        scale = np.sqrt(2.0 / n_features)
        self.weights = np.random.randn(n_features) * scale
        self.bias    = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias

    def _compute_loss(self, y_pred: np.ndarray,
                      y_true: np.ndarray) -> float:
        mse  = np.mean((y_pred - y_true)**2)
        l2   = self.l2_lambda * np.sum(self.weights**2)
        return mse + l2

    def _compute_gradients(self, X: np.ndarray, y_pred: np.ndarray,
                           y_true: np.ndarray) -> tuple:
        n = len(y_true)
        residual = y_pred - y_true

        grad_w = (2 / n) * (X.T @ residual) + 2 * self.l2_lambda * self.weights
        grad_b = (2 / n) * np.sum(residual)
        return grad_w, grad_b

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            epochs: int = 500, batch_size: int = 32,
            verbose: bool = True, patience: int = 30) -> "MultivariateRegressionModel":
        """
        Training dengan mini-batch SGD dan EXDM optimizer.
        Menggunakan early stopping berdasarkan validation loss.
        """
        n, p = X_train.shape
        self._init_weights(p)
        self.optimizer.reset()

        best_val_loss = np.inf
        best_weights  = self.weights.copy()
        best_bias     = self.bias
        patience_counter = 0

        for epoch in range(epochs):
            # Shuffle untuk mini-batch
            idx = np.random.permutation(n)
            X_shuf = X_train[idx]
            y_shuf = y_train[idx]

            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                Xb  = X_shuf[start:end]
                yb  = y_shuf[start:end]

                y_pred  = self.predict(Xb)
                loss    = self._compute_loss(y_pred, yb)
                grad_w, grad_b = self._compute_gradients(Xb, y_pred, yb)

                self.weights = self.optimizer.update(self.weights, grad_w)
                self.bias   -= 0.01 * grad_b  # bias sederhana

                epoch_loss += loss
                n_batches  += 1

            train_loss = epoch_loss / n_batches
            self.loss_history.append(train_loss)

            # Validation loss
            val_loss = None
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                val_loss = self._compute_loss(y_val_pred, y_val)
                self.val_loss_history.append(val_loss)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_weights     = self.weights.copy()
                    best_bias        = self.bias
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        if verbose:
                            print(f"  [Early Stop] Epoch {epoch+1} | val_loss={val_loss:.4f}")
                        break

            if verbose and (epoch + 1) % 50 == 0:
                msg = f"  Epoch {epoch+1:4d}/{epochs} | train_loss={train_loss:.4f}"
                if val_loss is not None:
                    msg += f" | val_loss={val_loss:.4f}"
                print(msg)

        # Restore best weights
        self.weights = best_weights
        self.bias    = best_bias
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Hitung metrik evaluasi lengkap."""
        y_pred = self.predict(X)
        residuals = y - y_pred
        n = len(y)

        mse  = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae  = np.mean(np.abs(residuals))

        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Nash-Sutcliffe Efficiency (NSE) — digunakan luas di iklim
        nse = 1 - ss_res / ss_tot if ss_tot > 0 else -np.inf

        # Mean Bias Error
        mbe = np.mean(y_pred - y)

        # Pearson Correlation
        corr = np.corrcoef(y_pred, y)[0, 1]

        return {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2": float(r2),
            "NSE": float(nse),
            "MBE": float(mbe),
            "Pearson_r": float(corr),
        }

    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """Hitung feature importance berdasarkan magnitude bobot ternormalisasi."""
        abs_weights = np.abs(self.weights)
        importance  = abs_weights / abs_weights.sum()
        df = pd.DataFrame({
            "feature":    feature_names,
            "weight":     self.weights,
            "importance": importance,
        }).sort_values("importance", ascending=False)
        return df

    def save(self, path: str):
        """Simpan model ke disk (JSON + NPZ)."""
        os.makedirs(path, exist_ok=True)
        np.savez(
            os.path.join(path, "weights.npz"),
            weights=self.weights,
            bias=np.array([self.bias]),
            loss_history=np.array(self.loss_history),
            val_loss_history=np.array(self.val_loss_history),
        )
        meta = {
            "l2_lambda":   self.l2_lambda,
            "n_features":  int(len(self.weights)),
            "final_train_loss": float(self.loss_history[-1]) if self.loss_history else None,
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        print(f"  [SAVE] Model disimpan ke: {path}")

    @classmethod
    def load(cls, path: str) -> "MultivariateRegressionModel":
        """Muat model dari disk."""
        data = np.load(os.path.join(path, "weights.npz"))
        with open(os.path.join(path, "meta.json")) as f:
            meta = json.load(f)

        optimizer = EXDMOptimizer()
        model = cls(optimizer=optimizer, l2_lambda=meta["l2_lambda"])
        model.weights = data["weights"]
        model.bias    = float(data["bias"][0])
        model.loss_history = list(data["loss_history"])
        model.val_loss_history = list(data["val_loss_history"])
        return model


# ═══════════════════════════════════════════════════════════════
#  DATA NORMALIZER (Z-score, NumPy scratch)
# ═══════════════════════════════════════════════════════════════

class ZScoreNormalizer:
    """Normalisasi Z-score dari scratch (tanpa sklearn)."""

    def __init__(self):
        self.mean_ = None
        self.std_  = None

    def fit(self, X: np.ndarray) -> "ZScoreNormalizer":
        self.mean_ = np.nanmean(X, axis=0)
        self.std_  = np.nanstd(X, axis=0)
        self.std_[self.std_ == 0] = 1.0  # hindari division by zero
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)



    def save(self, path: str):
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> "ZScoreNormalizer":
        data = np.load(path)
        norm = cls()
        norm.mean_ = data["mean"]
        norm.std_  = data["std"]
        return norm


# ═══════════════════════════════════════════════════════════════
#  MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Training Regresi Multivariat + EXDM")
    parser.add_argument("--train_X",  required=True)
    parser.add_argument("--train_y",  required=True)
    parser.add_argument("--val_X",    required=True)
    parser.add_argument("--val_y",    required=True)
    parser.add_argument("--output",   required=True, help="Direktori simpan model")
    parser.add_argument("--optimizer",default="exdm", choices=["exdm"])
    parser.add_argument("--epochs",   type=int,   default=500)
    parser.add_argument("--lr",       type=float, default=0.01)
    parser.add_argument("--l2",       type=float, default=1e-4)
    parser.add_argument("--seed",     type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    print("\n=== Training: Regresi Multivariat + EXDM ===")

    # Load data
    df_train_X = pd.read_csv(args.train_X)
    df_train_y = pd.read_csv(args.train_y)
    df_val_X   = pd.read_csv(args.val_X)
    df_val_y   = pd.read_csv(args.val_y)

    # Drop kolom non-numerik
    drop_cols = ["time", "Unnamed: 0"]
    X_train = df_train_X.drop(columns=[c for c in drop_cols if c in df_train_X]).values
    X_val   = df_val_X.drop(columns=[c for c in drop_cols if c in df_val_X]).values

    target_col = "temp_2m"
    y_train = df_train_y[target_col].values if target_col in df_train_y else df_train_y.iloc[:, 0].values
    y_val   = df_val_y[target_col].values   if target_col in df_val_y   else df_val_y.iloc[:, 0].values

    feature_names = [c for c in df_train_X.columns if c not in drop_cols]
    print(f"  Fitur: {len(feature_names)} | Train: {len(y_train)} | Val: {len(y_val)}")

    # Normalisasi
    norm_X = ZScoreNormalizer()
    X_train_n = norm_X.fit_transform(X_train)
    X_val_n   = norm_X.transform(X_val)

    norm_y = ZScoreNormalizer()
    y_mean = np.mean(y_train)
    y_std  = np.std(y_train)
    y_train_n = (y_train - y_mean) / y_std
    y_val_n   = (y_val   - y_mean) / y_std

    # Inisialisasi EXDM
    optimizer = EXDMOptimizer(
        lr=args.lr, beta1=0.9, beta2=0.999, decay=1e-4
    )

    # Training
    model = MultivariateRegressionModel(optimizer=optimizer, l2_lambda=args.l2)
    model.fit(X_train_n, y_train_n, X_val_n, y_val_n,
              epochs=args.epochs, verbose=True)

    # Evaluasi pada data val (skala asli)
    y_val_pred_n = model.predict(X_val_n)
    y_val_pred   = y_val_pred_n * y_std + y_mean
    metrics = model.evaluate(X_val_n, y_val_n)

    print(f"\n  --- Metrik Validasi ---")
    for k, v in metrics.items():
        print(f"    {k:12s} = {v:.4f}")

    # Feature importance
    fi = model.get_feature_importance(feature_names)
    print(f"\n  --- Top 10 Feature Importance ---")
    print(fi.head(10).to_string(index=False))

    # Simpan model + normalizer
    os.makedirs(args.output, exist_ok=True)
    model.save(args.output)
    norm_X.save(os.path.join(args.output, "normalizer_X.npz"))
    np.save(os.path.join(args.output, "target_stats.npy"),
            np.array([y_mean, y_std]))
    fi.to_csv(os.path.join(args.output, "feature_importance.csv"), index=False)

    # Simpan metrik
    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output, "val_metrics.csv"), index=False
    )

    print(f"\n  [SELESAI] Model tersimpan: {args.output}")


if __name__ == "__main__":
    main()
