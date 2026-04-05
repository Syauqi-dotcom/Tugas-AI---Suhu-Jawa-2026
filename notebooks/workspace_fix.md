# Pemodelan Prediksi Suhu Permukaan Jawa
## Regresi Multivariat 

---

**Dataset**: CORDEX South-East Asia (SEA)  
**Sumber**: Climate Data Store (CDS) Copernicus — https://cds.climate.copernicus.eu/  

---

### 0. Setup Function dan Library

Disini kita bakal setup seluruh library yang digunakan, serta membuat class untuk function yang nantinya akan dipanggil model untuk melakukan regresi terhadap iklim JAWA:

- Multivariate Regresi : Model Regresi dengan metode analitik yang digunakan
- ECDFBiasCorrector : Handling Bias yang digunakan [Subramani, dkk](https://arxiv.org/abs/2504.19145) untuk Global Climate Model
- ZScore : untuk melakukan normalisasi model


```python
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec

class MultivariateRegressionModel:
    def __init__(self, l2_lambda=1e-4):
        self.l2_lambda = l2_lambda
        self.weights = None
        self.bias = None

    def fit(self, X_train, y_train):
        """
        Latih Model Regresi Ridge dengan data historis dan Normal Equation
        
        Parameters:
        X_train : Matriks data fitur cuaca
        y_train : Vektor data suhu asli 
        """
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
        I = np.eye(X_design.shape[1])
        I[0,0] = 0

        theta = np.linalg.pinv(X_design.T @ X_design + self.l2_lambda * I) @ X_design.T @ y_train

        self.bias = theta[0]
        self.weights = theta[1:]

        abs_weights = np.abs(self.weights)
        pengaruh = abs_weights / (np.sum(abs_weights) + 1e-8)

        self.tabel_pengaruh_features = pd.DataFrame({
            "weight": self.weights, 
            "importance": pengaruh
            })
        return self
    
    def model_predict(self, X):
        """
        model prediksi
        """
        return X @ self.weights + self.bias
    
    def evaluate(self, X, y):
        """
        evaluasi model prediksi
        """
        y_pred = self.model_predict(X)
        residuals = y - y_pred

        mse = np.mean(residuals**2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(residuals))

        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0  

        return {
            "MSE score": float(mse),
            "RMSE score": float(rmse),
            "MAE score": float(mae),
            "R2 score": float(r2)
        }
    
    def ambil_peringkat_fitur_berpengaruh(self, nama_feature):
        self.tabel_pengaruh_features['feature'] = nama_feature
        tabel_terurut = self.tabel_pengaruh_features.sort_values("importance", ascending=False)
        return tabel_terurut

        
class EDCDFBiasCorrector:
    """
    Koreksi bias iklim metode EDCDF. 
    arxiv.org/html/2504.19145
    """

    def __init__(self, n_quantiles: int = 100):
        self.n_quantiles = n_quantiles
        self.quantile_levels = np.linspace(0, 1, n_quantiles + 1)
        self._obs_quantiles = {}   
        self._mod_quantiles = {}   
        self._fitted = False

    def fit(self, obs_hist: np.ndarray, mod_hist: np.ndarray):
        """Latih corrector dengan data historis (observasi vs model)."""
        obs_clean = obs_hist[~np.isnan(obs_hist)]
        mod_clean = mod_hist[~np.isnan(mod_hist)]

        self._obs_quantiles = np.quantile(obs_clean, self.quantile_levels)
        self._mod_quantiles = np.quantile(mod_clean, self.quantile_levels)
        self._fitted = True
        return self

    def transform(self, mod_proj: np.ndarray) -> np.ndarray:
        """Terapkan koreksi ke data proyeksi."""
        if not self._fitted:
            raise RuntimeError("Panggil fit() terlebih dahulu.")

        corrected = np.empty_like(mod_proj)
        for i, x in enumerate(mod_proj):
            if np.isnan(x):
                corrected[i] = np.nan
                continue

            p = np.searchsorted(self._mod_quantiles, x) / self.n_quantiles
            p = np.clip(p, 0, 1)

            obs_equiv = np.interp(p, self.quantile_levels, self._obs_quantiles)
            mod_equiv = np.interp(p, self.quantile_levels, self._mod_quantiles)

            delta = x - mod_equiv
            corrected[i] = obs_equiv + delta

        return corrected

    def fit_transform(self, obs_hist, mod_hist, mod_proj):
        return self.fit(obs_hist, mod_hist).transform(mod_proj)
    
class ZScore:
    def __init__(self):
        self.mean = None 
        self.std = None 
    
    def fit(self, X):
        """ 
        Cari rata-rata ma standar deviasi
        abaikan data Nan
        cegah error pembagian 0
        """ 
        self.mean = np.nanmean(X, axis=0)
        self.std = np.nanstd(X, axis=0)
        self.std[self.std == 0] = 1.0
        return self
    
    def transform(self, X):
        """ 
        z = (x - mean) / standar deviasi
        """ 
        return (X - self.mean) / self.std

```

### 1. Eksporasi Data

Disini kita nge-check data hasil olah yang kita dapatkan sebelum kita training, bagaimana persebaran datanya


```python
DATA_PATH   = '../data/processed/features/X_historical.csv'
TARGET_PATH = '../data/processed/targets/y_historical.csv'
RAW_PATH    = '../tmp_folder/historical_raw.csv'

df_raw = pd.read_csv(RAW_PATH, parse_dates=["time"])
df_X   = pd.read_csv(DATA_PATH, parse_dates=['time'])
df_y   = pd.read_csv(TARGET_PATH, parse_dates=['time'])

print(f'Data mentah : {df_raw.shape}')
print(f'Data fitur  : X={df_X.shape}')
print(f'Data target : y={df_y.shape}')
```

    Data mentah : (228, 16)
    Data fitur  : X=(216, 108)
    Data target : y=(216, 2)



```python
df_X.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>cloud_cover</th>
      <th>evaporation</th>
      <th>humidity</th>
      <th>specific_humidity</th>
      <th>precip</th>
      <th>pressure</th>
      <th>sea_level_pressure</th>
      <th>thermal_rad</th>
      <th>solar_rad</th>
      <th>...</th>
      <th>humidity_roll12</th>
      <th>pressure_roll3</th>
      <th>pressure_roll6</th>
      <th>pressure_roll12</th>
      <th>solar_rad_roll3</th>
      <th>solar_rad_roll6</th>
      <th>solar_rad_roll12</th>
      <th>month_sin</th>
      <th>month_cos</th>
      <th>temp_2m_anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>216</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>...</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>216.000000</td>
      <td>2.160000e+02</td>
      <td>2.160000e+02</td>
      <td>216.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1996-12-15 17:33:20</td>
      <td>69.642516</td>
      <td>4.908247</td>
      <td>81.336604</td>
      <td>17.675083</td>
      <td>6.704157</td>
      <td>1005.893454</td>
      <td>1010.025682</td>
      <td>411.992879</td>
      <td>220.585300</td>
      <td>...</td>
      <td>81.318390</td>
      <td>1005.900324</td>
      <td>1005.907275</td>
      <td>1005.916799</td>
      <td>220.529453</td>
      <td>220.763519</td>
      <td>220.916118</td>
      <td>-2.672759e-17</td>
      <td>-4.523131e-17</td>
      <td>0.004485</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1988-01-01 00:00:00</td>
      <td>52.963657</td>
      <td>2.727502</td>
      <td>75.502045</td>
      <td>16.010563</td>
      <td>0.653993</td>
      <td>1002.383540</td>
      <td>1006.674560</td>
      <td>387.194980</td>
      <td>134.194500</td>
      <td>...</td>
      <td>79.548189</td>
      <td>1003.289833</td>
      <td>1004.019667</td>
      <td>1005.101064</td>
      <td>143.139397</td>
      <td>162.341298</td>
      <td>186.353990</td>
      <td>-1.000000e+00</td>
      <td>-1.000000e+00</td>
      <td>-0.480507</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1992-06-23 12:00:00</td>
      <td>66.818310</td>
      <td>3.871746</td>
      <td>80.032270</td>
      <td>17.154692</td>
      <td>3.884860</td>
      <td>1005.117935</td>
      <td>1009.262905</td>
      <td>405.956558</td>
      <td>188.681255</td>
      <td>...</td>
      <td>80.896405</td>
      <td>1005.166145</td>
      <td>1005.365672</td>
      <td>1005.641834</td>
      <td>193.436947</td>
      <td>202.827869</td>
      <td>215.670383</td>
      <td>-5.915064e-01</td>
      <td>-5.915064e-01</td>
      <td>-0.153780</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1996-12-16 12:00:00</td>
      <td>70.224625</td>
      <td>4.789350</td>
      <td>81.488835</td>
      <td>17.723810</td>
      <td>6.518428</td>
      <td>1005.942525</td>
      <td>1010.115485</td>
      <td>412.625335</td>
      <td>223.802760</td>
      <td>...</td>
      <td>81.264621</td>
      <td>1005.928322</td>
      <td>1005.932230</td>
      <td>1005.962583</td>
      <td>224.321790</td>
      <td>221.390445</td>
      <td>222.826697</td>
      <td>-6.123234e-17</td>
      <td>-6.123234e-17</td>
      <td>0.021640</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2001-06-08 12:00:00</td>
      <td>72.886800</td>
      <td>5.970834</td>
      <td>83.034966</td>
      <td>18.216494</td>
      <td>9.134887</td>
      <td>1006.727710</td>
      <td>1010.860000</td>
      <td>418.469050</td>
      <td>253.330180</td>
      <td>...</td>
      <td>81.789941</td>
      <td>1006.680674</td>
      <td>1006.462329</td>
      <td>1006.127683</td>
      <td>246.697333</td>
      <td>240.080524</td>
      <td>227.814116</td>
      <td>5.915064e-01</td>
      <td>5.915064e-01</td>
      <td>0.152069</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2005-12-01 00:00:00</td>
      <td>78.335570</td>
      <td>7.619037</td>
      <td>85.912020</td>
      <td>18.891539</td>
      <td>15.000661</td>
      <td>1008.973100</td>
      <td>1012.823800</td>
      <td>431.001160</td>
      <td>308.766240</td>
      <td>...</td>
      <td>82.976364</td>
      <td>1008.152480</td>
      <td>1007.494410</td>
      <td>1006.757668</td>
      <td>299.030623</td>
      <td>276.384897</td>
      <td>238.501015</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.599548</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>4.598907</td>
      <td>1.180416</td>
      <td>2.186751</td>
      <td>0.654873</td>
      <td>3.670282</td>
      <td>1.266420</td>
      <td>1.210277</td>
      <td>9.004186</td>
      <td>39.974975</td>
      <td>...</td>
      <td>0.627878</td>
      <td>1.044531</td>
      <td>0.743454</td>
      <td>0.358970</td>
      <td>33.347860</td>
      <td>23.898914</td>
      <td>10.089051</td>
      <td>7.087493e-01</td>
      <td>7.087493e-01</td>
      <td>0.222742</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 108 columns</p>
</div>




```python
df_y.describe().round()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>temp_2m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>216</td>
      <td>216.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1996-12-15 17:33:20</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1988-01-01 00:00:00</td>
      <td>25.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1992-06-23 12:00:00</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1996-12-16 12:00:00</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2001-06-08 12:00:00</td>
      <td>26.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2005-12-01 00:00:00</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_raw.describe().round()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>time</th>
      <th>cloud_cover</th>
      <th>evaporation</th>
      <th>humidity</th>
      <th>specific_humidity</th>
      <th>precip</th>
      <th>pressure</th>
      <th>sea_level_pressure</th>
      <th>thermal_rad</th>
      <th>solar_rad</th>
      <th>wind_speed</th>
      <th>temp_2m</th>
      <th>temp_max_24h</th>
      <th>temp_min_24h</th>
      <th>wind_u</th>
      <th>wind_v</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>228</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
      <td>228.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1996-05-27 21:03:09.473684</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>81.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>1006.0</td>
      <td>1010.0</td>
      <td>412.0</td>
      <td>220.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>-3.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1986-01-01 00:00:00</td>
      <td>53.0</td>
      <td>3.0</td>
      <td>76.0</td>
      <td>16.0</td>
      <td>1.0</td>
      <td>1002.0</td>
      <td>1007.0</td>
      <td>387.0</td>
      <td>134.0</td>
      <td>3.0</td>
      <td>25.0</td>
      <td>26.0</td>
      <td>25.0</td>
      <td>-8.0</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1991-09-23 12:00:00</td>
      <td>67.0</td>
      <td>4.0</td>
      <td>80.0</td>
      <td>17.0</td>
      <td>4.0</td>
      <td>1005.0</td>
      <td>1009.0</td>
      <td>406.0</td>
      <td>189.0</td>
      <td>5.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>-6.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1996-06-16 00:00:00</td>
      <td>70.0</td>
      <td>5.0</td>
      <td>82.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>1006.0</td>
      <td>1010.0</td>
      <td>413.0</td>
      <td>225.0</td>
      <td>6.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>-5.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2001-03-08 18:00:00</td>
      <td>73.0</td>
      <td>6.0</td>
      <td>83.0</td>
      <td>18.0</td>
      <td>9.0</td>
      <td>1007.0</td>
      <td>1011.0</td>
      <td>418.0</td>
      <td>253.0</td>
      <td>8.0</td>
      <td>26.0</td>
      <td>27.0</td>
      <td>26.0</td>
      <td>0.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2005-12-01 00:00:00</td>
      <td>78.0</td>
      <td>8.0</td>
      <td>86.0</td>
      <td>19.0</td>
      <td>16.0</td>
      <td>1009.0</td>
      <td>1013.0</td>
      <td>431.0</td>
      <td>309.0</td>
      <td>9.0</td>
      <td>27.0</td>
      <td>28.0</td>
      <td>26.0</td>
      <td>5.0</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>40.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Melihat persebaran data yang didapat dari CORDEX
variables = [
    ('cloud_cover', 'Tutupan Awan',        '#d7191b'),
    ('temp_2m',   'Suhu 2m (°C)',         '#d7191c'),
    ('precip',    'Curah Hujan (mm/hari)', '#2c7bb6'),
    ('pressure',  'Tekanan (hPa)',         '#756bb1'),
    ('humidity',  'Kelembaban (%)',        '#31a354'),
    ('wind_u',    'Angin-U (m/s)',         '#fd8d3c'),
    ('wind_v',    'Angin-V (m/s)',         '#636363'),
    ('solar_rad', 'Radiasi Surya (W/m²)', '#e6550d'),
]

# buat layout
n_vars = len(variables)
rows, cols = 4, 2

plt.figure(figsize=(21, 13))
plt.suptitle('Tren Variabel Iklim CORDEX - Pulau Jawa (1976-2005)', fontsize=24, fontweight='bold', y=0.98)

for i, (col, label, color) in enumerate(variables):
    if col in df_raw.columns:
        plt.subplot(rows, cols, i + 1)
        
        # Plot data mentah
        plt.plot(df_raw['time'], df_raw[col], color=color, linewidth=0.6, alpha=0.5, label='Data Mentah')
        
        # Plot Moving Average
        roll = df_raw[col].rolling(12, center=True).mean()
        plt.plot(df_raw['time'], roll, color=color, linewidth=2.0, label='MA (12 Bulan)')
        
        plt.title(f'Tren {label}', fontsize=16, fontweight='bold')
        plt.ylabel(label, fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        if i == 0:
            plt.legend(fontsize=10)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.savefig('../results/figures/Gabungan_Tren_Persebaran_Features_CORDEX.pdf', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_8_0.png)
    



```python
# Matriks Korelasi
num_cols = [c for c in df_raw.columns if c != 'time']
X_all = df_raw[num_cols].values

n_vars = X_all.shape[1]
corr_matrix = np.zeros((n_vars, n_vars))
for i in range(n_vars):
    for j in range(n_vars):
        xi = X_all[:, i] - X_all[:, i].mean()
        xj = X_all[:, j] - X_all[:, j].mean()
        denom = np.sqrt(np.sum(xi**2) * np.sum(xj**2))
        corr_matrix[i,j] = np.sum(xi * xj) / denom if denom > 0 else 0

plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

plt.title('Matriks Korelasi Pearson\n Data CORDEX-SEA Pulau Jawa', fontweight='bold', fontsize=18)

plt.xticks(range(n_vars), num_cols, rotation=45, ha='right', fontsize=16)
plt.yticks(range(n_vars), num_cols, fontsize=16)

#kasih angka di dalam
for i in range(n_vars):
    for j in range(n_vars):
        plt.text(j, i, f'{corr_matrix[i,j]:.2f}',
                 ha='center', va='center', fontsize=8,
                 color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

plt.colorbar(shrink=0.8)

plt.tight_layout()
plt.savefig('../results/figures/Matriks_Korelasi_Pearson_Variabel.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../results/figures/Matriks_Korelasi_Pearson_Variabel.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_9_0.png)
    


### 2. Bias Correction (EDCDF)
Ini manggil class BIASECDFcorrector untuk memastikan bahwa data yang kita dapatkan itu merepresentasikan data asli dan ga halu pada kesalahan mean/quartile data yang menyimpang (mencegah systemic error)


```python
# Menambahkan noise 
data_observasi = df_raw['temp_2m'].values + np.random.normal(0, 0.1, len(df_raw))
data_observasi_cacat = df_raw['temp_2m'].values + 1.2

# Simulasi
n_proj = 200
t_proj = np.arange(n_proj)
seasonal_proj = np.sin(2* np.pi * t_proj / 12)
trend_proj = 0.005 * t_proj
proyeksi_data_cacat = 28.0 + 1.5 * seasonal_proj + trend_proj + 1.2 + np.random.normal(0, 0.3, n_proj) 

corrector = EDCDFBiasCorrector(n_quantiles=100)
corrected = corrector.fit_transform(data_observasi, data_observasi_cacat, proyeksi_data_cacat)

print(f'Sebelum koreksi: mean={proyeksi_data_cacat.mean():.3f}°C')
print(f'Sesudah koreksi: mean={corrected.mean():.3f}°C')
print(f'Referensi hist:  mean={data_observasi.mean():.3f}°C')
```

    Sebelum koreksi: mean=29.728°C
    Sesudah koreksi: mean=28.607°C
    Referensi hist:  mean=26.203°C


### 3 Persiapan Data Training

Disini kita bakal ngebagi data menjadi 3 (training, validation, dan test) kemudian data training di normalin biar punya skala yang sama (rata-rata 0 dan standar deviasinya 1)


```python
year = df_X['time'].dt.year

TRAIN_END = 2000
VAL_END   = 2003

mask_train = year <= TRAIN_END
mask_val   = (year > TRAIN_END) & (year <= VAL_END)
mask_test  = year > VAL_END

LEAK_COLS = [
    'temp_max_24h', 'temp_min_24h', 
    'temp_2m_roll3', 'temp_2m_roll6', 'temp_2m_roll12', 
    'temp_2m_anomaly'
]
feature_cols = [c for c in df_X.columns if c != 'time' and c not in LEAK_COLS]

X_train = df_X[mask_train][feature_cols].values
y_train = df_y[mask_train]['temp_2m'].values
X_val   = df_X[mask_val][feature_cols].values
y_val   = df_y[mask_val]['temp_2m'].values
X_test  = df_X[mask_test][feature_cols].values
y_test  = df_y[mask_test]['temp_2m'].values
t_test  = df_X[mask_test]['time'].values

print(f'Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}')
print(f'Periode train: {df_X[mask_train]["time"].min()} - {df_X[mask_train]["time"].max()}')
print(f'Periode val  : {df_X[mask_val]["time"].min()} - {df_X[mask_val]["time"].max()}')
print(f'Periode test : {df_X[mask_test]["time"].min()} - {df_X[mask_test]["time"].max()}')
```

    Train: (156, 101) | Val: (36, 101) | Test: (24, 101)
    Periode train: 1988-01-01 00:00:00 - 2000-12-01 00:00:00
    Periode val  : 2001-01-01 00:00:00 - 2003-12-01 00:00:00
    Periode test : 2004-01-01 00:00:00 - 2005-12-01 00:00:00



```python
# Normalisasi Z-score
norm_X = ZScore()
X_train_n = norm_X.fit(X_train).transform(X_train)
X_val_n   = norm_X.transform(X_val)
X_test_n  = norm_X.transform(X_test)

y_mean, y_std = y_train.mean(), y_train.std()
y_train_n = (y_train - y_mean) / y_std
y_val_n   = (y_val   - y_mean) / y_std
y_test_n  = (y_test  - y_mean) / y_std

print(f'X train normalized: mean={X_train_n.mean():.4f} | std={X_train_n.std():.4f}')
print(f'y train: mean={y_mean:.3f}°C | std={y_std:.3f}°C')
```

    X train normalized: mean=0.0000 | std=1.0000
    y train: mean=26.176°C | std=0.291°C


Setelah melakukan Normalisasi kita mendapatkan persamaan analitiknya berupa 

$$T_{pred} = 26.176 + 0.291 \times \left( \sum_{i=1}^{101} w_i \cdot \frac{X_i - \mu_i}{\sigma_i} \right)$$

$w_i$ adalah nilai weight dari masing-masing fitur yang didapatkan setelah melakukan training

### 4. Training Model : Regresi Multivariat
disini kita melakukan training menggunakan model algoritma Machine Learning Ridge Regression untuk menemukan weights dan bias yang terbaik


```python
model = MultivariateRegressionModel()

print(f'Fitur: {X_train_n.shape[1]}')
print(f'Samples: {len(y_train_n)}')
```

    Fitur: 101
    Samples: 156



```python
# Lakukan Training
model.fit(X_train_n, y_train_n)

#ini hasil yang kita dapatkan, kita mendapatkan 101 weights untuk persamaan analitik kita
# karena biasnya mencapai 0 maka kita bisa mengabaikan nilai biasnya
print(f'Weights shape: {model.weights.shape}')
print(f'Bias: {model.bias:.4f}')
```

    Weights shape: (101,)
    Bias: -0.0000



```python
#weights yang didapatkan setelah melakukan training (data training) untuk setiap fitur (101 fitur)
# weights ini akan masuk ke dalam persamaan analitik sebelumnya untuk mendapatkan estimasi suhu iklim kedepannya
print(model.weights)
```

    [ 2.22978478e-02 -1.33308457e-03 -1.34324889e+00  2.00751578e+00
      4.09501897e-02  1.90865746e-01 -7.92957011e-02  5.07077068e-02
      6.71610528e-02  4.13430870e-02 -1.12812664e-01 -5.67435029e-02
     -1.36578664e-02 -3.13803657e-02  9.98915206e-04  7.37630106e-03
      7.13773785e-03  6.54156062e-03  1.82676323e-02  3.83908172e-02
     -1.33189828e-02  3.18214922e-02  3.04035693e-01 -1.11751020e-01
     -2.66326167e-01 -1.01551780e-01  2.63519433e-01 -1.06412900e-01
      3.28094404e-01  3.42977422e-01  1.87318750e-01 -3.28339655e-01
     -3.54740752e-02  3.03464023e-02 -5.21755826e-03  3.43322978e-02
     -4.64147295e-02 -2.58196167e-01 -1.13836115e-01 -3.00695227e-02
     -8.36607999e-02 -9.70133975e-02  2.42302911e-01  1.56126663e-01
      3.66798694e-02  1.18447450e-01  9.29672017e-02 -4.13345306e-02
      1.71853886e-01  1.20876255e-02 -4.43528244e-02 -7.03527245e-02
     -5.00360962e-02  1.21004430e-01  3.64585586e-02  3.59830669e-02
     -1.31788656e-01  3.13467763e-02 -3.01315883e-02 -1.05748731e-02
      1.00534103e-02 -6.74449271e-02 -3.25503954e-02 -1.16307314e-01
      7.57266643e-02 -1.73638854e-01  6.17285786e-02  6.48614776e-02
      3.00621859e-03 -1.41804673e-01  3.73340349e-02  1.79970776e-02
     -1.66060155e-02 -1.06018876e-01 -1.11992986e-01  6.81519381e-02
      1.25365152e-01  2.59145729e-02  1.63131519e-02 -4.32607886e-02
      2.72236225e-03 -1.80374553e-02 -4.03000164e-02  5.59108581e-02
     -9.30015921e-02  3.22354423e-03  6.81263160e-03  1.46950425e-02
      6.03607045e-02 -4.46941419e-02 -4.91864332e-01  3.09387105e-02
     -2.03813213e-02 -7.53228249e-02  5.02598409e-02 -2.95023734e-02
      5.49850073e-02  4.88099730e-02 -5.20376943e-02 -6.21378704e-02
      1.11923503e-01]


### 5. Evaluasi Model

Disini kita mengevaluasi model yang sudah di train dengan MSE, RMSE, MAE, R2 Score dan melihat langsung Learning Curve dan prediksi yang dilakukan oleh Model Ridge Regression yang sudah kita latih


```python
matriks_train = model.evaluate(X_train_n, y_train_n)
matriks_val = model.evaluate(X_val_n, y_val_n)
matriks_test = model.evaluate(X_test_n, y_test_n)

print(f'Train MSE: {matriks_train["MSE score"]:.6f} | RMSE: {matriks_train["RMSE score"]:.6f} | MAE score: {matriks_train["MAE score"]:.6f} | R2 Score: {matriks_train["R2 score"]:.4f}')
print(f'Val MSE: {matriks_val["MSE score"]:.6f} | RMSE: {matriks_val["RMSE score"]:.6f} | MAE score: {matriks_val["MAE score"]:.6f} | R2 Score: {matriks_val["R2 score"]:.4f}')
print(f'Test MSE: {matriks_test["MSE score"]:.6f} | RMSE: {matriks_test["RMSE score"]:.6f} | MAE score: {matriks_test["MAE score"]:.6f} | R2 Score: {matriks_test["R2 score"]:.4f}')
```

    Train MSE: 0.000516 | RMSE: 0.022716 | MAE score: 0.018541 | R2 Score: 0.9995
    Val MSE: 0.006846 | RMSE: 0.082740 | MAE score: 0.065549 | R2 Score: 0.9950
    Test MSE: 0.010158 | RMSE: 0.100785 | MAE score: 0.077699 | R2 Score: 0.9915


dari hasilnya terlihat bahwa R2 Score terhadap data test didapatkan memang tinggi 99% tetapi perbandingan MSE terhadap data train (0.0005) dan data test (0.01) sekitar 20 kali yang berarti menunjukkan adanya indikasi overfitting pada model


```python
# Learning Curve plots
train_sizes = np.linspace(0.1, 1.0, 10)
train_rmses = []
val_rmses = []

for frac in train_sizes:
    n_samples = int(frac * len(X_train_n))
    if n_samples < 5 : 
        continue

    X_subset = X_train_n[:n_samples]
    y_subset = y_train_n[:n_samples]

    temp_model = MultivariateRegressionModel()
    temp_model.fit(X_subset, y_subset)

    pred_t = temp_model.model_predict(X_subset)
    res_t = y_subset - pred_t
    train_rmses.append(np.sqrt(np.mean(res_t**2)))

    pred_v = temp_model.model_predict(X_val_n)
    res_v = y_val_n - pred_v
    val_rmses.append(np.sqrt(np.mean(res_v**2)))

sample_counts = [int(frac * len(X_train_n)) for frac in train_sizes if int(frac * len(X_train_n)) >= 5]

plt.figure(figsize=(13, 5))

plt.plot(sample_counts, train_rmses, 'o-', linewidth=2, color='#2c7bb6', label='Training Error')
plt.plot(sample_counts, val_rmses, 's-', linewidth=2, color='#d7191c', label='Validation Error')

plt.title('Learning Curve: Menggunakan Data Training', fontweight='bold', fontsize=18)
plt.xlabel('Jumlah Sampel Data Latih (Bulan)', fontsize = 16)
plt.ylabel('RMSES (Normalized)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.savefig('../results/figures/Learning_Curve.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../results/figures/Learning_Curve.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_23_0.png)
    



```python
#Melakukan prediksi sampai 2045 (49 tahun) kedepan 
df_rcp85 = pd.read_csv('../data/processed/features/X_rcp85.csv')
dates_rcp85 = pd.to_datetime(df_rcp85['time'])

# 1. Feature Selection menggunakan list fitur yang identik dengan saat di-train
X_rcp85_raw = df_rcp85[feature_cols].values

# 2. Standarisasi/Z-Score Normalisasi menggunakan ZScoreNormalizer yang terlatih oleh sejarah
X_rcp85_n = norm_X.transform(X_rcp85_raw)

# 3. Inferensi MultiVariat ML Model
y_rcp85_pred_n = model.model_predict(X_rcp85_n)

# 4. Detransformasi ke representasi Suhu Derajat Celcius Nyata
y_rcp85_pred = (y_rcp85_pred_n * y_std) + y_mean
```


```python

plt.figure(figsize=(13, 5))

plt.plot(df_raw['time'], df_raw['temp_2m'], color='#2c7bb6', linewidth=0.8, alpha=0.4, label='Historis (CORDEX)')
roll_hist = df_raw['temp_2m'].rolling(12, center=True).mean()
plt.plot(df_raw['time'], roll_hist, color='#2c7bb6', linewidth=2.5, label='Rolling 12-bln (Hist)')

plt.plot(dates_rcp85, y_rcp85_pred, color='#d7191c', linewidth=0.8, alpha=0.3)

roll85 = pd.Series(y_rcp85_pred).rolling(24, center=True).mean()
plt.plot(dates_rcp85, roll85, color='#d7191c', linewidth=2.5, label='ML Prediksi RCP8.5')

plt.title('Prediksi Suhu Permukaan', fontweight='bold', fontsize=18)
plt.xlabel('Tahun', fontsize=16)
plt.ylabel('Suhu Permukaan (°C)', fontsize=16)
plt.legend(fontsize=10, ncol=3)
plt.grid(axis='y', alpha=0.3)

plt.savefig('../results/figures/Prediksi_Suhu_Permukaan_Jawa_2045.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../results/figures/Prediksi_Suhu_Permukaan_Jawa_2045.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_25_0.png)
    



```python
plt.figure(figsize=(13, 5))

baseline = df_raw['temp_2m'].mean()
kenaikan_vals = roll85.values - baseline

plt.fill_between(dates_rcp85, 0, kenaikan_vals, alpha=0.5, color='#d7191c', label='Anomali RCP8.5')

plt.title(f'Prediksi Kenaikan suhu dengan Baseline {baseline:.2f}°C (2008-2005)', fontweight='bold', fontsize=18)
plt.ylabel('Kenaikan Suhu (°C)', fontsize=16)
plt.xlabel('Tahun', fontsize=16)

plt.legend(fontsize=10)
plt.grid(axis='y', alpha=0.3)

plt.savefig('../results/figures/Prediksi_Kenaikan_Suhu_Jawa_2045.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../results/figures/Prediksi_Kenaikan_Suhu_Jawa_2045.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_26_0.png)
    




### 6 [Appendix] Pengaruh Feature

Ini kami cuma ingin tahu variabel apa yang paling mempengaruhi kenaikan iklim pulau jawa dan menjadi kesimpulan akhir dari model kita


```python
fi = model.ambil_peringkat_fitur_berpengaruh(feature_cols)
top15 = fi.head(15)

fig, ax = plt.subplots(figsize=(10, 6))

#Merah jika weight negatif, biru jika weight positif
colors = ['#d7191c' if w < 0 else '#2c7bb6' for w in top15['weight']]
bars = ax.barh(top15['feature'], top15['importance'], color=colors, alpha=0.85)

ax.set_title('Variabel Paling Berpengaruh Terhadap Perubahan Iklim\nRegresi Multivariat CORDEX-SEA', fontweight='bold', fontsize=18)
ax.set_xlabel('Pengaruh', fontsize=16)
ax.tick_params(axis='y', labelsize=16)
ax.invert_yaxis()

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.002, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center', fontsize=6)

fig.tight_layout()
plt.savefig('../results/figures/Variabel_paling_berpengaruh.pdf', dpi=150, bbox_inches='tight')
plt.savefig('../results/figures/Variabel_paling_berpengaruh.png', dpi=150, bbox_inches='tight')
plt.show()
```


    
![png](workspace_baru_files/workspace_baru_29_0.png)
    


### Referensi Utama
- Tangang et al. (2020) — CORDEX-SEA: *An integrated regional climate model projection for Southeast Asia* — Climate Dynamics
- Li et al. (2010) — EDCDF Bias Correction — J. Hydrology
- Ghaemi et al. (2023) — Regional Temperature Projection arXiv:2504.19145


