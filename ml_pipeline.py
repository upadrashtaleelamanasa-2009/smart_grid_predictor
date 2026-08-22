"""
ml_pipeline.py – Smart Grid Energy Predictor
Train & serve predictions using smart_grid_dataset.csv
"""
import os, json, warnings
import pandas as pd
import numpy as np
import joblib
warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "smart_grid_dataset.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURE_COLS = [
    "Hour", "Minute", "DayOfWeek", "Month", "IsWeekend",
    "Quarter", "DayOfYear",
    "Hour_sin", "Hour_cos", "DOW_sin", "DOW_cos",
    "Temperature_C", "Humidity_pct",
    "Lag_1", "Lag_2", "Lag_4", "Lag_8", "Lag_16", "Lag_96", "Lag_672",
    "Roll_mean_4", "Roll_mean_16", "Roll_mean_96",
    "Roll_std_4", "Roll_max_16", "Roll_min_16",
    "Solar_kW_prev", "Wind_kW_prev"
]


# ─── Load & Engineer ────────────────────────────────────────────────────
def load_and_engineer():
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]

    # Fix temperature col (encoding issue)
    temp_col = [c for c in df.columns if "Temp" in c][0]
    df.rename(columns={
        temp_col: "Temperature_C",
        "Humidity (%)": "Humidity_pct",
        "Power Consumption (kW)": "Power_kW",
        "Solar Power (kW)": "Solar_kW",
        "Wind Power (kW)": "Wind_kW",
        "Grid Supply (kW)": "Grid_kW",
        "Predicted Load (kW)": "PredictedLoad_kW",
        "Voltage (V)": "Voltage_V",
        "Current (A)": "Current_A",
        "Power Factor": "PowerFactor",
        "Reactive Power (kVAR)": "ReactivePower_kVAR",
        "Voltage Fluctuation (%)": "VoltageFluctuation_pct",
        "Overload Condition": "Overload",
        "Transformer Fault": "TransformerFault",
        "Electricity Price (USD/kWh)": "Price_USD_kWh",
    }, inplace=True)

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    df["Hour"]      = df["Timestamp"].dt.hour
    df["Minute"]    = df["Timestamp"].dt.minute
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    df["Month"]     = df["Timestamp"].dt.month
    df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)
    df["Quarter"]   = df["Timestamp"].dt.quarter
    df["DayOfYear"] = df["Timestamp"].dt.dayofyear
    # Cyclical time encoding
    df["Hour_sin"]  = np.sin(2 * np.pi * df["Hour"] / 24)
    df["Hour_cos"]  = np.cos(2 * np.pi * df["Hour"] / 24)
    df["DOW_sin"]   = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DOW_cos"]   = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Lag features (15-min intervals → 1 step=15min, 4=1h, 16=4h, 96=24h)
    df["Lag_1"]   = df["Power_kW"].shift(1)   # 15 min ago
    df["Lag_2"]   = df["Power_kW"].shift(2)   # 30 min ago
    df["Lag_4"]   = df["Power_kW"].shift(4)   # 1 hour ago
    df["Lag_8"]   = df["Power_kW"].shift(8)   # 2 hours ago
    df["Lag_16"]  = df["Power_kW"].shift(16)  # 4 hours ago
    df["Lag_96"]  = df["Power_kW"].shift(96)  # 24 hours ago
    df["Lag_672"] = df["Power_kW"].shift(672) # 1 week ago
    df["Solar_kW_prev"] = df["Solar_kW"].shift(1)
    df["Wind_kW_prev"]  = df["Wind_kW"].shift(1)

    # Rolling statistics
    df["Roll_mean_4"]  = df["Power_kW"].rolling(4,  min_periods=1).mean()
    df["Roll_mean_16"] = df["Power_kW"].rolling(16, min_periods=1).mean()
    df["Roll_mean_96"] = df["Power_kW"].rolling(96, min_periods=1).mean()
    df["Roll_std_4"]   = df["Power_kW"].rolling(4,  min_periods=1).std().fillna(0)
    df["Roll_max_16"]  = df["Power_kW"].rolling(16, min_periods=1).max()
    df["Roll_min_16"]  = df["Power_kW"].rolling(16, min_periods=1).min()

    df.dropna(inplace=True)
    return df


# ─── Train ───────────────────────────────────────────────────────────────
def train():
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    from sklearn.model_selection import train_test_split

    print("[ML] Loading and engineering features...")
    df = load_and_engineer()

    X = df[FEATURE_COLS]
    y = df["PredictedLoad_kW"]   # Use dataset's Predicted Load as target (high correlation with actual)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models_to_train = {
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
        "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1),
        "Linear Regression": LinearRegression(),
    }

    results = {}
    best_name, best_score, best_model = None, -np.inf, None

    for name, m in models_to_train.items():
        print(f"[ML] Training {name}...")
        if name == "Linear Regression":
            m.fit(X_train_s, y_train)
            preds = m.predict(X_test_s)
        else:
            m.fit(X_train, y_train)
            preds = m.predict(X_test)

        r2   = round(r2_score(y_test, preds), 4)
        mae  = round(mean_absolute_error(y_test, preds), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, preds)), 4)
        mape = round(np.mean(np.abs((y_test - preds) / (y_test + 1e-9))) * 100, 2)
        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}
        print(f"    R²={r2}  MAE={mae}  RMSE={rmse}")

        if r2 > best_score:
            best_score, best_name, best_model = r2, name, m

    print(f"[ML] Best model: {best_name} (R²={best_score})")

    # Save artifacts
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.joblib"))
    joblib.dump(scaler,     os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "model_metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "best_r2": best_score, "models": results}, f, indent=2)
    with open(os.path.join(MODEL_DIR, "best_model_name.txt"), "w") as f:
        f.write(best_name)

    # Save analytics for dashboard
    df_a = df.copy()
    # Hourly averages
    hourly = df_a.groupby("Hour")["Power_kW"].mean().reset_index()
    hourly.columns = ["Hour", "mean"]

    # Actual vs Predicted (sample 500 points from test set for charts)
    if best_name == "Linear Regression":
        X_t = scaler.transform(X_test)
        test_preds = best_model.predict(X_t)
    else:
        test_preds = best_model.predict(X_test)

    sample_idx = np.linspace(0, len(X_test)-1, 500, dtype=int)
    ts_sample  = df.iloc[-len(X_test):].iloc[sample_idx]["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist()
    actual_s   = y_test.values[sample_idx].round(3).tolist()
    pred_s     = test_preds[sample_idx].round(3).tolist()

    analytics = {
        "hourly_avg":  hourly.to_dict("records"),
        "date_range":  {"start": str(df["Timestamp"].min()), "end": str(df["Timestamp"].max())},
        "stats": {
            "total_records": len(df),
            "avg_power_kw":  round(df["Power_kW"].mean(), 3),
            "max_power_kw":  round(df["Power_kW"].max(), 3),
            "min_power_kw":  round(df["Power_kW"].min(), 3),
            "overload_pct":  round(df["Overload"].mean() * 100, 1),
            "fault_pct":     round(df["TransformerFault"].mean() * 100, 1),
            "avg_solar_kw":  round(df["Solar_kW"].mean(), 3),
            "avg_wind_kw":   round(df["Wind_kW"].mean(), 3),
        },
        "actual_vs_pred": {
            "timestamps": ts_sample,
            "actual":     actual_s,
            "predicted":  pred_s,
        }
    }
    with open(os.path.join(MODEL_DIR, "analytics.json"), "w") as f:
        json.dump(analytics, f, indent=2)

    print("[ML] All artifacts saved.")
    return best_name, best_score, results

# ─── Serve: Predict a range ──────────────────────────────────────────────
def load_model():
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    with open(os.path.join(MODEL_DIR, "best_model_name.txt")) as f:
        name = f.read().strip()
    with open(os.path.join(MODEL_DIR, "analytics.json")) as f:
        analytics = json.load(f)
    return model, scaler, name, analytics

def predict_range(from_date, to_date, from_hour, to_hour,
                  temperature, humidity):
    """
    Predict power consumption for each 15-min slot in the range.
    Returns: list of dicts with timestamp, predicted_kw, demand_level
    """
    model, scaler, model_name, analytics = load_model()
    stats      = analytics["stats"]

    # Load initial history buffer from dataset (need at least 672 steps for 1 week lag_672)
    # We will fallback to baseline averages if file doesn't load
    try:
        df = pd.read_csv(DATA_PATH)
        df.columns = [c.strip() for c in df.columns]
        power_hist = df["Power Consumption (kW)"].tail(1000).tolist()
        solar_hist = df["Solar Power (kW)"].tail(1000).tolist()
        wind_hist  = df["Wind Power (kW)"].tail(1000).tolist()
    except Exception:
        baseline = stats.get("avg_power_kw", 5.2)
        power_hist = [baseline] * 1000
        solar_hist = [stats.get("avg_solar_kw", 1.0)] * 1000
        wind_hist  = [stats.get("avg_wind_kw", 2.0)] * 1000

    from datetime import datetime, timedelta
    slots = []
    cur = datetime.strptime(f"{from_date} {from_hour:02d}:00", "%Y-%m-%d %H:%M")
    end = datetime.strptime(f"{to_date} {to_hour:02d}:00",   "%Y-%m-%d %H:%M")
    if end <= cur:
        end = cur + timedelta(hours=1)

    while cur <= end:
        hr   = cur.hour
        mn   = cur.minute
        dow  = cur.weekday()
        mon  = cur.month
        is_w = 1 if dow >= 5 else 0
        quarter = (mon - 1) // 3 + 1
        dayofyear = cur.timetuple().tm_yday

        # Time encoding
        hr_sin = np.sin(2 * np.pi * hr / 24)
        hr_cos = np.cos(2 * np.pi * hr / 24)
        dow_sin = np.sin(2 * np.pi * dow / 7)
        dow_cos = np.cos(2 * np.pi * dow / 7)

        # Lags
        lag_1   = power_hist[-1]
        lag_2   = power_hist[-2]
        lag_4   = power_hist[-4]
        lag_8   = power_hist[-8]
        lag_16  = power_hist[-16]
        lag_96  = power_hist[-96]
        lag_672 = power_hist[-672]

        # Solar & Wind prev
        solar_p = solar_hist[-1]
        wind_p  = wind_hist[-1]

        # Rolling
        r_mean_4  = np.mean(power_hist[-4:])
        r_mean_16 = np.mean(power_hist[-16:])
        r_mean_96 = np.mean(power_hist[-96:])
        r_std_4   = np.std(power_hist[-4:])
        r_max_16  = np.max(power_hist[-16:])
        r_min_16  = np.min(power_hist[-16:])

        feats = [[
            hr, mn, dow, mon, is_w,
            quarter, dayofyear,
            hr_sin, hr_cos, dow_sin, dow_cos,
            temperature, humidity,
            lag_1, lag_2, lag_4, lag_8, lag_16, lag_96, lag_672,
            r_mean_4, r_mean_16, r_mean_96,
            r_std_4, r_max_16, r_min_16,
            solar_p, wind_p
        ]]

        if model_name == "Linear Regression":
            pred = float(model.predict(scaler.transform(feats))[0])
        else:
            pred = float(model.predict(feats)[0])

        pred = round(max(0.5, min(15.0, pred)), 3)
        level, color, lbl = classify_demand(pred)

        slots.append({
            "timestamp": cur.strftime("%Y-%m-%d %H:%M"),
            "hour_label": cur.strftime("%d %b %H:%M"),
            "hour": hr,
            "kw": pred,
            "kwh": round(pred * 0.25, 4),   # 15-min interval = 0.25h
            "level": level,
            "color": color,
            "label": lbl,
        })

        # Append to running history for autoregressive multi-step forecasting
        power_hist.append(pred)
        # Simple cyclical solar/wind estimators
        hbase = stats.get("avg_power_kw", 5.2)
        sol_val = max(0, hbase * 0.35 * np.sin(np.pi * (hr - 6) / 12)) if 6 <= hr <= 18 else 0
        wnd_val = hbase * 0.2 * (1 + 0.3 * np.sin(2 * np.pi * hr / 24))
        solar_hist.append(sol_val)
        wind_hist.append(wnd_val)

        cur += timedelta(minutes=15)

    total_kwh = round(sum(s["kwh"] for s in slots), 3)
    avg_kw    = round(sum(s["kw"] for s in slots) / len(slots), 3) if slots else 0
    peak_kw   = round(max(s["kw"] for s in slots), 3) if slots else 0
    lv, co, lb = classify_demand(avg_kw)

    return {
        "slots":      slots,
        "total_kwh":  total_kwh,
        "avg_kw":     avg_kw,
        "peak_kw":    peak_kw,
        "slot_count": len(slots),
        "demand":     lv,
        "color":      co,
        "desc":       lb,
        "model_name": model_name,
        "model_r2":   round(get_metrics().get("best_r2", 0), 4),
    }

def get_metrics():
    path = os.path.join(MODEL_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def classify_demand(kw):
    if kw < 3.0:   return "LOW",           "#10B981", "Low grid load – Off-peak / Night baseload."
    if kw < 6.0:   return "MODERATE",      "#3B82F6", "Moderate – Normal daytime operation."
    if kw < 9.0:   return "HIGH",          "#F59E0B", "High – Peak hours, HVAC & appliances active."
    return             "CRITICAL PEAK", "#EF4444", "Critical – Grid near capacity. Overload risk!"

def models_exist():
    return os.path.exists(os.path.join(MODEL_DIR, "best_model.joblib"))

if __name__ == "__main__":
    train()
