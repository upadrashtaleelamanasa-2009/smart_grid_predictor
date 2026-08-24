"""
app.py – Smart Grid Energy Dashboard (Flask)
4-page multi-route app: Login → Home → Prediction → Dashboard
"""
import os, json, threading
from datetime import datetime, timedelta
from flask import (Flask, render_template, redirect, url_for,
                   session, request, jsonify, flash)
from flask_session import Session

import ml_pipeline as ml

import shutil
session_dir = os.path.join(os.path.dirname(__file__), ".flask_session")
if os.path.exists(session_dir):
    try:
        shutil.rmtree(session_dir)
    except Exception:
        pass
os.makedirs(session_dir, exist_ok=True)

app = Flask(__name__)
app.secret_key = "SmartGrid_SecretKey_2024_XK9#"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = session_dir
app.config["SESSION_PERMANENT"] = False
Session(app)

# ── Boot: train model if needed (background) ───────────────────────────
def _boot_train():
    if not ml.models_exist():
        print("[BOOT] Training ML model in background...")
        try:
            ml.train()
            print("[BOOT] Training complete.")
        except Exception as e:
            print(f"[BOOT] Training error: {e}")

threading.Thread(target=_boot_train, daemon=True).start()

# ── Helpers ────────────────────────────────────────────────────────────
from functools import wraps
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        u = session.get("user")
        if not u:
            return redirect(url_for("login"))
        if isinstance(u, str):
            db = get_user_db()
            user_rec = db.get(u.lower(), {})
            session["user"] = {
                "name":   user_rec.get("name") or u.split("@")[0].title(),
                "email":  u,
                "avatar": (user_rec.get("name") or u)[0].upper(),
                "role":   user_rec.get("role", "User"),
                "via":    "email"
            }
        return f(*args, **kwargs)
    return decorated

def get_analytics():
    path = os.path.join(os.path.dirname(__file__), "models", "analytics.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def get_metrics():
    return ml.get_metrics()

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 – LOGIN
# ══════════════════════════════════════════════════════════════════════


# ── Persistent User Database (users.json) ──────────────────────────────
USERS_FILE = os.path.join(os.path.dirname(__file__), "data", "users.json")

import hashlib

def _hash_password(pw):
    return hashlib.sha256(pw.encode('utf-8')).hexdigest()

def get_user_db():
    if os.path.exists(USERS_FILE):
        try:
            with open(USERS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    # Default seed users
    default_db = {
        "operator@smartgrid.org": {
            "email": "operator@smartgrid.org",
            "password_hash": _hash_password("demo123"),
            "name": "Operator Admin",
            "role": "Operator",
            "created_at": datetime.now().isoformat()
        },
        "user@smartgrid.org": {
            "email": "user@smartgrid.org",
            "password_hash": _hash_password("demo123"),
            "name": "Smart Grid User",
            "role": "User",
            "created_at": datetime.now().isoformat()
        }
    }
    save_user_db(default_db)
    return default_db

def save_user_db(db):
    os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)
    with open(USERS_FILE, 'w') as f:
        json.dump(db, f, indent=2)

def register_or_update_user(email, password, name=None, role="User"):
    db = get_user_db()
    email_clean = email.strip().lower()
    pw_hash = _hash_password(password)
    
    if email_clean in db:
        user_rec = db[email_clean]
        user_rec["last_login"] = datetime.now().isoformat()
        if name:
            user_rec["name"] = name
    else:
        if not name:
            name = email_clean.split("@")[0].replace(".", " ").title()
        user_rec = {
            "email": email_clean,
            "password_hash": pw_hash,
            "name": name,
            "role": role,
            "created_at": datetime.now().isoformat(),
            "last_login": datetime.now().isoformat()
        }
        db[email_clean] = user_rec
    
    save_user_db(db)
    return user_rec

# ══════════════════════════════════════════════════════════════════════
# PAGE 1 – LOGIN & REGISTRATION
# ══════════════════════════════════════════════════════════════════════
@app.context_processor
def inject_user():
    user = session.get("user")
    if isinstance(user, str):
        db = get_user_db()
        user_rec = db.get(user.lower(), {})
        user = {
            "name":   user_rec.get("name") or user.split("@")[0].title(),
            "email":  user,
            "avatar": (user_rec.get("name") or user)[0].upper(),
            "role":   user_rec.get("role", "User"),
            "via":    "email"
        }
        session["user"] = user
    elif not isinstance(user, dict):
        user = None

    role = user.get("role", "User") if isinstance(user, dict) else "User"
    return {
        "user": user,
        "show_sidebar": True if role == "Operator" else False
    }

@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        return auth_email()
    if request.path == "/login":
        session.clear()
    elif session.get("user"):
        return redirect(url_for("home"))
    return render_template("login.html", active_page="login", show_sidebar=False)

@app.route("/auth/signup", methods=["POST"])
def auth_signup():
    name     = request.form.get("name", "").strip()
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()
    role     = request.form.get("role", "User").strip()

    if not email or "@" not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("login"))
    if not password or len(password) < 4:
        flash("Password must be at least 4 characters long.", "error")
        return redirect(url_for("login"))

    db = get_user_db()
    if email in db:
        flash("Account already exists. Please log in.", "warning")
        return redirect(url_for("login"))

    user_rec = register_or_update_user(email, password, name, role)
    session["user"] = {
        "name":   user_rec["name"],
        "email":  user_rec["email"],
        "avatar": user_rec["name"][0].upper(),
        "role":   user_rec["role"],
        "via":    "email"
    }
    session.permanent = True
    flash(f"Account registered successfully! Welcome, {user_rec['name']}.", "success")
    return redirect(url_for("home"))

@app.route("/auth/email", methods=["POST"])
def auth_email():
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("password", "").strip()

    if not email or "@" not in email:
        flash("Please enter a valid email address.", "error")
        return redirect(url_for("login"))
    if not password:
        flash("Please enter your password.", "error")
        return redirect(url_for("login"))

    db = get_user_db()
    pw_hash = _hash_password(password)

    # 1. Verify email exists in database
    if email not in db:
        flash("Account not found. Please click 'Create Account' to register your details.", "error")
        return redirect(url_for("login"))

    user_rec = db[email]

    # 2. Verify exact password set when creating account
    stored_hash = user_rec.get("password_hash")
    if stored_hash and stored_hash != pw_hash:
        if stored_hash == _hash_password("oauth_firebase"):
            # Set password for Google OAuth account setting an email password
            user_rec["password_hash"] = pw_hash
        else:
            flash("Incorrect password. Please enter the password you set while creating your account.", "error")
            return redirect(url_for("login"))

    # Update last login timestamp
    user_rec["last_login"] = datetime.now().isoformat()
    db[email] = user_rec
    save_user_db(db)

    session["user"] = {
        "name":   user_rec.get("name") or email.split("@")[0].replace(".", " ").title(),
        "email":  user_rec["email"],
        "avatar": (user_rec.get("name") or email)[0].upper(),
        "role":   user_rec.get("role", "User"),
        "via":    "email"
    }
    session.permanent = True
    flash(f"Welcome back, {session['user']['name']}! Signed in as {session['user']['role']}.", "success")
    return redirect(url_for("home"))

@app.route("/auth/firebase", methods=["POST"])
def auth_firebase():
    """Called by Firebase JS SDK after successful Google sign-in"""
    data  = request.get_json(silent=True) or {}
    name  = data.get("displayName", "User")
    email = data.get("email", "").lower()
    photo = data.get("photoURL", "")
    if not email:
        return jsonify({"ok": False, "error": "No email"}), 400

    # Save/update user details in persistent JSON database
    user_rec = register_or_update_user(email, "oauth_firebase", name)

    session["user"] = {
        "name":   name,
        "email":  email,
        "avatar": photo or name[0].upper(),
        "role":   user_rec.get("role", "User"),
        "via":    "google"
    }
    session.permanent = True
    return jsonify({"ok": True, "redirect": url_for("home")})

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ══════════════════════════════════════════════════════════════════════
# PAGE 2 – HOME / INPUT
# ══════════════════════════════════════════════════════════════════════
@app.route("/home")
@login_required
def home():
    today = datetime.now().strftime("%Y-%m-%d")
    return render_template("home.html",
        user=session["user"],
        today=today,
        models_ready=ml.models_exist(),
        active_page="home",
        step=1,
        show_sidebar=True
    )

# ══════════════════════════════════════════════════════════════════════
# PAGE 3 – PREDICTION
# ══════════════════════════════════════════════════════════════════════
@app.route("/prediction", methods=["GET", "POST"])
@login_required
def prediction():
    if request.method == "GET":
        result = session.get("prediction")
        if not result and ml.models_exist():
            today_str = datetime.now().strftime("%Y-%m-%d")
            try:
                result = ml.predict_range(today_str, today_str, 7, 19, 28.0, 60.0)
                session["prediction"] = result
                session["prediction_params"] = {
                    "from_date": today_str, "to_date": today_str,
                    "from_hour": 7, "to_hour": 19,
                    "temperature": 28.0, "humidity": 60.0
                }
            except Exception as e:
                pass
        if not result:
            flash("Please generate a prediction first from the Input Details page.", "warning")
            return redirect(url_for("home"))
        params = session.get("prediction_params", {})
        return render_template("prediction.html",
            user=session["user"],
            r=result,
            from_date=params.get("from_date", datetime.now().strftime("%Y-%m-%d")),
            to_date=params.get("to_date", datetime.now().strftime("%Y-%m-%d")),
            from_hour=params.get("from_hour", 7),
            to_hour=params.get("to_hour", 19),
            temperature=params.get("temperature", 28),
            humidity=params.get("humidity", 60),
            active_page="prediction",
            step=2,
            show_sidebar=True
        )

    from_date   = request.form.get("from_date",  datetime.now().strftime("%Y-%m-%d"))
    to_date     = request.form.get("to_date",    datetime.now().strftime("%Y-%m-%d"))
    from_hour   = int(request.form.get("from_hour", 7))
    to_hour     = int(request.form.get("to_hour",  19))
    temperature = float(request.form.get("temperature", 28))
    humidity    = float(request.form.get("humidity", 60))

    if not ml.models_exist():
        flash("ML model is still training. Please wait a moment.", "warning")
        return redirect(url_for("home"))

    try:
        result = ml.predict_range(from_date, to_date, from_hour, to_hour,
                                  temperature, humidity)
        session["prediction"] = result
        session["prediction_params"] = {
            "from_date": from_date, "to_date": to_date,
            "from_hour": from_hour, "to_hour": to_hour,
            "temperature": temperature, "humidity": humidity
        }
        return render_template("prediction.html",
            user=session["user"],
            r=result,
            from_date=from_date, to_date=to_date,
            from_hour=from_hour, to_hour=to_hour,
            temperature=temperature, humidity=humidity,
            active_page="prediction",
            step=2,
            show_sidebar=True
        )
    except Exception as e:
        flash(f"Prediction error: {e}", "error")
        return redirect(url_for("home"))

# ══════════════════════════════════════════════════════════════════════
# PAGE 4 – DASHBOARD
# ══════════════════════════════════════════════════════════════════════
@app.route("/dashboard")
@login_required
def dashboard():
    pred  = session.get("prediction", {})
    if not pred:
        return redirect(url_for("home"))
    return render_template("dashboard.html",
        user=session["user"],
        pred=pred,
        metrics=get_metrics(),
        analytics=get_analytics(),
        active_page="dashboard",
        step=3,
        show_sidebar=True
    )

# ══════════════════════════════════════════════════════════════════════
# API ENDPOINTS (called by dashboard JS)
# ══════════════════════════════════════════════════════════════════════
@app.route("/api/dataset-sample")
@login_required
def api_dataset_sample():
    """Return sampled historical data for charts"""
    import pandas as pd
    import numpy as np
    period = request.args.get("period", "7d")
    rmap   = {"24h": 96, "7d": 672, "30d": 2880, "all": 9600}
    n      = rmap.get(period, 672)

    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "smart_grid_dataset.csv"))
    temp_col = [c for c in df.columns if "Temp" in c][0]
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    sub   = df.tail(n).copy()
    step  = max(1, len(sub) // 400)
    sub   = sub.iloc[::step].reset_index(drop=True)

    return jsonify({
        "timestamps":   sub["Timestamp"].dt.strftime("%Y-%m-%d %H:%M").tolist(),
        "power":        sub["Power Consumption (kW)"].round(3).tolist(),
        "predicted":    sub["Predicted Load (kW)"].round(3).tolist(),
        "solar":        sub["Solar Power (kW)"].round(3).tolist(),
        "wind":         sub["Wind Power (kW)"].round(3).tolist(),
        "price":        sub["Electricity Price (USD/kWh)"].round(4).tolist(),
        "power_factor": sub["Power Factor"].round(3).tolist(),
        "overload":     sub["Overload Condition"].tolist(),
        "voltage":      sub["Voltage (V)"].round(2).tolist(),
    })

@app.route("/api/analytics")
@login_required
def api_analytics():
    return jsonify(get_analytics())

@app.route("/api/metrics")
@login_required
def api_metrics():
    return jsonify(get_metrics())

@app.route("/api/model-status")
def api_model_status():
    return jsonify({"ready": ml.models_exist()})

@app.route("/api/export-csv")
@login_required
def api_export_csv():
    from flask import Response
    import csv, io
    pred = session.get("prediction", {})
    slots = pred.get("slots", [])
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["timestamp","kw","kwh","level"])
    w.writeheader()
    for s in slots:
        w.writerow({"timestamp": s["timestamp"], "kw": s["kw"],
                    "kwh": s["kwh"], "level": s["level"]})
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=smart_grid_report.csv"})

@app.route("/admin/users")
@login_required
def admin_users():
    u = session.get("user", {})
    if u.get("role") != "Operator":
        flash("Access restricted to Grid Operators only.", "error")
        return redirect(url_for("home"))
    db = get_user_db()
    users_list = list(db.values())
    return render_template("users_admin.html",
                           show_sidebar=True,
                           user=u,
                           active_page="admin_users",
                           users_list=users_list)

@app.route("/api/export-users-csv")
@login_required
def api_export_users_csv():
    u = session.get("user", {})
    if u.get("role") != "Operator":
        flash("Access restricted to Grid Operators only.", "error")
        return redirect(url_for("home"))
    import csv, io
    db = get_user_db()
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["name", "email", "role", "created_at", "last_login"])
    w.writeheader()
    for u_rec in db.values():
        w.writerow({
            "name": u_rec.get("name", ""),
            "email": u_rec.get("email", ""),
            "role": u_rec.get("role", "User"),
            "created_at": u_rec.get("created_at", ""),
            "last_login": u_rec.get("last_login", "")
        })
    return Response(buf.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=registered_users_database.csv"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
