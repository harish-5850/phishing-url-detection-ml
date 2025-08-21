from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import re
from urllib.parse import urlparse
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Replace with a secure key in production

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Mock user database (replace with a real database like SQLite in production)
users = {}

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, username):
        self.id = username

@login_manager.user_loader
def load_user(username):
    if username in users:
        return User(username)
    return None

# Train the Random Forest model
def train_model():
    dataset_path = r"C:\Users\rutht\edygrade\dataset\dataset_phishing.csv"
    
    reduced_features = [
        "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens", "nb_at", "nb_qm",
        "nb_and", "nb_eq", "nb_slash", "http_in_path", "https_token", "ratio_digits_url",
        "prefix_suffix", "nb_subdomains"
    ]

    # Check for dataset
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Cannot start application: dataset_phishing.csv not found at {dataset_path}. "
            "Please place the dataset in the specified path."
        )

    try:
        df = pd.read_csv(dataset_path)
        df['status'] = df['status'].map({'phishing': 0, 'legitimate': 1})
        df.fillna(0, inplace=True)

        X_reduced = df[reduced_features]
        y = df['status']

        X_train_r, _, y_train_r, _ = train_test_split(X_reduced, y, test_size=0.2, stratify=y, random_state=42)

        scaler = StandardScaler()
        X_train_r_scaled = scaler.fit_transform(X_train_r)

        rf_model = RandomForestClassifier(random_state=42)
        rf_model.fit(X_train_r_scaled, y_train_r)

        return rf_model, scaler, reduced_features
    except Exception as e:
        raise RuntimeError(f"Error training model: {e}")

# Train model at startup
try:
    rf_model, scaler, reduced_features = train_model()
except (FileNotFoundError, RuntimeError) as e:
    print(e)
    exit(1)

# URL feature extraction (no tldextract)
def extract_url_features(url):
    parsed = urlparse(url)
    hostname = parsed.hostname or ""
    path = parsed.path or ""

    # Approximate subdomains by splitting hostname
    if hostname:
        hostname_parts = hostname.split('.')
        nb_subdomains = max(0, len(hostname_parts) - 2)  # e.g., sub.sub.example.com -> 2
    else:
        nb_subdomains = 0

    features = {
        "length_url": len(url),
        "length_hostname": len(hostname),
        "ip": 1 if re.match(r"\d{1,3}(\.\d{1,3}){3}", hostname) else 0,
        "nb_dots": url.count('.'),
        "nb_hyphens": url.count('-'),
        "nb_at": url.count('@'),
        "nb_qm": url.count('?'),
        "nb_and": url.count('&'),
        "nb_eq": url.count('='),
        "nb_slash": url.count('/'),
        "http_in_path": int('http' in path.lower()),
        "https_token": int('https' in url.lower() and not url.startswith('https')),
        "ratio_digits_url": sum(c.isdigit() for c in url) / len(url) if len(url) > 0 else 0,
        "prefix_suffix": int('-' in hostname),  # Check hyphen in hostname
        "nb_subdomains": nb_subdomains
    }
    return features

# Predict URL
def predict_url_reduced(model, scaler, url, feature_list):
    try:
        features_dict = extract_url_features(url)
        features_df = pd.DataFrame([features_dict])

        for col in feature_list:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[feature_list]
        input_scaled = scaler.transform(features_df)

        prediction = model.predict(input_scaled)[0]
        return "Legitimate" if prediction == 1 else "Phishing"
    except Exception as e:
        raise ValueError(f"Error predicting URL: {e}")

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            user = User(username)
            login_user(user)
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users:
            flash('Username already exists', 'error')
        else:
            users[username] = password  # Store in memory (use database in production)
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/dashboard', methods=['GET', 'POST'])
@login_required
def dashboard():
    result = None
    url = None
    if request.method == 'POST':
        url = request.form['url'].strip()
        if not url:
            flash('Please enter a valid URL', 'error')
        else:
            try:
                result = predict_url_reduced(rf_model, scaler, url, reduced_features)
                return render_template('result.html', result=result, url=url)
            except ValueError as e:
                flash(str(e), 'error')
    return render_template('dashboard.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)