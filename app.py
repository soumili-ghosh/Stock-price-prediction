from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import requests  # <-- Added this import

app = Flask(__name__)
app.secret_key = "supersecretkey"
GEMINI_API_KEY = "AIzaSyAvKCYBnpQCMC_D1KcbwO3x68Sqh52Lsfo"  # Replace with your real key
GEMINI_MODEL = "gemini-2.5"

# DATABASE
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# USER MODEL
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

with app.app_context():
    db.create_all()

# =========================
# STOCK PREDICTION FUNCTION
# =========================
def stock_prediction(ticker="TCS.NS", forecast_date=None):
    if forecast_date is None:
        forecast_date = (pd.to_datetime("today") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    today = pd.to_datetime("today").normalize()
    df = yf.download(ticker, start="2020-01-01", end=today)
    df = df.dropna()

    if df.empty:
        return None, None

    df["Date"] = pd.to_datetime(df.index)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    X = df[["Days"]]
    y = df["Close"]

    X_train, X_test = X[:-100], X[-100:]
    y_train, y_test = y[:-100], y[-100:]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)

    # ARIMA
    arima = ARIMA(y_train, order=(5, 1, 0))
    arima_fit = arima.fit()

    # Forecast for given date
    future_days = (pd.to_datetime(forecast_date) - df["Date"].min()).days
    future_X = np.array([[future_days]])

    predictions = {
        "last_price": round(float(df["Close"].iloc[-1]), 2),
        "last_date": str(df["Date"].iloc[-1].date()),
        "forecast_date": forecast_date,
        "Linear Regression": round(float(lr.predict(future_X)[0]), 2),
        "Random Forest": round(float(rf.predict(future_X)[0]), 2),
        "Decision Tree": round(float(dt.predict(future_X)[0]), 2),
        "ARIMA": round(float(arima_fit.forecast(steps=future_days - len(y_train)).values[0]), 2),
    }

    performance = {
        "Linear Regression R2": r2_score(y_test, lr.predict(X_test)),
        "Random Forest R2": r2_score(y_test, rf.predict(X_test)),
        "Decision Tree R2": r2_score(y_test, dt.predict(X_test)),
        "ARIMA R2": r2_score(y_test, arima_fit.forecast(steps=100)),
    }

    return predictions, performance

# =========================
# STOCK LIST
# =========================
STOCKS = {
    "SUZLON.NS": "Suzlon Energy", "IDEA.NS": "Vodafone Idea", "YESBANK.NS": "Yes Bank", 
    "JPPOWER.NS": "Jaiprakash Power", "TTML.NS": "Tata Teleservices", "RENUKA.NS": "Shree Renuka Sugars", 
    "NHPC.NS": "NHPC", "IOC.NS": "Indian Oil", "PNB.NS": "Punjab National Bank", "BANKBARODA.NS": "Bank of Baroda", 
    "IDFCFIRSTB.NS": "IDFC First Bank", "RPOWER.NS": "Reliance Power", "JINDALSTEL.NS": "Jindal Steel", 
    "SAIL.NS": "Steel Authority of India", "BHEL.NS": "BHEL", "ONGC.NS": "ONGC", "GAIL.NS": "GAIL", 
    "BEL.NS": "Bharat Electronics", "HINDZINC.NS": "Hindustan Zinc", "UNIONBANK.NS": "Union Bank of India", 
    "CANBK.NS": "Canara Bank", "SOUTHBANK.NS": "South Indian Bank", "INDIANB.NS": "Indian Bank", 
    "TATAMOTORS.NS": "Tata Motors", "ASHOKLEY.NS": "Ashok Leyland", "TATASTEEL.NS": "Tata Steel", 
    "COALINDIA.NS": "Coal India", "HINDCOPPER.NS": "Hindustan Copper"
}

# =========================
# TECHNICAL ANALYSIS FUNCTION
# =========================
def analyze_stock(ticker):
    hist = yf.Ticker(ticker).history(period="6mo")
    if hist.empty:
        return {"error": "No data available"}

    support = round(hist["Low"].min(), 2)
    resistance = round(hist["High"].max(), 2)
    current = round(hist["Close"].iloc[-1], 2)
    stance = "Buy" if current < (support + resistance) / 2 else "Sell"

    return {
        "ticker": ticker,
        "support": support,
        "resistance": resistance,
        "current_price": current,
        "stance": stance,
        "entry": current,
        "stoploss": round(support * 0.95, 2),
        "target": round(resistance * 1.05, 2)
    }

# =========================
# ROUTES
# =========================
@app.route("/index", methods=["GET","POST"])
def index():
    ticker = "TCS.NS"
    forecast_date = "2027-12-25"

    if request.method == "POST":
        ticker = request.form.get("ticker")
        forecast_date = request.form.get("forecast_date")

    predictions, performance = stock_prediction(ticker, forecast_date)
    return render_template("index.html", predictions=predictions, performance=performance, ticker=ticker)
@app.route("/chat", methods=["POST"])
def chat():
    import requests  # make sure this is imported

    user_msg = request.json.get("message")
    
    headers = {
        "Authorization": f"Bearer {GEMINI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GEMINI_MODEL,
        "messages": [{"role": "user", "content": user_msg}]
    }

    try:
        response = requests.post(
            "https://gemini.googleapis.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        bot_reply = data["choices"][0]["message"]["content"]
    except Exception as e:
        print("Gemini API error:", e)
        bot_reply = "Sorry, I cannot answer right now."

    return jsonify({"reply": bot_reply})


@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password)

        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for("register"))

        user = User(username=username, email=email, password=hashed_password)
        db.session.add(user)
        db.session.commit()
        flash("Registration successful. Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            session["user_id"] = user.id
            session["username"] = user.username
            flash(f"Welcome {user.username}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid credentials.", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", stocks=STOCKS)

@app.route("/assistant", methods=["GET", "POST"])
def assistant():
    models = ["Gemini", "GPT-4", "Mistral", "Gamma"]
    prediction = None
    analysis = None
    chart_data = None
    ai_response = None

    if request.method == "POST":
        selected_model = request.form.get("model")
        ticker = request.form.get("ticker")
        forecast_date = request.form.get("forecast_date")  # <-- Fix here

        # Predictions
        prediction, _ = stock_prediction(ticker, forecast_date)

        # Last closing date
        hist = yf.Ticker(ticker).history(period="6mo")
        if not hist.empty:
            last_close_date = hist.index[-1].strftime("%Y-%m-%d")
            prediction["last_price"] = round(float(hist["Close"].iloc[-1]), 2)
            prediction["last_date"] = last_close_date

        # Technical analysis
        analysis = analyze_stock(ticker)

        # Chart data
        hist.reset_index(inplace=True)
        chart_data = {
            "dates": hist["Date"].astype(str).tolist(),
            "close": hist["Close"].tolist(),
            "open": hist["Open"].tolist(),
            "high": hist["High"].tolist(),
            "low": hist["Low"].tolist(),
            "volume": hist["Volume"].tolist(),
        }

        ai_response = f"""
        Using {selected_model}, here’s the stock insight:
        - Last Closing Price ({prediction['last_date']}): {prediction['last_price']}
        - Forecasted Price on {forecast_date}: {prediction['Linear Regression']}
        - Market stance: {analysis['stance']}
        - Suggested Entry: {analysis['entry']}, Stoploss: {analysis['stoploss']}, Target: {analysis['target']}
        """

    # <-- Make sure to always return render_template
    return render_template(
        "assistant.html",
        models=models,
        stocks=STOCKS,
        prediction=prediction,
        analysis=analysis,
        chart_data=json.dumps(chart_data) if chart_data else None,
        ai_response=ai_response
    )
@app.route("/volatility", methods=["GET", "POST"])
def Volatility():
    ticker = "TCS.NS"
    data = None

    if request.method == "POST":
        ticker = request.form.get("ticker")

        # Fetch stock data
        df = yf.download(ticker, period="6mo")
        df = df.dropna()

        if not df.empty:
            # Daily returns
            df["Daily Return"] = df["Close"].pct_change()

            # Rolling Volatility (20-day std dev of returns)
            df["Volatility"] = df["Daily Return"].rolling(window=20).std()

            # Collect summary stats
            data = {
                "ticker": ticker,
                "mean_volatility": round(df["Volatility"].mean(), 4),
                "max_volatility": round(df["Volatility"].max(), 4),
                "min_volatility": round(df["Volatility"].min(), 4),
            }

    return render_template("Volatility.html", ticker=ticker, data=data)

@app.route("/visuals", methods=["GET", "POST"])
def visuals():
    ticker = request.args.get('ticker', 'AAPL')  # Default Apple
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")

    # Prepare charts data
    hist.reset_index(inplace=True)
    data = {
        "dates": hist['Date'].astype(str).tolist(),
        "close": hist['Close'].tolist(),
        "open": hist['Open'].tolist(),
        "volume": hist['Volume'].tolist(),
        "high": hist['High'].tolist(),
        "low": hist['Low'].tolist()
    }

    # Get predictions for next 30 days
    forecast_date = (pd.to_datetime("today") + pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    predictions, _ = stock_prediction(ticker, forecast_date)

    pred_data = {
        "forecast_date": predictions["forecast_date"],
        "Linear Regression": predictions["Linear Regression"],
        "Random Forest": predictions["Random Forest"],
        "Decision Tree": predictions["Decision Tree"],
        "ARIMA": predictions["ARIMA"]
    }

    return render_template("visuals.html", ticker=ticker, data=json.dumps(data), pred=json.dumps(pred_data))


if __name__ == "__main__":
    app.run(debug=True)

