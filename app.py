from flask import Flask, render_template, request, redirect, url_for, session, flash
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import r2_score
import json
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

app.secret_key = "supersecretkey"

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

# Create DB tables
with app.app_context():
    db.create_all()


# STOCK PREDICTION FUNCTION 
def stock_prediction(ticker="TCS.NS", forecast_date="2027-12-25"):
    df = yf.download(ticker, start="2020-01-01", end="2025-08-22")
    df = df.dropna()

    # Use only Close price
    df["Date"] = pd.to_datetime(df.index)
    df["Days"] = (df["Date"] - df["Date"].min()).dt.days

    X = df[["Days"]]
    y = df["Close"]

    # Train-Test Split
    X_train, X_test = X[:-100], X[-100:]
    y_train, y_test = y[:-100], y[-100:]

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    lr_r2 = r2_score(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    rf_r2 = r2_score(y_test, y_pred_rf)

    #  Decision Tree 
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    dt_r2 = r2_score(y_test, y_pred_dt)

    # ARIMA 
    arima = ARIMA(y_train, order=(5,1,0))
    arima_fit = arima.fit()
    y_pred_arima = arima_fit.forecast(steps=100)
    arima_r2 = r2_score(y_test, y_pred_arima)

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
        "Linear Regression R2": lr_r2,
        "Random Forest R2": rf_r2,
        "Decision Tree R2": dt_r2,
        "ARIMA R2": arima_r2,
    }

    return predictions, performance
@app.route("/visuals", methods=["GET", "POST"])
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

@app.route("/", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        password = request.form.get("password")
        hashed_password = generate_password_hash(password, method="pbkdf2:sha256")


        if User.query.filter_by(email=email).first():
            flash("Email already exists!", "danger")
            return redirect(url_for("register"))

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("Registration Successful! Please login.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")
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


@app.route("/index", methods=["GET","POST"])
def index():
    ticker = "TCS.NS"
    forecast_date = "2027-12-25"

    if request.method == "POST":
        ticker = request.form.get("ticker")
        forecast_date = request.form.get("forecast_date")

    predictions, performance = stock_prediction(ticker, forecast_date)
    return render_template("index.html", predictions=predictions, performance=performance, ticker=ticker)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    ticker = "TCS.NS"
    return render_template("dashboard.html", ticker=ticker)

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
            flash("Invalid Email or Password!", "danger")
            return redirect(url_for("login"))

    return render_template("login.html")



if __name__ == "__main__":
    app.run(debug=True)
