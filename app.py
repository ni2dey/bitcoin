from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime
from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
from flask_bcrypt import Bcrypt
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/college"
mongo = PyMongo(app)
bcrypt = Bcrypt(app)
auth = HTTPBasicAuth()

# Set Matplotlib to non-interactive backend
matplotlib.use("Agg")

app = Flask(__name__)

# Load Pre-trained Model
model = load_model('model.keras')


# Helper Function to Convert Matplotlib Plots to HTML
def plot_to_html(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    buf.close()
    return f"data:image/png;base64,{data}"

# Corrected route to be more intuitive


@app.route('/')
def index():
    return render_template('index.html')


@auth.login_required
@app.route('/prediction', methods=['GET'])
def prediction():
    return render_template('prediction.html')


@app.route('/signup')
def signup():
    return render_template('signup.html')


@app.route('/login')
def login():
    return render_template('login.html')


@app.route('/auth/signup', methods=['POST'])
def register():
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type"}), 415

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    email = data.get('email')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    hashed_password = generate_password_hash(password)
    mongo.db.users.insert_one(
        {'username': username, 'email': email, 'password': hashed_password})
    return jsonify({"redirect": url_for('prediction')})


@auth.verify_password
def verify_password(username, password):
    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        return username
    return None


@app.route('/auth/login', methods=['POST'])
def login_logic():
    if request.content_type != 'application/json':
        return jsonify({"error": "Unsupported Media Type"}), 415

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return jsonify({"error": "Missing username or password"}), 400

    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        return jsonify({"redirect": url_for('prediction')})
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route('/auth/protected', methods=['GET'])
@auth.login_required
def protected():
    return jsonify({'message': f'This is a protected route, {auth.current_user()}!'})


@app.route('/pricepredict', methods=["GET", "POST"])
def price():
    if request.method == "POST":
        stock = request.form.get("stock")
        no_of_days = int(request.form.get("no_of_days"))
        return redirect(url_for("predict", stock=stock, no_of_days=no_of_days))
    return render_template("price.html")


@app.route("/predict")
def predict():
    stock = request.args.get("stock", "BTC-USD")
    no_of_days = int(request.args.get("no_of_days", 10))

    try:
        # Fetch Stock Data
        end = datetime.now()
        start = datetime(end.year - 10, end.month, end.day)
        stock_data = yf.download(stock, start, end)

        if stock_data.empty:
            return render_template(
                "result.html",
                stock=stock,
                error="Invalid stock ticker or no data available.",
            )

        # Data Preparation
        splitting_len = int(len(stock_data) * 0.9)
        x_test = stock_data[["Close"]][splitting_len:]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(x_test)

        x_data = []
        y_data = []
        for i in range(100, len(scaled_data)):
            x_data.append(scaled_data[i - 100: i])
            y_data.append(scaled_data[i])

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        # Predictions
        predictions = model.predict(x_data)
        inv_predictions = scaler.inverse_transform(predictions)
        inv_y_test = scaler.inverse_transform(y_data)

        # Prepare Data for Plotting
        plotting_data = pd.DataFrame(
            {
                "Original Test Data": inv_y_test.flatten(),
                "Predicted Test Data": inv_predictions.flatten(),
            },
            index=x_test.index[100:],
        )

        # Generate Plots
        # Plot 1: Original Closing Prices
        fig1 = plt.figure(figsize=(15, 6))
        plt.plot(stock_data["Close"], "b", label="Close Price")
        plt.title("Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        plt.legend()
        original_plot = plot_to_html(fig1)

        # Plot 2: Original vs Predicted Test Data
        fig2 = plt.figure(figsize=(15, 6))
        plt.plot(plotting_data["Original Test Data"],
                 label="Original Test Data")
        plt.plot(
            plotting_data["Predicted Test Data"],
            label="Predicted Test Data",
            linestyle="--",
        )
        plt.legend()
        plt.title("Original vs Predicted Closing Prices")
        plt.xlabel("Date")
        plt.ylabel("Close Price")
        predicted_plot = plot_to_html(fig2)

        # Plot 3: Future Predictions
        last_100 = stock_data[["Close"]].tail(100)
        last_100_scaled = scaler.transform(last_100)

        future_predictions = []
        last_100_scaled = last_100_scaled.reshape(1, -1, 1)
        for _ in range(no_of_days):
            next_day = model.predict(last_100_scaled)
            future_predictions.append(scaler.inverse_transform(next_day))
            last_100_scaled = np.append(
                last_100_scaled[:, 1:, :], next_day.reshape(1, 1, -1), axis=1
            )

        future_predictions = np.array(future_predictions).flatten()

        fig3 = plt.figure(figsize=(15, 6))
        plt.plot(
            range(1, no_of_days + 1),
            future_predictions,
            marker="o",
            label="Predicted Future Prices",
            color="purple",
        )
        plt.title("Future Close Price Predictions")
        plt.xlabel("Days Ahead")
        plt.ylabel("Predicted Close Price")
        plt.grid(alpha=0.3)
        plt.legend()
        future_plot = plot_to_html(fig3)

        return render_template(
            "result.html",
            stock=stock,
            original_plot=original_plot,
            predicted_plot=predicted_plot,
            future_plot=future_plot,
            enumerate=enumerate,
            future_predictions=future_predictions,
        )
    except Exception as e:
        return render_template(
            "result.html", stock=stock, error=f"An error occurred: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)
