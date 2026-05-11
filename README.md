# 📈 Stock Price Prediction Using LSTM Deep Learning

## 📌 Introduction

Stock market prediction is one of the most challenging problems in financial analysis because stock prices are highly dynamic and influenced by many external factors such as news, company performance, economic conditions, and investor sentiment.

This project implements a **Deep Learning based Stock Price Prediction System** using **LSTM (Long Short-Term Memory)** networks. The model is trained using historical stock market data and predicts future stock prices based on past trends.

The project demonstrates:
- Time-series forecasting
- Data preprocessing
- Deep learning with TensorFlow/Keras
- Financial data visualization
- Predictive analytics

---

# 🎯 Objectives

The main objectives of this project are:

- To collect historical stock market data
- To preprocess and normalize the data
- To build an LSTM deep learning model
- To train the model on historical prices
- To predict future stock prices
- To compare actual vs predicted prices visually
- To predict the next day stock price

---

# 🧠 What is LSTM?

LSTM (Long Short-Term Memory) is a special type of Recurrent Neural Network (RNN) designed to handle sequential and time-series data.

Unlike traditional neural networks, LSTM can:
- Remember long-term dependencies
- Learn trends over time
- Handle sequential information efficiently

This makes LSTM highly suitable for:
- Stock market forecasting
- Weather prediction
- Speech recognition
- Time-series analysis

---

# 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow/Keras | Deep Learning Framework |
| Pandas | Data Manipulation |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Scikit-learn | Data Preprocessing |
| Yahoo Finance API | Stock Data Collection |

---

# 📂 Project Structure

```
stock_lstm/
│
├── stock_lstm.py          # Main project source code
├── requirements.txt       # Required Python libraries
├── README.md              # Project documentation
└── 
```

# ⚙️ System Requirements

## Hardware Requirements

- Minimum 4GB RAM
- Dual Core Processor
- Internet connection for downloading stock data

---

## Software Requirements

- Python 3.9 or higher
- VS Code / PyCharm / Jupyter Notebook (Optional)

---

# 📦 Installation Guide

## Step 1: Clone the Repository

```bash
git clone <repository-link>
cd stock_lstm
```

---

## Step 2: Create Virtual Environment (Optional but Recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run the following command:

```bash
python stock_lstm.py
```

---

# 📊 Dataset Information

The project uses stock market data from Yahoo Finance.

The dataset includes:

- Open Price
- High Price
- Low Price
- Close Price
- Volume

For prediction, only the **Close Price** is used.

---

# 🔄 Project Workflow

## 1️⃣ Data Collection

Historical stock data is downloaded using:

```python
yf.download()
```

### Example

```python
stock = "AAPL"
```

---

## 2️⃣ Data Preprocessing

The preprocessing steps include:

- Selecting closing prices
- Removing missing values
- Scaling values between 0 and 1 using MinMaxScaler
- Creating sequences of previous 60 days

---

## 3️⃣ Feature Scaling

Normalization is important because neural networks perform better when values are in a smaller range.

The project uses:

```python
MinMaxScaler(feature_range=(0,1))
```

---

## 4️⃣ Sequence Generation

The model uses the previous 60 days of stock prices to predict the next day.

| Previous 60 Days | Predicted Day |
|------------------|---------------|
| Day 1 → Day 60 | Day 61 |

---

## 5️⃣ Building the LSTM Model

The model architecture contains:

- 3 LSTM layers
- Dropout layers to reduce overfitting
- Dense output layer

---

## 🧠 Model Architecture

```text
Input Layer
   ↓
LSTM Layer
   ↓
Dropout
   ↓
LSTM Layer
   ↓
Dropout
   ↓
LSTM Layer
   ↓
Dropout
   ↓
Dense Output Layer
```

---

# 🧮 Model Parameters

| Parameter | Value |
|-----------|-------|
| Epochs | 25 |
| Batch Size | 32 |
| Sequence Length | 60 |
| Optimizer | Adam |
| Loss Function | Mean Squared Error |

---

# 📈 Output

The project generates:

- Predicted stock prices
- Actual vs Predicted graph
- Next-day predicted price

---

# 📉 Sample Output

## Actual vs Predicted Stock Prices

The graph compares:

- Actual market prices
- Model predicted prices

This helps evaluate model performance visually.

---

# 🔮 Next Day Prediction

The model predicts the next trading day's closing price.

### Example Output

```text
Next Day Predicted Price: 213.45
```

---

# 📊 Example Stocks

You can test the model using different stock symbols.

| Company | Symbol |
|---------|--------|
| Apple | AAPL |
| Tesla | TSLA |
| Microsoft | MSFT |
| Google | GOOG |
| Reliance | RELIANCE.NS |
| TCS | TCS.NS |

---

# 👨‍💻 Author

## Shubham Dalvi
B.Sc Computer Science <br>
M.Sc Computer Science