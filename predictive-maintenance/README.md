# 📱 Smartphone Predictive Maintenance & Live Phyphox Analyzer

A clean, responsive, single-page Streamlit dashboard that runs an **XGBoost classification model** on smartphone sensor data. It features a manual input form and a **Live Phyphox Sensor Analysis** section for real-time machine health monitoring and Remaining Useful Life (RUL) estimation.

---

## Project Structure

```text
predictive-maintenance/
│
├── app.py                  # Streamlit application dashboard
├── requirements.txt        # Python package dependencies
├── README.md               # User setup & operation instructions
│
├── models/
│   ├── xgboost_model.pkl   # Serialized trained XGBoost classifier
│   └── preprocessing.pkl   # Imputer medians & evaluation metrics
│
├── phyphox/
│   └── smartphone_predictive_maintenance.phyphox  # Custom experiment file
│
└── utils/
    ├── phyphox_client.py   # HTTP Client for Phyphox REST API
    ├── preprocessing.py    # Clean column mapping & imputer methods
    ├── prediction.py       # Wrapper for class index & probability predictions
    ├── feature_engineering.py # Window feature extractions (RMS, Std Dev, Variance)
    └── rul.py              # Health Score & RUL estimator algorithms
```

---

## 🚀 Setup & Execution Guide

Follow these steps to set up and run the predictive maintenance prototype:

### Step 1 — Install Dependencies
Ensure you have Python installed, then install the required Python packages:
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Streamlit Dashboard
Launch the dashboard from the project root directory:
```bash
streamlit run app.py
```
This opens the dashboard in your web browser at `http://localhost:8501`.

---

## 📡 Live Phone Connection Guide

To stream real-time data from your phone's sensors, configure the custom experiment in the **Phyphox** application.

### Step 3 — Load the Custom Experiment in Phyphox
Phyphox allows loading custom XML experiment configurations (`.phyphox` files) to specify sample rates and sensor buffers:
1. Transfer the file `phyphox/smartphone_predictive_maintenance.phyphox` from your laptop to your smartphone (via Email, messaging app, cloud drive, or local transfer).
2. Open the file on your smartphone. It will automatically prompt you to open it inside the **Phyphox** app.
3. The custom experiment **"Predictive Maintenance"** will now appear in your list of experiments. Open it.

### Step 4 — Enable Remote Access in Phyphox
1. Open the **Predictive Maintenance** experiment on your phone.
2. Tap the **Three Dots (options menu)** in the top-right corner.
3. Tap **Allow remote access** (or **Enable Remote Access**).
4. Tap **OK** on the warning dialog. A small URL address banner will appear at the bottom or top of the app screen (e.g. `http://192.168.1.100:8080`).

### Step 5 — Find the Phone's Phyphox URL
- Note the IP address and port displayed by the app. It must match the format:
  ```text
  http://<IP_ADDRESS>:<PORT>
  ```
- *Important:* Ensure your phone and laptop are connected to the same Wi-Fi network or mobile hotspot so they can communicate over HTTP.

### Step 6 — Connect the Device in the Dashboard
1. On the Streamlit dashboard, scroll to the **📡 Live Device Connection** section.
2. Paste the URL (e.g. `http://192.168.1.100:8080`) into the **Phyphox Device URL** input field.
3. Click the **🔌 Connect Device** button.
4. The status indicator should turn green and show **● CONNECTED**.

### Step 7 — Start Live Analysis
1. Click the **▶ Start Live Analysis** button on the dashboard.
2. The smartphone will start recording sensor readings, and the dashboard will query and stream values at a rate of ~4 Hz.
3. You will see values updating in the **📡 Live Sensor Values** panel, and rolling charts drawing the last 100 samples.

---

## 🧠 Real-Time Health & RUL Prediction

### Step 8 — Testing Good / Bad / Critical Predictions
The dashboard feeds the live measurements into the pre-trained XGBoost model to classify current device health:
* **🟢 GOOD**: When the phone is sitting flat and steady (default stationary state).
* **🟠 BAD**: When you pick up the phone, move it, or subject it to gentle rocking motions (simulating minor machine wear/wobble).
* **🔴 CRITICAL**: When you shake or jiggle the phone rapidly (simulating heavy mechanical failure, severe imbalance, or motor vibration).

### Step 9 — Understanding the RUL Estimate
The **Estimated Remaining Useful Life (RUL)** panel evaluates the phone's physical conditions over a rolling window of 100 samples:
1. **Health Score (%)**: Calculates a degradation index based on rolling Accel standard deviation (vibrational noise), Gyroscope RMS, and Linear Acceleration RMS. It merges this physical deviation with the classifier's probability of being in the "Good" state.
2. **Estimated RUL**: Translates the health score into a remaining lifetime window (up to 72 hours / 3.0 days). As vibrations increase and classification moves to BAD/CRITICAL, RUL decays non-linearly.
3. **Health Trend**: Displays **Stable** (Health > 85%), **Degrading** (Health 50% - 85%), or **Critical Alert** (Health < 50%) to summarize structural decay.
