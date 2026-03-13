# GPS + IMU Sensor Fusion for Trajectory Estimation

**Prem M Patel** — Computer Engineer  
prempatel7740@gmail.com

---

## The Problem

Two sensors. Both imperfect. Neither alone reliable.

GPS tells you where you are on Earth — but only once per second, with three to five metres of noise, and nothing at all when you step indoors. IMU tracks every movement you make at a hundred readings per second — but its errors accumulate, and after a few minutes of integration, your estimated position has drifted metres from reality.

This project solves a problem that aerospace engineers, autonomous vehicle teams, and navigation researchers work on every day: how do you combine these two imperfect, asynchronous sensors into one continuous, accurate position estimate — and what happens when the GPS dies completely?

The answer involves a 60-year-old algorithm called the Kalman Filter, and a deep learning model called LSTM. The same pairing that guides ISRO rockets through the upper atmosphere, where GPS is unreliable and every metre of positional error matters.

---

## What This Project Does

The pipeline takes raw sensor data from a person walking through real indoor and outdoor environments in Finland and produces:

- A cleaned, noise-filtered IMU signal using a Butterworth low-pass filter
- A smooth, continuous trajectory from Kalman-fused GPS and IMU data
- A trained LSTM model that predicts position accurately with GPS completely absent
- An interactive map showing the real walking path on actual streets
- Quantitative accuracy metrics comparing the system against ground truth

The entire pipeline runs in Google Colab with no local setup required.

---

## Dataset

**ADVIO — Apple Device Visual Inertial Odometry**  
Collected by Aalto University, Helsinki, Finland.

| Property | Detail |
|---|---|
| Source | Zenodo record 1476931 |
| Sequence used | advio-01 |
| Environment | Shopping mall, metro station, street |
| IMU sampling rate | ~100 Hz |
| GPS sampling rate | ~1 Hz |
| Total samples | 25,971 (after cleaning) |
| Download size | ~120 MB |

The dataset contains accelerometer, gyroscope, GPS, barometer, and magnetometer readings from a standard iPhone, along with high-accuracy ground truth poses captured by a separate tracking system.

---

## Technical Approach

### Part 1 — Signal Processing

Raw IMU data from a walking person contains genuine motion signal mixed with high-frequency vibration noise from footsteps and phone casing resonance. A 4th-order Butterworth low-pass filter with a 5 Hz cutoff separates these cleanly, preserving the motion signal while removing noise above the human walking frequency range.

The filter is applied using `filtfilt` — a zero-phase implementation that runs the filter forward and backward through the signal, cancelling the phase delay that would otherwise shift sensor readings out of alignment with their true timestamps. For navigation, timestamp accuracy is as important as signal accuracy.

GPS coordinates update at 1 Hz. IMU updates at 100 Hz. Before any fusion can happen, both sensors must exist on the same timeline. Linear interpolation creates a GPS estimate at every IMU timestamp, filling the 99 gaps between each real GPS reading.

### Part 2 — Kalman Filter

The Kalman Filter maintains a state vector of four quantities at every timestep:

```
x = [ position_x, position_y, velocity_x, velocity_y ]
```

At each of the 25,971 timesteps, it runs two operations:

**Predict** — use the IMU acceleration to propagate the state forward through the physics of motion:

```
x_predicted = F * x_previous
P_predicted = F * P * F^T + Q
```

**Update** — correct the prediction using the GPS measurement, weighted by relative confidence:

```
K = P * H^T * inv(H * P * H^T + R)
x_updated = x_predicted + K * (z_gps - H * x_predicted)
P_updated = (I - K * H) * P_predicted
```

The Kalman Gain K is the key quantity. It is large when GPS is trustworthy relative to the prediction, and small when the IMU prediction is more reliable. The filter computes this balance automatically at every step, requiring no manual tuning beyond the noise covariance parameters R (GPS uncertainty, set to 5.0 m) and Q (process noise, set to 0.1).

When GPS is lost, the update step is skipped entirely. The filter continues running on IMU alone — this is dead reckoning, and it works well for short blackout windows of five to thirty seconds.

### Part 3 — LSTM for GPS-Loss Recovery

Dead reckoning from IMU integration drifts over time. For longer GPS blackouts, we need something smarter than pure integration — a model that has learned the relationship between IMU signal patterns and actual motion.

GPS blackouts are simulated by erasing GPS readings for five random windows of five to ten seconds each across the sequence. This gives the model training examples of what GPS loss looks like and what the correct position should be during that loss.

The model receives 200-timestep sliding windows (2 seconds of IMU data at 100 Hz) and predicts the 2D position at the end of each window. Input shape: (200, 6) — six sensor channels across 200 timesteps.

**Architecture:**

```
Input         (200, 6)
LSTM          128 units, return_sequences=True
BatchNorm + Dropout(0.3)
LSTM          64 units, return_sequences=False
BatchNorm + Dropout(0.3)
Dense         32 units, ReLU activation
Dropout(0.2)
Dense         2 units  (fused_x, fused_y)
```

Total parameters: 121,442.

Training uses Mean Squared Error loss, the Adam optimiser at an initial learning rate of 0.001, early stopping with patience 8, and learning rate reduction on plateau with factor 0.5. The 80/20 train-test split uses the first 412 windows for training and holds out the final 104 windows for evaluation.

### Part 4 — Visualisation

Results are visualised at every stage: raw sensor signals, filter comparison, GPS interpolation, Kalman trajectory, GPS blackout windows, training curves, prediction accuracy, and a final interactive Folium map showing the fused trajectory on real OpenStreetMap tiles.

---

## Results

| Metric | Value |
|---|---|
| Kalman Filter — average GPS deviation | 0.628 m |
| Kalman Filter — maximum GPS deviation | 0.655 m |
| LSTM — train MAE (normalised) | 0.2375 |
| LSTM — test MAE (normalised) | 0.2926 |
| LSTM — mean position error | 0.010 m |
| LSTM — maximum position error | 0.025 m |

The Kalman Filter reduces raw GPS noise from the 3–5 metre range down to sub-metre positional smoothness. The LSTM achieves centimetre-level mean position error on the held-out test set, with train and test MAE within 0.055 of each other — indicating the model generalised rather than memorised.

---

## Tech Stack

| Component | Library |
|---|---|
| Data handling | NumPy, Pandas |
| Signal processing | SciPy |
| Kalman Filter | FilterPy |
| Deep learning | TensorFlow / Keras |
| Evaluation | scikit-learn |
| Visualisation | Matplotlib, Folium |
| Environment | Google Colab |

---

## Repository Structure

```
.
├── Project1_Theory.ipynb          Theory notebook — all equations, concepts,
│                                  derivations. Read this before the code.
│
├── Project1_Implementation.ipynb  Implementation notebook — full pipeline,
│                                  documented step by step for Colab.
│
├── README.md                      This file.
│
└── outputs/                       Generated after running the notebook
    ├── raw_sensor_data.png
    ├── filtered_signal.png
    ├── gps_interpolation.png
    ├── fused_trajectory.png
    ├── gps_blackout.png
    ├── train_test_split.png
    ├── training_curves.png
    ├── lstm_predictions.png
    ├── lstm_error.png
    ├── trajectory_map.html
    ├── gps_imu_lstm_model.h5
    ├── scaler_X.pkl
    └── scaler_y.pkl
```

---

## How to Run

**1.** Open `Theory.ipynb` in Colab. Read it completely. No code to run.

**2.** Open `Implementation.ipynb` in Colab.

**3.** Enable GPU: `Runtime > Change runtime type > T4 GPU or v5e-1 TPU > Save`

**4.** Run cells in order from top to bottom. Each cell has a markdown explanation directly above it describing what it does and what to check in the output.

**5.** Total runtime is approximately 15 minutes on the Colab free tier.

The dataset downloads automatically in Cell 2 from Zenodo. No manual data setup is required.

---

## Real-World Relevance

The technique implemented here is not academic. It is the same fundamental approach used in production systems across multiple industries.

**Rocketry** — ISRO's Inertial Systems Unit runs a Kalman Filter in the flight computer of every PSLV, GSLV, and LVM3 launch. Above 30 km, GPS is unreliable. The vehicle navigates on IMU alone — the dead reckoning problem this LSTM was trained to solve. The F matrix, H matrix, and predict-update loop in Cell 7 are variants of what runs in that flight software.

**Aviation** — Commercial aircraft carry multiple independent Inertial Reference Systems. During GPS jamming events, which have become common in certain airspace regions, the IRS takes over completely. Airlines restrict speed and altitude in affected regions because their navigation degrades. This project demonstrates the alternative.

**Autonomous vehicles** — Every self-driving car fuses LiDAR, cameras, GPS, and IMU using a Kalman Filter running at 100+ Hz. In tunnels and underground car parks where GPS is unavailable, the vehicle falls back on IMU integration and learned motion models — the same combination of Kalman and LSTM implemented here.

**Indoor navigation** — GPS does not work inside buildings. Pedestrian dead reckoning using IMU, fused with WiFi or BLE beacons using Kalman filtering, is the dominant approach for hospital navigation, warehouse logistics, and airport terminal guidance.

**Wearables** — Activity trackers and smartwatches use accelerometer-based step detection and gyroscope-based heading estimation to track movement when GPS signal is weak or absent. The signal processing techniques in this project — particularly the zero-phase Butterworth filter — are standard in embedded sensor processing.

The physics is the same across all of these. Only the precision requirements and the hardware change.

---

## Key Concepts Implemented

- Butterworth low-pass filter — zero-phase implementation with `filtfilt`
- Nyquist theorem and frequency-domain signal analysis
- Linear interpolation for multi-rate sensor synchronisation
- Flat-earth coordinate projection (lat/lon to metres)
- Kalman Filter — state space model, transition matrix, measurement matrix, Kalman Gain
- Dead reckoning under GPS blackout
- Sliding window segmentation for time-series machine learning
- StandardScaler normalisation and inverse transform
- LSTM with stacked layers, BatchNormalization, and Dropout
- Early stopping and learning rate scheduling
- RMSE, MAE, and Euclidean distance error metrics
- Folium interactive map generation

---

## Author

**Prem M Patel**  
Computer Engineer
prempatel7740@gmail.com

---

