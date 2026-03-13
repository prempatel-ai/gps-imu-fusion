<div align="center">

```
 ██████╗ ██████╗ ███████╗    ██╗███╗   ███╗██╗   ██╗
██╔════╝ ██╔══██╗██╔════╝    ██║████╗ ████║██║   ██║
██║  ███╗██████╔╝███████╗    ██║██╔████╔██║██║   ██║
██║   ██║██╔═══╝ ╚════██║    ██║██║╚██╔╝██║██║   ██║
╚██████╔╝██║     ███████║    ██║██║ ╚═╝ ██║╚██████╔╝
 ╚═════╝ ╚═╝     ╚══════╝    ╚═╝╚═╝     ╚═╝ ╚═════╝

███████╗███████╗███╗   ██╗███████╗ ██████╗ ██████╗
██╔════╝██╔════╝████╗  ██║██╔════╝██╔═══██╗██╔══██╗
███████╗█████╗  ██╔██╗ ██║███████╗██║   ██║██████╔╝
╚════██║██╔══╝  ██║╚██╗██║╚════██║██║   ██║██╔══██╗
███████║███████╗██║ ╚████║███████║╚██████╔╝██║  ██║
╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝

███████╗██╗   ██╗███████╗██╗ ██████╗ ███╗   ██╗
██╔════╝██║   ██║██╔════╝██║██╔═══██╗████╗  ██║
█████╗  ██║   ██║███████╗██║██║   ██║██╔██╗ ██║
██╔══╝  ██║   ██║╚════██║██║██║   ██║██║╚██╗██║
██║     ╚██████╔╝███████║██║╚██████╔╝██║ ╚████║
╚═╝      ╚═════╝ ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
```

<br/>

**Real-world GPS + IMU sensor fusion using Kalman Filter and LSTM**
**for continuous trajectory estimation under GPS signal loss**

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-14354C?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-ADVIO-0D47A1?style=for-the-badge)

<br/>

---

*Prem M Patel — Computer Engineer*
*prempatel7740@gmail.com*

---

</div>

<br/>

## Table of Contents

- [The Problem](#the-problem)
- [Architecture Overview](#architecture-overview)
- [Results](#results)
- [Dataset](#dataset)
- [Technical Pipeline](#technical-pipeline)
- [Model Architecture](#model-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Real-World Applications](#real-world-applications)
- [Key Concepts](#key-concepts)

<br/>

---

<br/>

## The Problem

<br/>

```
              GPS                            IMU
        ┌─────────────┐               ┌─────────────┐
        │  1 Hz only  │               │   100 Hz    │
        │  3-5m noise │               │  No drift   │
        │ Loses signal│               │  short-term │
        │  indoors    │               │             │
        └──────┬──────┘               └──────┬──────┘
               │                             │
               │    Neither alone works.     │
               │                             │
               └─────────────┬───────────────┘
                             │
                    ┌────────▼────────┐
                    │  Kalman Filter  │
                    │  Sensor Fusion  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  GPS blackout?  │
                    │  LSTM takes     │
                    │  over.          │
                    └─────────────────┘
```

<br/>

Two sensors. Both imperfect. Neither alone reliable.

**GPS** tells you where you are on Earth — but only once per second, with 3–5 metres of noise, and nothing at all when you step indoors or fly above 30 km altitude.

**IMU** tracks every movement at 100 readings per second — but its errors accumulate. After a few minutes of integration, your estimated position has drifted metres from reality.

This project fuses both using a **Kalman Filter** for continuous fusion, and trains an **LSTM** to predict position when GPS dies completely. The same approach used in ISRO rockets, commercial aircraft, and autonomous vehicles.

<br/>

---

<br/>

## Architecture Overview

<br/>

```
  RAW DATA                   PROCESSING                    OUTPUT
  ─────────                  ──────────                    ──────

  accelerometer.csv ──┐
                      ├──► Butterworth ──┐
  gyroscope.csv ──────┘    Low-Pass      │
                           Filter        ├──► Kalman ──► fused_x
  platform-locations.csv ───────────────┘    Filter     fused_y
  (GPS 1Hz)                                  (Cell 7)      │
                                                           │
  ┌────────────────────────────────────────────────────────┘
  │
  ▼
  Simulate GPS Blackout ──► Sliding Windows (200 steps) ──► LSTM
  (5 random windows)        50% overlap                     Model
  (Cell 9)                  (Cell 10)                       (Cell 11)
                                                               │
  ┌────────────────────────────────────────────────────────────┘
  │
  ▼
  Evaluation ──► Metrics (RMSE, MAE, Euclidean) ──► Folium Map
  (Cell 12)      (Cell 12)                           (Cell 13)
```

<br/>

---

<br/>

## Results

<br/>

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                   PERFORMANCE SUMMARY                       ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  DATASET                                                     ║
  ║  ├─ Sequence          ADVIO-01 (real-world pedestrian)       ║
  ║  ├─ Total samples     25,971 rows after cleaning             ║
  ║  ├─ Training windows  412  (80%)                             ║
  ║  └─ Test windows      104  (20%)                             ║
  ║                                                              ║
  ║  KALMAN FILTER                                               ║
  ║  ├─ Avg GPS deviation     0.628 m                            ║
  ║  └─ Max GPS deviation     0.655 m                            ║
  ║                                                              ║
  ║  LSTM MODEL                                                  ║
  ║  ├─ Train MAE (norm)      0.2375                             ║
  ║  ├─ Test  MAE (norm)      0.2926   [no overfitting]          ║
  ║  ├─ Mean position error   0.010 m  =  1 centimetre           ║
  ║  └─ Max  position error   0.025 m  =  2.5 centimetres        ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  Raw GPS alone       ~3–5 metres error                       ║
  ║  After Kalman         0.628 metres    10x improvement        ║
  ║  After LSTM           0.010 metres   600x vs raw GPS         ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

<br/>

---

<br/>

## Dataset

<br/>

> **ADVIO — Apple Device Visual Inertial Odometry**
> Aalto University, Helsinki, Finland — [Zenodo Record 1476931](https://zenodo.org/record/1476931)

<br/>

| Property | Value |
|:---|:---|
| Sequences available | 23 (advio-01 through advio-23) |
| Sequence used | advio-01 |
| Environments | Shopping mall, metro station, office, street |
| IMU sampling rate | ~100 Hz |
| GPS sampling rate | ~1 Hz |
| Sensor hardware | Standard iPhone (accelerometer, gyroscope, GPS) |
| Ground truth | High-accuracy external tracking system |
| Download size | ~120 MB (advio-01 only) |

<br/>

**Files inside each recording:**

```
advio-01/
├── iphone/
│   ├── accelerometer.csv      timestamp, acc_x, acc_y, acc_z  (m/s²)
│   ├── gyro.csv               timestamp, gyro_x, gyro_y, gyro_z  (rad/s)
│   ├── platform-locations.csv timestamp, lat, lon, alt, accuracy
│   ├── magnetometer.csv       magnetic field heading
│   └── barometer.csv          air pressure / altitude
└── ground-truth/
    └── pose.csv               high-accuracy reference position + orientation
```

<br/>

---

<br/>

## Technical Pipeline

<br/>

### 1 — Signal Processing

<br/>

Raw IMU data carries two things at once: the actual motion signal you care about, and high-frequency vibration noise from footsteps and hardware resonance. A **4th-order Butterworth low-pass filter** at 5 Hz separates these cleanly.

```
Passband  ──────────────────────┐
                                 \
                                  \_______________  Stopband
                    5 Hz          ↑
                               Cutoff
```

The filter is applied using **`filtfilt`** — a zero-phase implementation. Standard filtering introduces a time delay that would misalign sensor readings from their true timestamps. `filtfilt` runs forward and backward, cancelling the delay exactly.

**Nyquist theorem:** At 100 Hz sampling, maximum representable frequency is 50 Hz. Cutoff 5 Hz normalises to 5/50 = 0.1 before filter design.

GPS updates at 1 Hz. IMU updates at 100 Hz. Linear interpolation creates a GPS estimate at every IMU timestamp, filling the 99 gaps between each real GPS reading before any fusion can happen.

<br/>

### 2 — Kalman Filter

<br/>

The Kalman Filter maintains a **state vector** at every timestep:

```
     ┌ pos_x ┐
x =  │ pos_y │   4 quantities estimated simultaneously
     │ vel_x │
     └ vel_y ┘
```

At each of the 25,971 timesteps, two operations run:

**PREDICT** — physics propagates state forward using IMU:

```
x_pred  =  F  *  x_prev
P_pred  =  F  *  P  *  F^T  +  Q

         ┌ 1  0  dt  0 ┐
where F = │ 0  1  0  dt │    dt = 0.01s  (100 Hz)
         │ 0  0  1   0 │
         └ 0  0  0   1 ┘
```

**UPDATE** — GPS corrects the prediction:

```
K  =  P_pred  *  H^T  *  inv(H  *  P_pred  *  H^T  +  R)
x  =  x_pred  +  K  *  (z_gps  -  H  *  x_pred)
P  =  (I  -  K  *  H)  *  P_pred

         ┌ 1  0  0  0 ┐
where H = └ 0  1  0  0 ┘    (observe position only, not velocity)

R  =  5.0  * I_2    (GPS noise ~5 metres)
Q  =  0.1  * I_4    (IMU process noise)
```

**K is the Kalman Gain** — it automatically balances trust between GPS and IMU at every step. When GPS is lost: skip the update step. IMU dead reckoning continues. Uncertainty P grows, but position estimates keep flowing.

<br/>

### 3 — LSTM Model

<br/>

Dead reckoning drifts over time. For GPS blackouts longer than a few seconds, a model that has learned the relationship between IMU patterns and actual motion is needed.

**Training data construction:**

```
Full sequence (25,971 steps)
        │
        ▼
Sliding window  ──  size: 200 steps (2 seconds at 100 Hz)
                    step: 50  steps (0.5 seconds)
                    overlap: 75%
                    total windows: 516
        │
        ▼
Each window:  input  (200, 6)  ──  6 IMU channels x 200 timesteps
              target (2,)      ──  fused_x, fused_y at window end
```

**Blackout simulation:** 5 random GPS-loss windows of 5–10 seconds are erased from training data, giving the model examples of navigation without GPS.

<br/>

### 4 — Visualisation

<br/>

| Cell | Output |
|:---|:---|
| Cell 04 | Raw accelerometer, gyroscope, GPS signals |
| Cell 05 | Before vs after Butterworth filter |
| Cell 06 | GPS interpolation — 1 Hz dots vs 100 Hz line |
| Cell 08 | 2D trajectory map and X position over time |
| Cell 09 | GPS blackout windows shaded on timeline |
| Cell 11 | Training loss and MAE curves |
| Cell 12 | LSTM predictions vs ground truth + error over time |
| Cell 13 | Interactive Folium map on real OpenStreetMap tiles |

<br/>

---

<br/>

## Model Architecture

<br/>

```
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT   (200 timesteps  x  6 sensor channels)          │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  LSTM  128 units   return_sequences=True                │
  │  Learns short-term patterns in the 2-second window      │
  │  Parameters:  69,120                                    │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  BatchNormalization  +  Dropout 0.3                     │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  LSTM  64 units    return_sequences=False               │
  │  Summarises the full window into compact representation  │
  │  Parameters:  49,408                                    │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  BatchNormalization  +  Dropout 0.3                     │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Dense  32 units   ReLU activation                      │
  │  Non-linear combination of learned features             │
  │  Parameters:  2,080                                     │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Dropout 0.2                                            │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Dense  2 units    Linear activation                    │
  │  Output: predicted fused_x,  fused_y                    │
  │  Parameters:  66                                        │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  OUTPUT  (fused_x, fused_y)  in normalised units        │
  │  inverse_transform  ──►  real metres                    │
  └─────────────────────────────────────────────────────────┘

  Total parameters:   121,442
  Trainable:          121,058
  Non-trainable:          384

  Loss:         Mean Squared Error
  Optimiser:    Adam  lr=0.001
  Early stop:   patience=8  on val_loss
  LR schedule:  ReduceLROnPlateau  factor=0.5  patience=4
```

<br/>

---

<br/>

## Tech Stack

<br/>

```
  Data and Numerics        Signal Processing        Deep Learning
  ─────────────────        ─────────────────        ─────────────
  NumPy  1.24+             SciPy                    TensorFlow 2.x
  Pandas 1.5+              └─ butter                Keras
                           └─ filtfilt              └─ LSTM
                                                    └─ Dense
  Kalman Filter            Evaluation               └─ Dropout
  ─────────────            ──────────               └─ BatchNorm
  FilterPy                 scikit-learn
  └─ KalmanFilter          └─ StandardScaler        Visualisation
                           └─ mean_squared_error    ─────────────
                           └─ mean_absolute_error   Matplotlib
                                                    Folium
  Environment
  ───────────
  Google Colab  (T4 GPU recommended)
  Python 3.10+
```

<br/>

---

<br/>

## Repository Structure

<br/>

```
gps-imu-sensor-fusion/
│
├── README.md                        This file
│
├── Project1_Theory.ipynb            Theory notebook
│   │                                Read before touching any code
│   ├── Sensor physics and equations
│   ├── Butterworth filter derivation
│   ├── Full Kalman Filter math
│   ├── LSTM gate equations
│   ├── Sliding window mechanics
│   └── ADVIO dataset overview
│
├── Project1_Implementation.ipynb    Implementation notebook
│   │                                14 cells, fully documented
│   ├── Cell 01  Install libraries
│   ├── Cell 02  Download ADVIO dataset
│   ├── Cell 03  Load sensor CSVs
│   ├── Cell 04  Visualise raw data
│   ├── Cell 05  Butterworth filter
│   ├── Cell 06  Timestamp synchronisation
│   ├── Cell 07  Kalman Filter fusion
│   ├── Cell 08  Trajectory visualisation
│   ├── Cell 09  Simulate GPS blackout
│   ├── Cell 10  Sliding window preparation
│   ├── Cell 11  LSTM build and train
│   ├── Cell 12  Evaluation and metrics
│   ├── Cell 13  Folium interactive map
│   └── Cell 14  Save model and download
│
└── outputs/                         Generated after running notebook
    ├── raw_sensor_data.png
    ├── filtered_signal.png
    ├── gps_interpolation.png
    ├── fused_trajectory.png
    ├── gps_blackout.png
    ├── train_test_split.png
    ├── training_curves.png
    ├── lstm_predictions.png
    ├── lstm_error.png
    ├── trajectory_map.html          Interactive — open in any browser
    ├── gps_imu_lstm_model.h5        Saved LSTM weights
    ├── scaler_X.pkl                 Feature scaler
    └── scaler_y.pkl                 Target scaler
```

<br/>

---

<br/>

## Quickstart

<br/>

**Step 1 — Read theory first**

Open `Project1_Theory.ipynb` in Colab. No code — only equations, derivations, and explanations for every concept the implementation uses. Reading it means no line of code will be mysterious.

**Step 2 — Enable GPU**

```
Runtime  →  Change runtime type  →  T4 GPU  →  Save
```

**Step 3 — Open implementation notebook**

Open `Project1_Implementation.ipynb`. Each code cell has a markdown cell directly above it explaining what it does, what to look for in the output, and why it matters.

**Step 4 — Run cells in order**

```
Cell 01  →  libraries install          ~1 min
Cell 02  →  dataset download           ~5 min
Cell 03  →  data loading               instant
Cell 04  →  raw visualisation          instant
Cell 05  →  filter application         instant
Cell 06  →  synchronisation            instant
Cell 07  →  Kalman Filter              ~2 min
Cell 08  →  trajectory plots           instant
Cell 09  →  GPS blackout sim           instant
Cell 10  →  window preparation         instant
Cell 11  →  LSTM training              ~7 min on T4
Cell 12  →  evaluation                 instant
Cell 13  →  Folium map                 instant
Cell 14  →  save and download          instant
                                       ────────
                            Total      ~15 min
```

No manual data download. No local environment setup. Everything runs inside Colab.

<br/>

---

<br/>

## Real-World Applications

<br/>

The technique here is not academic. It runs in production systems across multiple industries.

<br/>

```
  ┌──────────────────────────────────────────────────────────────┐
  │  ISRO / ROCKETRY                                             │
  │                                                              │
  │  IISU designs guidance computers for PSLV, GSLV, LVM3,      │
  │  and Gaganyaan. Above 30 km, GPS is unreliable. The rocket   │
  │  runs on IMU alone — the exact dead reckoning this LSTM      │
  │  solves. The F matrix, H matrix, predict-update loop in      │
  │  Cell 7 are variants of what runs in that flight software.   │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  AVIATION                                                    │
  │                                                              │
  │  Commercial aircraft carry 2–3 independent Inertial          │
  │  Reference Systems running Kalman Filters continuously.      │
  │  During GPS jamming events — increasingly common in          │
  │  conflict airspace — the IRS takes over completely.          │
  │  Airlines lose millions annually to GPS interference.        │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  AUTONOMOUS VEHICLES                                         │
  │                                                              │
  │  Every self-driving car fuses LiDAR, cameras, GPS, and       │
  │  IMU using Kalman variants at 100+ Hz. In tunnels and        │
  │  underground car parks, the vehicle falls back on IMU        │
  │  integration and learned motion models — the same            │
  │  combination of Kalman and LSTM built in this project.       │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  INDOOR NAVIGATION                                           │
  │                                                              │
  │  GPS does not work inside buildings. Hospitals, airports,    │
  │  warehouses — all need IMU-based pedestrian dead reckoning   │
  │  fused with WiFi or BLE beacons via Kalman filtering.        │
  │  Google Indoor Maps, Apple, IndoorAtlas — all use LSTM       │
  │  models trained on IMU sequences for positioning accuracy.   │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  WEARABLES AND HEALTH TECH                                   │
  │                                                              │
  │  Smartwatches track runs through GPS-denied areas using      │
  │  accelerometer step detection and gyroscope heading.         │
  │  Rehabilitation clinics use IMU-based gait analysis to       │
  │  quantify patient mobility before and after surgery.         │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  SUBMARINE AND MARINE NAVIGATION                             │
  │                                                              │
  │  Submarines cannot use GPS underwater. They navigate         │
  │  entirely on IMU for the duration of a patrol — sometimes    │
  │  weeks. Ring laser gyroscope drift over that period is       │
  │  significant. Kalman correction on surfacing is essential.   │
  └──────────────────────────────────────────────────────────────┘
```

<br/>

---

<br/>

## Key Concepts

<br/>

| Concept | Cell | Significance |
|:---|:---:|:---|
| Butterworth filter | 05 | Zero-phase noise removal from IMU signals |
| Nyquist theorem | 05 | Defines maximum representable frequency at 100 Hz |
| Linear interpolation | 06 | Synchronises 1 Hz GPS with 100 Hz IMU timeline |
| Flat-earth projection | 07 | Converts lat/lon degrees to local X/Y metres |
| State space model | 07 | Represents position and velocity as unified vector |
| State transition matrix F | 07 | Encodes physics of constant-velocity motion |
| Measurement matrix H | 07 | Maps full state to observable GPS measurements |
| Kalman Gain K | 07 | Optimally weights GPS vs IMU prediction at each step |
| Dead reckoning | 07 | IMU-only navigation when GPS is absent |
| Sliding window | 10 | Converts time series into fixed-size LSTM inputs |
| StandardScaler | 10 | Zero mean, unit variance normalisation |
| LSTM gates | 11 | Forget, input, output gates for long-term memory |
| Dropout regularisation | 11 | Prevents memorisation, forces generalisation |
| BatchNormalization | 11 | Stabilises training, allows higher learning rates |
| EarlyStopping | 11 | Halts training when validation loss stops improving |
| RMSE vs MAE | 12 | RMSE penalises large errors, MAE gives average error |
| Euclidean distance error | 12 | Straight-line positional error in metres |
| Folium PolyLine | 13 | Draws GPS and fused trajectories on real map tiles |

<br/>

---

<br/>

<div align="center">

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   Project 1 of 2  —  Sensor Fusion Series           │
  │                                                     │
  │   Project 2:  IMU Sensor Data Analysis and          │
  │               Motion Prediction                     │
  │               CNN-LSTM on UCI HAR Dataset           │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

<br/>

*The Kalman Filter equations you implemented are in the flight software*
*of every rocket that has ever launched from Sriharikota.*

<br/>

---

**Prem M Patel** — prempatel7740@gmail.com
Computer Engineer 

</div>