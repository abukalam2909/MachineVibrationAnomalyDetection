# Machine Vibration Anomaly Detection using Autoencoders

## Overview
This project detects anomalies in machine vibration data using a deep learning **autoencoder** model. The idea is simple:  
- **Train the autoencoder only on normal (non-anomalous) vibration signals.**  
- The model learns to compress and reconstruct these normal patterns.  
- If a new signal cannot be reconstructed well (high reconstruction error), it is flagged as an anomaly.

This approach is widely used in **predictive maintenance** to identify early signs of machine failure.

---

## Dataset
- **Source:** Mounted in Google Drive (e.g., `/content/drive/My Drive/Vibration_Data_Anomaly_Detection`)
- **Shape:**  
  - **Samples:** 1702  
  - **Time Points per Sample:** 20,000  
  - **Axes:** 3 (e.g., X, Y, Z vibration components)
- **Labels:** 1 = Normal, 0 = Anomaly

---

## Workflow

### Data Preparation
1. **Downsampling**  
   - The original signals (20,000 points) are downsampled by a factor of 10 → **2000 points** per sample.  
   - This reduces computation while preserving key vibration patterns.  
   ```python
   downsample_factor = 10
   new_shape = data.shape[0], data.shape[1] // downsample_factor, downsample_factor, data.shape[2]
   data_sampled = np.mean(data.reshape(new_shape), axis=2)
   ```

2. **Normalization**  
   - **Min-Max scaling** is applied per sample to scale values to [0, 1].  
   - Ensures all features have equal importance.  
   ```python
   min_values = np.min(data, axis=1, keepdims=True)
   max_values = np.max(data, axis=1, keepdims=True)
   scaled_data = (data - min_values) / (max_values - min_values + 1e-11)
   ```

3. **Flattening**  
   - After normalization, the three axes are flattened into one vector.

4. **Train-Validation Split**  
   - 80% training, 20% validation  
   - Stratified split to preserve class balance.

---

### Model Architecture
A **fully connected autoencoder** built using TensorFlow/Keras:

- **Encoder**  
  - Dense(128) → Dense(64) → Dense(32) with ReLU activations
- **Decoder**  
  - Dense(32) → Dense(64) → Dense(128) → Dense(original_input_dim) with sigmoid activation

```python
autoencoder.compile(optimizer=Adam(learning_rate=1e-4), loss='mae')
```

---

### Training
- **Loss Function:** Mean Absolute Error (MAE)  
- **Training Data:** Only **normal** samples  
- **Callbacks:**  
  - `EarlyStopping` – Stop if no improvement for 8 epochs  
  - `ReduceLROnPlateau` – Reduce LR when stuck  
  - `ModelCheckpoint` – Save the best model

---

### Threshold Calculation
- After training, the **reconstruction error** is computed for validation data.
- Threshold is determined using **Interquartile Range (IQR)**:
  ```python
  IQR = Q3 - Q1
  lower_threshold = Q1 - 1.5 * IQR
  upper_threshold = Q3 + 1.5 * IQR
  ```
- Samples outside this range are classified as anomalies.

---

### Evaluation
- Predictions are compared with true labels using **precision, recall, and F1-score**:
  ```python
  print(classification_report(y_valid, predicted_labels))
  ```
- Histogram of reconstruction errors shows threshold line for anomaly detection.

---

## Results
- **Model**: Autoencoder with 3 encoding and 4 decoding layers
- **Loss Trend:** Both training and validation loss decreased steadily, indicating good generalization.
- **Performance Metrics:** Provided via `classification_report`

---

## How to Run
1. **Clone repo & mount dataset in Colab**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   dataset_path = "/content/drive/My Drive/Vibration_Data_Anomaly_Detection"
   ```

2. **Run all preprocessing, model training, and evaluation cells in `Machine_Vibration_Anomaly_Detection.ipynb`.**

3. **Adjust hyperparameters** (`EPOCHS`, `BATCH_SIZE`, `learning_rate`) as needed.

---

## Future Improvements
- Implement **Conv1D Autoencoder** for better time-series feature extraction
- Use **sequence-to-sequence models (LSTM Autoencoders)** for long-term dependencies
- Deploy with **FastAPI** for real-time inference
