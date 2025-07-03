# src/predict.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error

from .config import (
    FEATURES, TARGET_FEATURE, RESULT_DIR
)


# --- NEW: Update the function signature to accept `predictions_save_path` ---
def make_prediction_and_plot(model, X_test, y_test, scaler, horizon, plot_save_path, predictions_save_path):

    print(f"--- 开始预测与评估: {horizon}天 ---")

    # 1. Set model to evaluation mode and make predictions (unchanged)
    model.eval()
    with torch.no_grad():
        predictions_scaled = model(X_test)

    predictions_numpy_scaled = predictions_scaled.cpu().numpy()
    y_test_numpy_scaled = y_test.cpu().numpy()

    # 2. Inverse transform ALL predictions and actuals (unchanged)
    def inverse_transform_batch(scaled_data, scaler_obj):
        data_reshaped = scaled_data.reshape(-1, 1)
        num_features = len(FEATURES)
        target_idx = FEATURES.index(TARGET_FEATURE)
        dummy_array = np.zeros((data_reshaped.shape[0], num_features))
        dummy_array[:, target_idx] = data_reshaped[:, 0]
        unscaled_array = scaler_obj.inverse_transform(dummy_array)
        return unscaled_array[:, target_idx]

    predictions_unscaled = inverse_transform_batch(predictions_numpy_scaled, scaler)
    y_test_unscaled = inverse_transform_batch(y_test_numpy_scaled, scaler)

    # 3. Calculate MSE and MAE on the UN-SCALED (restored) data (unchanged)
    mse = mean_squared_error(y_test_unscaled, predictions_unscaled)
    mae = mean_absolute_error(y_test_unscaled, predictions_unscaled)
    print(f"  - 评估结果 (on original-scale data):")
    print(f"    - 均方误差 (MSE): {mse:.4f}")
    print(f"    - 平均绝对误差 (MAE): {mae:.4f}")

    # 4. For plotting, we only need the first sequence from the unscaled arrays
    first_prediction_unscaled = predictions_unscaled[:horizon]
    first_actual_unscaled = y_test_unscaled[:horizon]

    # --- NEW: Save the prediction results to a CSV file ---
    try:
        # Ensure the 'predictions' directory exists
        predictions_dir = os.path.dirname(predictions_save_path)
        os.makedirs(predictions_dir, exist_ok=True)

        # Create a pandas DataFrame with actual and predicted values
        results_df = pd.DataFrame({
            'day': range(1, horizon + 1),
            'actual': first_actual_unscaled,
            'predicted': first_prediction_unscaled
        })
        # Save the DataFrame to the specified CSV file
        results_df.to_csv(predictions_save_path, index=False)
        print(f"预测结果已保存至: {predictions_save_path}")

    except Exception as e:
        print(f"Error saving prediction results: {e}")


    # 5. Plotting the first prediction vs the first actual sequence (unchanged)
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 8))

    plot_index = range(horizon)

    plt.plot(plot_index, first_actual_unscaled, label='Actual Future Power', color='blue', marker='.')
    plt.plot(plot_index, first_prediction_unscaled, label='Predicted Future Power', color='red', linestyle='--')
    plt.title(f'Power Consumption Prediction vs Actual ({horizon} Days) - MAE: {mae:.2f}', fontsize=16)
    plt.xlabel('Days into the Future', fontsize=12)
    plt.ylabel('Global Active Power', fontsize=12)
    plt.legend()
    plt.grid(True)

    os.makedirs(RESULT_DIR, exist_ok=True)
    plt.savefig(plot_save_path)
    print(f"结果图已保存至: {plot_save_path}")

    return mse, mae