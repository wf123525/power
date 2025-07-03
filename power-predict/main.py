# main.py

import os
import argparse
import numpy as np
import torch
import joblib
import json
from src.data_loader import load_data
from src.model import LSTMForecast, TransformerForecast, Autoformer
from src.train import train_model
from src.predict import make_prediction_and_plot
from src.config import *


def run_task(model_type, horizon, params, base_model_name, base_scaler_name, base_plot_name, run_num):
    """
    Executes a single run of training, evaluation, and saving results.
    """
    # 1. Create unique filenames for this specific run
    run_model_name = base_model_name.replace('.pt', f'_run{run_num}.pt')
    run_scaler_name = base_scaler_name.replace('.pkl', f'_{model_type}_run{run_num}.pkl')
    run_plot_name = f'{base_plot_name}_{model_type}_run{run_num}.png'
    # --- NEW: Add a filename for the prediction results CSV ---
    run_predictions_name = f'predictions_{model_type}_{horizon}d_run{run_num}.csv'

    model_path = os.path.join(MODEL_DIR, run_model_name)
    scaler_path = os.path.join(MODEL_DIR, run_scaler_name)
    plot_path = os.path.join(RESULT_DIR, run_plot_name)
    # --- NEW: Define the full path for the prediction results CSV ---
    predictions_path = os.path.join('predictions', run_predictions_name)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # 2. Load data (unchanged)
    X_train, y_train, X_test, y_test, scaler = load_data(
        train_path=TRAIN_FILE,
        test_path=TEST_FILE,
        sequence_length=SEQUENCE_LENGTH,
        horizon=horizon
    )
    joblib.dump(scaler, scaler_path)
    print(f"Data loaded. Scaler saved to {scaler_path}")

    # 3. Create the model instance (unchanged)
    if model_type == 'lstm':
        model = LSTMForecast(output_size=horizon, **params['model'])
    elif model_type == 'transformer':
        model = TransformerForecast(output_size=horizon, **params['model'])
    elif model_type == 'autoformer':
        model = Autoformer(output_size=horizon, **params['model'])
    else:
        raise ValueError("Unknown model type specified.")

    print(f"Initialized {model_type.upper()} model.")

    # 4. Train the model (unchanged)
    trained_model = train_model(
        model=model,
        X_train=X_train, y_train=y_train,
        X_val=X_test, y_val=y_test,
        horizon=horizon,
        model_save_path=model_path,
        params=params
    )

    # 5. Make predictions, plot, and save prediction data
    mse, mae = make_prediction_and_plot(
        model=trained_model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        horizon=horizon,
        plot_save_path=plot_path,
        # --- NEW: Pass the prediction save path to the function ---
        predictions_save_path=predictions_path
    )
    return mse, mae


def main():
    """
    Main function to run the forecasting experiment.
    (This function remains unchanged from the previous version)
    """
    parser = argparse.ArgumentParser(description="Use PyTorch models for power consumption forecasting.")
    parser.add_argument('--model', '-m', type=str, required=True, choices=['lstm', 'transformer', 'autoformer'],
                        help="Model to use for the experiment.")
    parser.add_argument('--horizon', '-H', type=int, required=True, choices=[90, 365], help="Forecast horizon in days.")
    args = parser.parse_args()

    # --- Parameter setup (unchanged) ---
    if args.horizon == 90:
        horizon = SHORT_TERM_HORIZON
        scaler_name = SHORT_TERM_SCALER_NAME
        plot_name = SHORT_TERM_PLOT_NAME
        if args.model == 'lstm':
            params = LSTM_SHORT_TERM_PARAMS
            model_name = LSTM_SHORT_TERM_MODEL_NAME
        elif args.model == 'transformer':
            params = TRANSFORMER_SHORT_TERM_PARAMS
            model_name = TRANSFORMER_SHORT_TERM_MODEL_NAME
        else: # autoformer
            params = AUTOFORMER_SHORT_TERM_PARAMS
            model_name = AUTOFORMER_SHORT_TERM_MODEL_NAME
    else:  # args.horizon == 365
        horizon = LONG_TERM_HORIZON
        scaler_name = LONG_TERM_SCALER_NAME
        plot_name = LONG_TERM_PLOT_NAME
        if args.model == 'lstm':
            params = LSTM_LONG_TERM_PARAMS
            model_name = LSTM_LONG_TERM_MODEL_NAME
        elif args.model == 'transformer':
            params = TRANSFORMER_LONG_TERM_PARAMS
            model_name = TRANSFORMER_LONG_TERM_MODEL_NAME
        else: # autoformer
            params = AUTOFORMER_LONG_TERM_PARAMS
            model_name = AUTOFORMER_LONG_TERM_MODEL_NAME

    mse_scores, mae_scores = [], []

    print("\n" + "=" * 60)
    print(f"Starting {NUM_RUNS} runs for {args.model.upper()} model, {args.horizon}-day forecast...")
    print("=" * 60)

    # --- Experiment loop (unchanged) ---
    for i in range(NUM_RUNS):
        print(f"\n--- Experiment Run {i + 1}/{NUM_RUNS} ---")
        mse, mae = run_task(
            model_type=args.model,
            horizon=horizon,
            params=params,
            base_model_name=model_name,
            base_scaler_name=scaler_name,
            base_plot_name=plot_name,
            run_num=i + 1
        )
        if mse is not None and mae is not None:
            mse_scores.append(mse)
            mae_scores.append(mae)

    # --- Summarize and save results (unchanged from previous version) ---
    if mse_scores and mae_scores:
        print("\n" + "=" * 60)
        print(f"Experiment Summary: {args.model.upper()} model, {horizon}-day forecast ({NUM_RUNS} runs average)")
        print("=" * 60)
        print(f"Mean Squared Error (MSE):")
        print(f"  - Average: {np.mean(mse_scores):.4f}")
        print(f"  - Std Dev: {np.std(mse_scores):.4f}")
        print(f"\nMean Absolute Error (MAE):")
        print(f"  - Average: {np.mean(mae_scores):.4f}")
        print(f"  - Std Dev: {np.std(mae_scores):.4f}")
        print("=" * 60)

        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)

        results_data = {
            "model_name": args.model,
            "forecast_horizon_days": horizon,
            "evaluation_metrics": {
                "mse": {
                    "average": np.mean(mse_scores),
                    "std_dev": np.std(mse_scores)
                },
                "mae": {
                    "average": np.mean(mae_scores),
                    "std_dev": np.std(mae_scores)
                }
            }
        }

        json_filename = f"{args.model}_{horizon}d_summary.json"
        json_filepath = os.path.join(output_dir, json_filename)

        with open(json_filepath, 'w') as f:
            json.dump(results_data, f, indent=4)

        print(f"\nSummary results have been saved to: {json_filepath}")
        print("=" * 60)


if __name__ == '__main__':
    main()