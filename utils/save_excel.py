import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime, timedelta

def save_predictions_to_excel(model, test_loader, station_ids, config_name):
    """
    Save model predictions and ground-truth values to Excel files by station ID.
    """
    model.eval()
    device = next(model.parameters()).device
    scalers = test_loader.dataset.scalers
    feature_names = ['Temperature', 'RH', 'PM10']
    prediction_length = test_loader.dataset.prediction_length
    num_stations = len(station_ids)
    
    print(f"\n\033[34mSaving predictions...\033[0m")

    # Create results directory
    save_dir = f"./results/{config_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Create a dictionary for each station to store predictions and ground truth
    station_predictions = {station_id: {} for station_id in station_ids}
    
    batch_count = 0
    first_timestamp = None
    last_timestamp = None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            batch_count += 1
            batch = batch.to(device)
            
            # Get model output
            model_output = model(batch)
            if isinstance(model_output, tuple):
                output = model_output[0]
            else:
                output = model_output

            predictions = output.cpu().numpy()

            # Get ground truth
            ground_truth = batch.y.cpu().numpy()
            batch_size = 1 if not hasattr(batch, 'batch') else (batch.batch[-1].item() + 1)
            
            # Get timestamps
            if hasattr(batch, 'target_timestamps'):
                # Get timestamps for each sample
                target_timestamps = []
                timestamps_list = batch.target_timestamps
                if isinstance(timestamps_list[0], list):
                    target_timestamps = timestamps_list
                else:
                    for i in range(batch_size):
                        start_idx = i * prediction_length
                        end_idx = (i + 1) * prediction_length
                        batch_timestamps = []
                        for j in range(start_idx, end_idx):
                            if j < len(timestamps_list):
                                batch_timestamps.append(timestamps_list[j])
                        target_timestamps.append(batch_timestamps)
            else:
                print("\033[31mWarning: batch does not have target_timestamps attribute.\033[0m")
                continue

            predictions_denorm = np.zeros_like(predictions)
            ground_truth_denorm = np.zeros_like(ground_truth)
            
            for i in range(3):
                # Denormalize predictions
                predictions_denorm[:, i] = scalers[i].inverse_transform(
                    predictions[:, i].reshape(-1, 1)).ravel()
                
                # Denormalize ground truth
                ground_truth_denorm[:, i] = scalers[i].inverse_transform(
                    ground_truth[:, i].reshape(-1, 1)).ravel()
                
                # Validate denormalized data range
                if i == 0:  # temperature
                    predictions_denorm[:, i] = np.clip(predictions_denorm[:, i], -30, 40)
                    ground_truth_denorm[:, i] = np.clip(ground_truth_denorm[:, i], -30, 40)
                elif i == 1:  # RH
                    predictions_denorm[:, i] = np.clip(predictions_denorm[:, i], 0, 100)
                    ground_truth_denorm[:, i] = np.clip(ground_truth_denorm[:, i], 0, 100)
                elif i == 2:  # PM10
                    predictions_denorm[:, i] = np.clip(predictions_denorm[:, i], 0, 1000)
                    ground_truth_denorm[:, i] = np.clip(ground_truth_denorm[:, i], 0, 1000)
            

            try:
                predictions_reshaped = predictions_denorm.reshape(batch_size, prediction_length, num_stations, 3)
                ground_truth_reshaped = ground_truth_denorm.reshape(batch_size, prediction_length, num_stations, 3)
            except ValueError as e:
                print(f"\033[31mWarning: Batch {batch_idx} data cannot be reshaped, skipping. Error: {e}\033[0m")
                continue
            
            # Collect predictions and ground truth for each station
            for b in range(batch_size):
                current_timestamps = target_timestamps[b]  # Get current sample timestamps
                for s_idx, station_id in enumerate(station_ids):
                    for t_idx in range(prediction_length):
                        if t_idx < len(current_timestamps):
                            # Record first and last timestamps
                            current_timestamp = current_timestamps[t_idx]
                            if isinstance(current_timestamp, list):
                                current_timestamp = current_timestamp[0]  # If it's a list, take the first element
                                
                            if first_timestamp is None or current_timestamp < first_timestamp:
                                first_timestamp = current_timestamp
                            if last_timestamp is None or current_timestamp > last_timestamp:
                                last_timestamp = current_timestamp
                            
                            # Use timestamp string as key
                            if isinstance(current_timestamp, datetime):
                                timestamp_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
                            else:
                                # Convert to datetime if not already
                                timestamp_str = pd.to_datetime(current_timestamp).strftime('%Y-%m-%d %H:%M:%S')

                            # Store predictions and ground truth
                            if timestamp_str not in station_predictions[station_id]:
                                station_predictions[station_id][timestamp_str] = {
                                    'prediction': predictions_reshaped[b, t_idx, s_idx],
                                    'ground_truth': ground_truth_reshaped[b, t_idx, s_idx]
                                }
                            else:
                                # if timestamp exists, take average
                                prev_pred = station_predictions[station_id][timestamp_str]['prediction']
                                prev_gt = station_predictions[station_id][timestamp_str]['ground_truth']
                                
                                station_predictions[station_id][timestamp_str]['prediction'] = (prev_pred + predictions_reshaped[b, t_idx, s_idx]) / 2
                                station_predictions[station_id][timestamp_str]['ground_truth'] = (prev_gt + ground_truth_reshaped[b, t_idx, s_idx]) / 2

    print(f"Prediction time range: {first_timestamp} to {last_timestamp}.")
    
    # Save predictions and ground truth for each station to separate Excel files
    for station_id, predictions_dict in station_predictions.items():
        if not predictions_dict:
            print(f"\033[31mWarning: Station {station_id} has no predictions.\033[0m")
            continue
        
        # Create DataFrame
        df_data = []
        sorted_timestamps = sorted(predictions_dict.keys())
        
        # Organize data
        for timestamp_str in sorted_timestamps:
            data = predictions_dict[timestamp_str]
            pred = data['prediction']
            gt = data['ground_truth']
            
            row_data = {
                'Timestamp': pd.to_datetime(timestamp_str),  
            }
            
            # Add predictions and ground truth
            for j, feature in enumerate(feature_names):
                row_data[f'pre_{feature}'] = pred[j]
                row_data[f'true_{feature}'] = gt[j]
            
            df_data.append(row_data)
        
        # Create DataFrame and sort by timestamp
        df = pd.DataFrame(df_data)
        df = df.sort_values('Timestamp')
        
        # Excel file path
        excel_path = os.path.join(save_dir, f'{station_id}.xlsx')
        
        # Use ExcelWriter to save with formatting
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Predictions', index=False)
            
            worksheet = writer.sheets['Predictions']
            for idx, col in enumerate(df.columns):
                max_length = max(
                    df[col].astype(str).apply(len).max(),
                    len(col)
                )
                worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
        
        print(f"Station {station_id} saved to {excel_path}, {len(df)} rows.")

    
    print(f"\n\033[32mAll predictions and ground truth saved to {save_dir} directory.\033[0m")
    return save_dir
