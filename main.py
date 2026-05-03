import torch
import os
from utils.utils import  read_data, set_seed
from data_utils.dataset import STGraphDataset
from utils.save_excel import save_predictions_to_excel
import argparse
import yaml
from models.model import STGCN
from torch_geometric.loader import DataLoader as PyGDataLoader
import numpy as np
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime
import pandas as pd

# Hyperparameter config file, data file, training data save path
config_path = 'config/config-10.yaml'
data_path = './data/data.h5'
train_record = './log-10.txt'

# Create directory to save training data and predictions
config_name = os.path.splitext(os.path.basename(config_path))[0]
os.makedirs(f'./results/{config_name}', exist_ok=True)


# Set up argparse to parse command line arguments
def get_parameters(config_path):

    with open(config_path, 'r') as config_path:
        configs = yaml.safe_load(config_path)

    parser = argparse.ArgumentParser(description="Run neural network with different hyperparameters")
    # Argument definition
    parser.add_argument('--batch_size', type=int, default=configs['batch_size'], help='batch_size')
    parser.add_argument('--epochs', type=int, default=configs['epochs'], help='epochs')
    parser.add_argument('--lr', type=float, default=configs['lr'], help='lr')
    parser.add_argument('--lr_min', type=int, default=configs['lr_min'], help='lr_min')
    parser.add_argument('--window_size', type=int, default=configs['window_size'], help='window size')
    parser.add_argument('--prediction_length', type=int, default=configs['prediction_length'], help='prediction length')
    parser.add_argument('--stride', type= int, default=configs['stride'], help='stride')
    parser.add_argument('--dropout', type=float, default=configs['dropout'], help='dropout')
    parser.add_argument('--seed', type=int, default=configs['seed'], help='random seed')
    parser.add_argument('--hidden_dim', type=int, default=configs['hidden_dim'], help='hidden_dim')
    parser.add_argument('--train_ratio', type=float, default=configs['train_ratio'], help='train_ratio')
    parser.add_argument('--val_ratio', type=float, default=configs['val_ratio'], help='val_ratio')
    parser.add_argument('--test_ratio', type=float, default=configs['test_ratio'], help='test_ratio')
    parser.add_argument('--weight_decay', type=float, default=configs['weight_decay'], help='weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=configs['warmup_epochs'], help='warmup_epochs')
    parser.add_argument('--patience', type=int, default=configs['patience'], help='patience')
    parser.add_argument('--gradient_clip', type=int, default=configs['gradient_clip'], help='gradient clip')
    parser.add_argument('--num_workers', type=int, default=configs['num_workers'], help='gradient clip')
    parser.add_argument('--temp_loss_weight', type=float, default=configs['temp_loss_weight'], help='temp_loss_weight')
    parser.add_argument('--use_dynamic_weights', type=bool, default=configs['use_dynamic_weights'], help='use_dynamic_weights')
    parser.add_argument('--weight_update_interval', type=int, default=configs['weight_update_interval'], help='weight_update_interval')
    args = parser.parse_args()

    return args

def get_dataset_indices(climate_timestamps, train_ratio, val_ratio):
    # Calculate dataset split indices
    total_size = len(climate_timestamps)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)

    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))

    return train_indices, val_indices, test_indices

def create_dataset(static_df, dynamic_df, station_ids,
                window_size, prediction_length, stride,
                climate_timestamps,
                train_indices, val_indices, test_indices,):
    train_dataset = STGraphDataset(
        static_features=static_df,
        dynamic_features=dynamic_df,
        station_ids=station_ids,
        window_size=window_size,
        prediction_length=prediction_length,
        stride=stride,
        timestamps=climate_timestamps,
        normalize=True,
        is_train=True,
        indices=train_indices
    )

    val_dataset = STGraphDataset(
        static_features=static_df,
        dynamic_features=dynamic_df,
        station_ids=station_ids,
        window_size=window_size,
        prediction_length=prediction_length,
        stride=stride,
        timestamps=climate_timestamps,
        normalize=True,
        is_train=False,
        scalers=train_dataset.scalers,
        indices=val_indices
    )

    # Create test set
    test_dataset = STGraphDataset(
        static_features=static_df,
        dynamic_features=dynamic_df,
        station_ids=station_ids,
        window_size=window_size,
        prediction_length=prediction_length,
        stride=stride,
        timestamps=climate_timestamps,
        normalize=True,
        is_train=False,
        scalers=train_dataset.scalers,
        indices=test_indices
    )
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, test_dataset, num_workers, batch_size):
    train_loader = PyGDataLoader(
        train_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )

    val_loader = PyGDataLoader(
        val_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory=True,
        persistent_workers=True if  num_workers > 0 else False,
        prefetch_factor=2
    )

    test_loader = PyGDataLoader(
        test_dataset,
        batch_size= batch_size,
        num_workers= num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2
    )

    return train_loader, val_loader, test_loader


def print_loader_timestamps(loader, name, max_batches=2):
    """
    Print sample timestamps of data loader
    """
    print(f"\n{name}Loader Sample Timestamps:")
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= max_batches:
            break

        try:
            if hasattr(batch, 'input_timestamps'):
                print(f"\nBatch {batch_idx + 1}:")
                # Get batch size
                batch_size = len(batch.input_timestamps) if isinstance(batch.input_timestamps[0], list) else 1

                for i in range(min(1, batch_size)):  # Only print first sample
                    # Get single sample timestamps
                    if isinstance(batch.input_timestamps[0], list):
                        input_first = batch.input_timestamps[i][0]
                        input_last = batch.input_timestamps[i][-1]
                        target_first = batch.target_timestamps[i][0]
                        target_last = batch.target_timestamps[i][-1]
                    else:
                        start_idx = i * batch.prediction_length
                        end_idx = (i + 1) * batch.prediction_length
                        input_first = batch.input_timestamps[start_idx]
                        input_last = batch.input_timestamps[min(end_idx - 1, len(batch.input_timestamps) - 1)]
                        target_first = batch.target_timestamps[start_idx]
                        target_last = batch.target_timestamps[min(end_idx - 1, len(batch.target_timestamps) - 1)]

                    time_diff_input = (pd.to_datetime(input_last) - pd.to_datetime(input_first)).total_seconds() / 3600
                    time_diff_target = (pd.to_datetime(target_last) - pd.to_datetime(
                        target_first)).total_seconds() / 3600

                    print(f"Sample {i + 1}:")
                    print(f"Input time window: {input_first} to {input_last} (~{time_diff_input:.1f} hour(s))")
                    print(f"Prediction time window: {target_first} to {target_last} (~{time_diff_target:.1f} hour(s))")

        except Exception as e:
            print(f"\033[33mBatch {batch_idx + 1} timestamp acquisition failed: {str(e)}\033[0m")

def main():
    # Read hyperparameters
    args = get_parameters(config_path)
    # Set random seed
    set_seed(args.seed)

    try:
        # Read data
        static_df, dynamic_df, features, station_ids, climate_timestamps = read_data(data_path)
        print("\n=== Raw Data Time Range ===")
        print(f"Start time: {climate_timestamps[0]}")
        print(f"End time: {climate_timestamps[-1]}")
        print(f"Length of timestamp: {len(climate_timestamps)}")

        train_indices, val_indices, test_indices = get_dataset_indices(climate_timestamps, args.train_ratio, args.val_ratio)

        print("\n=== Dataset Split Info ===")
        print(f"Training set index range: {train_indices[0]} - {train_indices[-1]}")
        print(f"Validation set index range: {val_indices[0]} - {val_indices[-1]}")
        print(f"Test set index range: {test_indices[0]} - {test_indices[-1]}")


        train_dataset, val_dataset, test_dataset = create_dataset(
            static_df,
            dynamic_df,
            station_ids,
            args.window_size,
            args.prediction_length,
            args.stride,
            climate_timestamps,
            train_indices,
            val_indices,
            test_indices
        )

        train_loader, val_loader, test_loader = create_data_loaders(
            train_dataset,
            val_dataset,
            test_dataset,
            args.num_workers,
            args.batch_size
        )

        # Calculate input feature dimension (including time encoding)
        sample_data = train_dataset[0]
        input_dim = sample_data.x.size(-1)

        # Initialize model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {device}")

        model = STGCN(
            node_feature_dim=input_dim,
            hidden_dim=args.hidden_dim,
            output_dim=3,
            num_stations=6,
            dropout=args.dropout,
            window_size=args.window_size,
            prediction_length=args.prediction_length,
            temp_loss_weight=args.temp_loss_weight
        ).to(device)

        # Define optimizer and loss function
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # Define learning rate scheduler
        def get_lr_multiplier(epoch):
            if epoch < args.warmup_epochs:
                # Linear warmup
                return epoch / args.warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return max(args.lr_min / args.lr, cosine_decay)

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=get_lr_multiplier
        )
        
        criterion = nn.MSELoss()
        
        # Add dimension check before training loop
        for batch in train_loader:
            batch = batch.to(device)
            output = model(batch)
            # print(f"Model output shape: {len(output)}")
            break
        
        # Training loop
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        feature_errors_history = []  # Record feature error history
        
        print("\nStarting training...")
        grad_scaler = GradScaler()
        for epoch in range(args.epochs):
            model.train()
            train_loss = 0
            epoch_feature_errors = torch.zeros(3).to(device)

            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()

                with autocast():
                    output, loss, feature_errors, _ = model(batch)
                    epoch_feature_errors += feature_errors

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            epoch_feature_errors /= len(train_loader)
            train_losses.append(train_loss)
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_feature_errors = torch.zeros(3).to(device)
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    batch = batch.to(device)
                    output, loss, feature_errors, feature_relative_errors = model(batch)
                    val_loss += loss.item()
                    val_feature_errors += feature_errors

                val_loss /= len(val_loader)
                val_feature_errors /= len(val_loader)
                val_losses.append(val_loss)

                # Update feature weights
                if args.use_dynamic_weights and epoch % args.weight_update_interval == 0:
                    model.update_feature_weights(val_feature_errors)

            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            scheduler.step()

            # Print loss
            print(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '\
                    f'Epoch {epoch + 1}/{args.epochs}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}.')
            # Save loss record
            with open(train_record, 'a') as f:
                f.write(f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")} '\
                       f'Epoch {epoch + 1}/{args.epochs}, '
                       f'Train Loss: {train_loss:.4f}, '
                       f'Val Loss: {val_loss:.4f}.\n')

            # Save current best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'feature_weights': model.feature_weights.data,
                }, f'./results/{config_name}/best_model.pth')
        
        # Load best model for testing
        checkpoint = torch.load(f'./results/{config_name}/best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])

        
        # Save predictions
        save_predictions_to_excel(model, test_loader, station_ids, config_name)
        
    except Exception as e:
        print(f"\033[33mError during training: {str(e)}\033[0m")
        print(f"\033[33mAdditional debug info: \033[0m")

        if 'batch' in locals():
            print(f"\033[33mBatch info:\033[0m")
            print(f"\033[33mx shape: {batch.x.shape}\033[0m")
            print(f"\033[33my shape: {batch.y.shape}\033[0m")
            print(f"\033[33mnum_stations: {batch.num_stations}\033[0m")
            print(f"\033[33mtime_steps: {batch.time_steps}\033[0m")
        raise e

if __name__ == '__main__':
    main()
