import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from typing import Tuple, List
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import os
from sklearn.preprocessing import MinMaxScaler


class STGraphDataset(Dataset):
    """Spatio-temporal graph dataset for processing graph-structured time-series data"""
    
    def __init__(self,
                 static_features: pd.DataFrame,
                 dynamic_features: np.ndarray,
                 station_ids: List[str],
                 window_size: int = 60, 
                 prediction_length: int = 10,
                 stride: int = 1,
                 distance_threshold: float = 1.0, # 1.0 km == 1000 m
                 normalize: bool = True,
                 timestamps: List[pd.Timestamp] = None,
                 is_train: bool = True,
                 scalers: List = None,
                 indices: List[int] = None,
                 ablation: str = None):
        """
        Initialize the spatio-temporal graph dataset:
        Args:
            static_features: DataFrame of static features, including sand_barrier and spatial features.
            dynamic_features: ~
            station_ids: List of station IDs.
            window_size: Size of the input time window.
            stride: Sliding-window stride. 
            distance_threshold: Distance threshold for constructing spatial edges, in kilometers.
            normalize: Whether to normalize dynamic features.
            timestamps
        """
        self.station_ids = station_ids
        self.window_size = window_size
        self.prediction_length = prediction_length
        self.stride = 1
        self.distance_threshold = distance_threshold
        self.num_stations = len(station_ids)
        self.ablation = ablation

        if indices is not None:
            dynamic_features = dynamic_features[indices]
            if timestamps is not None:
                timestamps = [timestamps[i] for i in indices]

        # 1. One-hot encoding of sand_barrier.
        sand_barrier = static_features['sand_barrier'].values
        self.num_barrier_types = len(np.unique(sand_barrier))
        sand_barrier_onehot = np.eye(self.num_barrier_types)[sand_barrier.astype(int) - 1]

        # 2. Spatial features.
        spatial_features = static_features[['height', 'altitude', 'longitude', 'latitude']].values
        self.sand_barrier_features = torch.FloatTensor(sand_barrier_onehot)
        self.spatial_features = torch.FloatTensor(spatial_features)

        # Adjust timestamps to full-hour values.
        if timestamps is not None:
            assert len(timestamps) == dynamic_features.shape[0], \
                f"Timestamps length ({len(timestamps)}) doesn't match dynamic features length ({dynamic_features.shape[0]})"
            self.timestamps = timestamps

        else:
            print("\n\033[31mWarning: No timestamp info provided, using default timestamp.\033[0m")
            start_time = pd.Timestamp.now().floor('H')
            self.timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(dynamic_features.shape[0])]

        # Normalization.
        if normalize:
            if is_train:
                self.scalers = []
                normalized_features = np.zeros_like(dynamic_features)

                for i in range(3):
                    feature_data = dynamic_features[..., i].reshape(-1, 1)
                    scaler = StandardScaler()
                    scaler.fit(feature_data)
                    normalized_data = scaler.transform(feature_data)
                    normalized_features[..., i] = normalized_data.reshape(dynamic_features.shape[0], -1)
                    self.scalers.append(scaler)
                
                ndvi_scaler = MinMaxScaler()
                ndvi_data = dynamic_features[..., 3].reshape(-1, 1)
                ndvi_scaler.fit(ndvi_data)
                normalized_ndvi = ndvi_scaler.transform(ndvi_data)
                normalized_features[..., 3] = normalized_ndvi.reshape(dynamic_features.shape[0], -1)
                self.scalers.append(ndvi_scaler)
                
            else:
                assert scalers is not None, "Scalers fitted on the training set must be provided."
                self.scalers = scalers
                normalized_features = np.zeros_like(dynamic_features)
                
                for i in range(3):
                    feature_data = dynamic_features[..., i].reshape(-1, 1)
                    normalized_data = self.scalers[i].transform(feature_data)
                    normalized_features[..., i] = normalized_data.reshape(dynamic_features.shape[0], -1)
                
                ndvi_data = dynamic_features[..., 3].reshape(-1, 1)
                normalized_ndvi = self.scalers[3].transform(ndvi_data)
                normalized_features[..., 3] = normalized_ndvi.reshape(dynamic_features.shape[0], -1)
        
        self.dynamic_features = normalized_features if normalize else dynamic_features

        if timestamps is not None:
            self.timestamps = timestamps
            print("\n=== Timestamp information ===")
            print(f"Start time: {self.timestamps[0]}.")
            print(f"End time: {self.timestamps[-1]}.")
            print(f"Length of timestamp: {len(self.timestamps)}.")
        else:
            print("\n\033[31mWarning: No timestamp info provided, using default timestamp.\033[0m")
            start_time = pd.Timestamp.now().floor('H')
            self.timestamps = [start_time + pd.Timedelta(minutes=i) for i in range(len(dynamic_features))]

        """
        Calculate the number of valid time windows while ensuring that
            the target window does not exceed the data range.
        """
        total_steps = len(self.dynamic_features)
        max_start = total_steps - self.window_size - self.prediction_length
        self.num_windows = (max_start // self.stride + 1) if max_start >= 0 else 0
        # Ablation of static environmental features.
        # Barrier ablation
        if self.ablation in ('Barrier', 'AllEnv'):
            self.sand_barrier_features = torch.zeros_like(self.sand_barrier_features)

        # Elevation ablation: elevation feature

        if self.ablation in ('Elev', 'AllEnv'):
            self.spatial_features[:, 1] = 0

        # Coordinate ablation: longitude and latitude features
        if self.ablation in ('Coord', 'AllEnv'):
            self.spatial_features[:, 2:4] = 0

        self.edge_index, self.edge_attr = self._compute_spatial_edges()
        # For the AllEnv setting, set all edge features to zero.
        if self.ablation == 'AllEnv':
            self.edge_attr = torch.zeros_like(self.edge_attr)

        self.time_encodings = self._compute_time_encodings()


    def _compute_time_encodings(self) -> np.ndarray:
        """Compute time encodings."""
        time_steps = np.arange(len(self.dynamic_features))
        
        day_in_week = (time_steps % 7) / 7.0
        hour_in_day = (time_steps % 24) / 24.0
        
        # Use sine and cosine encodings.
        encodings = np.stack([
            np.sin(2 * np.pi * day_in_week),
            np.cos(2 * np.pi * day_in_week),
            np.sin(2 * np.pi * hour_in_day),
            np.cos(2 * np.pi * hour_in_day),
            np.sin(4 * np.pi * hour_in_day),
            np.cos(4 * np.pi * hour_in_day)
        ], axis=1)
        
        return encodings
    
    def _compute_spatial_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute spatial edges and edge features between stations."""
        edges = []
        edge_attrs = []
        
        try:
            for i in range(self.num_stations):
                for j in range(i + 1, self.num_stations):
                    station1_spatial = self.spatial_features[i]
                    station2_spatial = self.spatial_features[j]

                    # Calculate horizontal distance using longitude and latitude.
                    distance = self._calculate_distance(
                        float(station1_spatial[2]), float(station1_spatial[3]),
                        float(station2_spatial[2]), float(station2_spatial[3])
                    )
                    
                    if distance <= self.distance_threshold:
                        # Construct edge features.
                        edge_attr = [
                            distance,
                            float(station1_spatial[1] - station2_spatial[1]),
                            float(station1_spatial[0]),
                            float(station2_spatial[0]),
                            float(station1_spatial[1]),
                            float(station2_spatial[1])
                        ]
                        
                        edges.extend([[i, j], [j, i]])
                        edge_attrs.extend([edge_attr, edge_attr])

            # If no edges are created, add self-loops.
            if not edges:
                print("No edges created, adding self-loops")
                for i in range(self.num_stations):
                    edges.append([i, i])
                    station_spatial = self.spatial_features[i]
                    edge_attr = [
                        0.0,
                        0.0,
                        float(station_spatial[0]),
                        float(station_spatial[0]),
                        float(station_spatial[1]),
                        float(station_spatial[1])
                    ]
                    edge_attrs.append(edge_attr)

            # Convert to NumPy arrays first, and then convert to tensors.
            edges_np = np.array(edges, dtype=np.int64)
            edge_attrs_np = np.array(edge_attrs, dtype=np.float32)
            
            edge_index = torch.from_numpy(edges_np).t()
            edge_attr = torch.from_numpy(edge_attrs_np)

            return edge_index, edge_attr

        except Exception as e:
            print(f"Error in _compute_spatial_edges:")
            print(f"Error message: {str(e)}")
            print(f"Number of stations: {self.num_stations}")
            print(f"Spatial features shape: {self.spatial_features.shape}")
            raise e
    
    @staticmethod
    def _calculate_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """Calculate the distance between two longitude-latitude coordinates, in kilometers."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def __len__(self) -> int:
        return self.num_windows
    
    def __getitem__(self, idx: int) -> Data:
        """Get the data sample at the specified index."""

        # Calculate the start and end indices of the time window.
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        target_start_idx = end_idx
        target_end_idx = target_start_idx + self.prediction_length

        # Add safety checks.
        if idx > 0:
            prev_target_end = (start_idx - self.stride) + self.window_size + self.prediction_length
            expected_target_start = prev_target_end - self.prediction_length + self.stride
            if target_start_idx != expected_target_start:
                print(f"\033[31mWarning: Prediction window shift anomaly detected at index {idx}.\033[0m")
                print(f"\033[33mThe current prediction window is expected to begin at: {expected_target_start}.\033[0m")
                print(f"\033[33mThe actual current prediction window starts from: {target_start_idx}.\033[0m")

        # Validate timestamp continuity.
        if hasattr(self, 'timestamps'):
            input_timestamps = self.timestamps[start_idx:end_idx]
            target_timestamps = self.timestamps[target_start_idx:target_end_idx]

            # Check the continuity between input and target timestamps.
            if len(input_timestamps) > 0 and len(target_timestamps) > 0:
                time_diff = (target_timestamps[0] - input_timestamps[-1]).total_seconds() / 60
                if time_diff != 1:
                    print(f"\033[31mWarning: Time discontinuity! Interval is {time_diff} minutes.\033[0m")
                    print(f"\033[33mEnter end time:{input_timestamps[-1]}.\033[0m")
                    print(f"\033[33mTarget start time: {target_timestamps[0]}.\033[0m")

        # Validate the index range.
        if target_end_idx > len(self.dynamic_features):
            raise IndexError(f"\033[33mIndex{idx} out of range, Index of end {target_end_idx} over the length of data {len(self.dynamic_features)}\033[0m")


        input_seq = self.dynamic_features[start_idx:end_idx]
        target_seq = self.dynamic_features[target_start_idx:target_end_idx]

        input_timestamps = self.timestamps[start_idx:end_idx]
        target_timestamps = self.timestamps[target_start_idx:target_end_idx]

        if len(input_timestamps) > 0 and len(target_timestamps) > 0:
            time_diff = (target_timestamps[0] - input_timestamps[-1]).total_seconds() / 60
            if time_diff != 1:
                print(f"\033[31mWarning: Time discontinuity! Interval is {time_diff} minutes.\033[0m")
                print(f"\033[33mInput end time: {input_timestamps[-1]}.\033[0m")
                print(f"\033[33mTarget start time: {target_timestamps[0]}.\033[0m")

        # 1. Process dynamic features.
        x = torch.FloatTensor(input_seq)
        
        # 2. Get time encodings.
        time_encoding = torch.FloatTensor(self.time_encodings[start_idx:end_idx])
        time_encoding = time_encoding.unsqueeze(1).expand(-1, self.num_stations, -1)
        
        # 3. Expand sand-barrier features along the temporal dimension.
        sand_barrier = self.sand_barrier_features.unsqueeze(0).expand(self.window_size, -1, -1)

        # 4. Reshape all features.
        x = x.reshape(self.window_size * self.num_stations, -1)
        time_encoding = time_encoding.reshape(self.window_size * self.num_stations, -1)
        sand_barrier = sand_barrier.reshape(self.window_size * self.num_stations, -1)

        # —— Ablation of dynamic environmental features. —— #
        if self.ablation in ('NDVI', 'AllEnv'):
            x[:, 3:4] = 0

        # 5. Concatenate features.
        x = torch.cat([x, sand_barrier, time_encoding], dim=1)

        # 6. Process the target sequence.
        target_seq = target_seq[:, :, :3]
        target = torch.FloatTensor(target_seq)
        # target = target[-1]
        target = target.reshape(-1, 3)


        data = Data(
            x=x,
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=target,
            num_nodes=x.size(0),
            num_stations=self.num_stations,
            time_steps=self.window_size,
            prediction_length=self.prediction_length,
            input_timestamps=input_timestamps,
            target_timestamps=target_timestamps
        )

        return data
