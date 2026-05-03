import torch
import h5py
from datetime import datetime
import pandas as pd
import numpy as np
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def read_data(data_path):

    with h5py.File(data_path, 'r') as f:
        # read static_data
        static_group = f['static_data']
        station_ids = list(static_group.keys())
        static_features = []

        for loc in station_ids:
            features = static_group[loc][:]
            # data : Float
            sand_barrier, height, altitude, longitude, latitude = map(float, features)

            static_features.append({
                'station_id': loc,
                'sand_barrier': sand_barrier,
                'height': height,
                'altitude': altitude,
                'longitude': longitude,
                'latitude': latitude
            })

        static_df = pd.DataFrame(static_features)
        static_df.set_index('station_id', inplace=True)

        # read dynamic_data
        climate_group = f['dynamic_data']
        # all timestamps
        time_steps = sorted(list(climate_group.keys()))
        dynamic_features = ['temperature', 'humidity', 'pm10', 'ndvi']

        num_time_steps, num_locations, num_features = len(time_steps), len(station_ids), len(dynamic_features)

        dynamic_data = np.zeros((num_time_steps, num_locations, num_features))
        climate_timestamps = []
        datetime_format = '%Y-%m-%d %H:%M'

        # format time
        for t, time in enumerate(time_steps):
            tmp = datetime.strptime(time, datetime_format)
            climate_timestamps.append(tmp)

            for l, loc in enumerate(station_ids):
                dynamic_data[t, l, :] = climate_group[time][loc][:]


    return static_df, dynamic_data, dynamic_features, station_ids, climate_timestamps

# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)




