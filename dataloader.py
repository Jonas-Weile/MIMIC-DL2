## General imports
import os
import pickle
import torch
import numpy as np
from typing import Any, Tuple
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

## MIMIC-III Benchmarks Imports
from mimic3models import common_utils
from mimic3benchmark.readers import InHospitalMortalityReader


CACHED_DIR = os.path.join("data/cached")


def read_and_extract_features(reader):
    ret = common_utils.read_chunk(reader, reader.get_number_of_examples())
    period, features = 'all', 'all'

    X = common_utils.extract_features_from_rawdata(ret['X'], ret['header'], period, features)
    return X, ret['y'], ret['name']


def get_cached_filename(input: bool, mode: str) -> str:
    if input:
        filename = os.path.join(CACHED_DIR, f"{mode}_X_cached.pt")
    else:
        filename = os.path.join(CACHED_DIR, f"{mode}_y_cached.pt")
    return filename

        
def cached_data_exists(mode: str) -> bool:
    X_cached = get_cached_filename(True, mode)
    y_cached = get_cached_filename(False, mode)
    return os.path.exists(X_cached) and os.path.exists(y_cached)


class MIMIC3(Dataset):

    def __init__(
            self,
            mode: str,
            scaler: StandardScaler = None,
            cache: bool = True,
    ) -> None:
        self.transform = transforms.Compose([transforms.ToTensor()])
        
        assert mode in ['train', 'val', 'test'], 'Dataset must be either train, val or test!'
        if mode in ['train', 'val']:
            folder = 'train'
        else:
            folder = 'test'    

        # Check if we should use cached data
        if cache and cached_data_exists(mode):
            X, y = self.get_cached_data(mode)

        else:
            X, y = self.prepare_data(mode, folder, scaler, cache)

        # small fix - make sure the data always contains an even number of data points
        if len(X) % 2 > 0:
            X = X[:-1]
            y = y[:-1]

        self.data = (X, y)


    def get_cached_data(self, mode: str):
        X_cached = get_cached_filename(True, mode)
        y_cached = get_cached_filename(False, mode)
        X = torch.load(X_cached)
        y = torch.load(y_cached)

        
        # if we are training, load the cached scaler as well
        if mode == 'train':
            scaler_cached = os.path.join(CACHED_DIR, 'scaler_cached')
            with open(scaler_cached, 'rb') as f:
                self.scaler = pickle.load(f)
                
        return (X, y)


    def get_scaler(self):
        return self.scaler


    def prepare_data(self, mode: str,  folder: str, scaler: StandardScaler, cache: bool):
        reader = InHospitalMortalityReader(dataset_dir=os.path.join(f"data/in-hospital-mortality/{folder}"),
                                                listfile=os.path.join(f"data/in-hospital-mortality/{mode}_listfile.csv"),
                                                period_length=48.0)

        (X, y, names) = read_and_extract_features(reader)

        # Replace NaN values with the mean of that feature
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0, copy=True)
        imputer.fit(X)

        X = np.array(imputer.transform(X), dtype=np.float32)

        # Normalize data to have zero mean and unit variance
        if not scaler: # If we received a scaler, use this one
            scaler = StandardScaler()
            scaler.fit(X)
            self.scaler = scaler

        X = scaler.transform(X)

        # Convert to Tensor
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)
        y = torch.reshape(y, (len(y), 1))
        
        # Save the data
        if cache:
            if not os.path.exists(CACHED_DIR):
                os.makedirs(CACHED_DIR)
            X_cached = os.path.join(CACHED_DIR, f'{mode}_X_cached.pt')
            y_cached = os.path.join(CACHED_DIR, f'{mode}_y_cached.pt')
            scaler_cached = os.path.join(CACHED_DIR, 'scaler_cached')
            torch.save(X, X_cached)
            torch.save(y, y_cached)
            with open(scaler_cached, 'wb') as f:
                pickle.dump(scaler, f)

        return (X, y)


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[0][index]
        label = self.data[1][index]
        return data, label


    def __len__(self) -> int:
        return len(self.data[1])