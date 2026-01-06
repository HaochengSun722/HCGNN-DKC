# data_reader.py
import pandas as pd
import torch
from domiknows.sensor.pytorch.sensors import TorchSensor
from config import ALL_VARIABLES, TARGET_VARIABLES
import numpy as np
from domiknows.sensor.pytorch.sensors import FunctionalSensor

class UrbanCSVDataReader:
    """A standardized CSV data reader that meets the DomiKnows DataNode building requirements."""
    def __init__(self, csv_file_path, batch_size=32):
        self.csv_file_path = csv_file_path
        self.batch_size = batch_size
        self.all_variables = ALL_VARIABLES
        self.target_variables = TARGET_VARIABLES
        
        self.known_variables = ['BLA', 'BLP', 'BL', 'LU', 'SBLU', 'SLR']
        
        # Define the mapping from integers to letters (DomiKnowS requires strings, GNN requires ints).
        self.label_mapping = {
            'MABH': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'BN': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'FAR': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'BD': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'BL': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'},
            'LU': {0: 'A', 1: 'B', 2: 'E', 3: 'G', 4: 'M', 5: 'R', 6: 'S', 7: 'U'},
            'ABF': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'CL': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'RBS': {0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I', 8: 'J'},
            'HBN': {0: 'A', 1: 'F', 2: 'G', 3: 'H', 4: 'I', 5: 'J'},
            'AHBH': {0: 'A', 1: 'F', 2: 'G', 3: 'H', 4: 'I', 5: 'J'},
            'ABH': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'},
            'DHBH': {0: 'A', 1: 'G', 2: 'H', 3: 'I', 4: 'J'},
            'BLA': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'},
            'BLP': {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J'},
            'HBFR': {0:'A', 1:'F', 2: 'G', 3: 'H'},
            'SBLU': {i: str(i) for i in range(66)},  
            'SLR': {i: str(i) for i in range(31)},
        }
        
        # Define a reverse mapping (string to integer).
        self.reverse_mapping = {}
        for var_name, mapping in self.label_mapping.items():
            self.reverse_mapping[var_name] = {v: k for k, v in mapping.items()}
        
        self.load_data()
    
    def load_data(self):
        """load_data"""
        try:
            self.df = pd.read_csv(self.csv_file_path)
            print(f"Data loaded successfully, number of samples: {len(self.df)}")

            self.data_array = self.df[self.all_variables].values.astype(np.float32)

            for var_name in self.target_variables:
                unique_values = self.df[var_name].unique()
                min_val, max_val = unique_values.min(), unique_values.max()
                expected_min, expected_max = 0, len(self.label_mapping[var_name]) - 1
                
                if min_val < expected_min or max_val > expected_max:
                    print(f"Warning: {var_name} tag is outside the expected range!")
                
            print(f"data shape: {self.data_array.shape}")
            print(f"target variables: ({len(self.df)}, {len(self.target_variables)})")
                
        except Exception as e:
            print(f"Error in LOAD DATA: {e}")
            raise
    
    def __iter__(self):
        """Iterator"""
        for i in range(0, len(self.df), self.batch_size):
            batch_indices = range(i, min(i + self.batch_size, len(self.df)))
            yield self.make_batch(batch_indices)
    
    def make_batch(self, indices):
        batch_data = {}
        batch_size = len(indices)

        # 1. Create an instance for ROOT-block
        batch_data['block'] = list(range(batch_size))
        batch_data['block_raw'] = torch.zeros(batch_size, 1, dtype=torch.float32)

        # 2. Ensure all feature data is of float type.
        for i, var_name in enumerate(self.all_variables):
            var_data = self.data_array[indices, i:i+1]
            
            # Ensure 2D tensor
            if len(var_data.shape) == 1:
                var_data = var_data.reshape(-1, 1)
                
            tensor_data = torch.tensor(var_data, dtype=torch.float32)
            tensor_data = tensor_data.requires_grad_(True)
            batch_data[f'{var_name}_raw'] = tensor_data

        # 3. Enforce the existence of known variables
        for var_name in self.known_variables:
            feature_key = f'{var_name}_raw'
            if feature_key not in batch_data or batch_data[feature_key] is None:
                print(f"❌Critical error: The known variable {var_name} is missing in the data reader!")
                batch_data[feature_key] = torch.randn(batch_size, 1, dtype=torch.float32) * 0.1 + 1.0

        # 4. Label data - Keep it as an integer without gradients.
        for var_name in self.target_variables:
            int_labels = self.df[var_name].iloc[indices].values.astype(np.int64)
            
            str_labels = []
            for int_label in int_labels:
                if int_label in self.label_mapping[var_name]:
                    str_labels.append(self.label_mapping[var_name][int_label])
                else:
                    default_val = list(self.label_mapping[var_name].values())[0]
                    str_labels.append(default_val)
            
            batch_data[var_name] = str_labels
            batch_data[f'{var_name}_label'] = torch.tensor(int_labels, dtype=torch.long)
        
        # 5. Final validation
        #self._strict_validate_batch(batch_data, batch_size)
        
        return batch_data

    def _strict_validate_batch(self, batch_data, batch_size):
        """validate_batch"""
        print(f"\n Data reader batch verification (size: {batch_size}):")

        known_missing = []
        for var_name in self.known_variables:
            feature_key = f'{var_name}_raw'
            if feature_key not in batch_data:
                known_missing.append(var_name)
                print(f"  ❌ {var_name}: MISSING")
            else:
                feature = batch_data[feature_key]
                if feature is None:
                    known_missing.append(var_name)
                    print(f"  ❌ {var_name}: = None")
                elif not isinstance(feature, torch.Tensor):
                    known_missing.append(var_name)
                    print(f"  ❌ {var_name}: Error type - {type(feature)}")
                elif feature.shape[0] != batch_size:
                    known_missing.append(var_name)
                    print(f"  ❌ {var_name}: Error batch size - {feature.shape[0]} vs {batch_size}")
                else:
                    unique_count = len(torch.unique(feature))
                    mean_val = feature.mean().item()
                    print(f"  ✅ {var_name}: Shape={feature.shape}, Unique={unique_count}, Average={mean_val:.3f}")
        
        if known_missing:
            print(f"❌ Data reader: {len(known_missing)} missing variables: {known_missing}")
        else:
            print("✅ Data reader: All known variables are complete.")
    
    def validate_labels(self):
        "Verify that the tag mapping is correct."
        print("\n=== Label mapping verification ===")
        sample_batch = next(iter(self))
        
        for var_name in self.target_variables:
            str_labels = sample_batch[var_name]
            int_labels = sample_batch[f'{var_name}_label']
            
            print(f"\n{var_name}:")
            print(f"  String type: {str_labels[:5]}")
            print(f"  Int type: {int_labels[:5].tolist()}")
            
            for i, (str_label, int_label) in enumerate(zip(str_labels[:3], int_labels[:3])):
                expected_str = self.label_mapping[var_name][int_label.item()]
                if str_label != expected_str:
                    print(f"  Error: Index {i} -int_variable {int_label} should map to '{expected_str}', but to '{str_label}'")
                else:
                    print(f"  Correct: int_variable {int_label} -> '{str_label}'")
        
        print("=== Done(Label mapping valid) ===\n")

class UrbanFeatureSensor(FunctionalSensor):
    def __init__(self, variable_name, *pres, **kwargs):
        super().__init__(*pres, **kwargs)
        self.variable_name = variable_name

    def forward(self, *args):
        key = f'{self.variable_name}_raw'
        data = self.context_helper.get(key, torch.zeros(1, 1, dtype=torch.float32))
        
        if data is None:
            print(f"⚠️ {self.variable_name} Feature sensor: Receives None data, uses the default zero tensor")
            data = torch.zeros(1, 1, dtype=torch.float32)

        if not isinstance(data, torch.Tensor):
            print(f"⚠️ {self.variable_name} Feature sensor: The data is not a tensor; it is of type {type(data)}. Convert it to a tensor.")
            data = torch.tensor(data, dtype=torch.float32)
        elif data.dtype != torch.float32:
            print(f"⚠️ {self.variable_name} Feature sensor: The data is not a floating-point type; it is of type {data.dtype}. Please convert it to a float type.")
            data = data.float()
        
        if data.dim() == 1:
            data = data.unsqueeze(1)
        elif data.dim() > 2:
            data = data.view(data.size(0), -1)

        if not data.requires_grad:
            data = data.detach().requires_grad_(True)
            
        return data

class UrbanLabelSensor(FunctionalSensor):
    def __init__(self, variable_name, *pres, **kwargs):
        kwargs['label'] = True
        super().__init__(*pres, **kwargs)
        self.variable_name = variable_name

    def forward(self, *args):
        key = f'{self.variable_name}_label'
        labels = self.context_helper.get(key, torch.zeros(1, dtype=torch.long))
        
        if labels is None:
            print(f"⚠️ {self.variable_name} Label sensor: Received None data, using the default zero tensor")
            labels = torch.zeros(1, dtype=torch.long)

        if not isinstance(labels, torch.Tensor):
            print(f"⚠️ {self.variable_name} Label sensor: The labels are not tensors{type(labels)}. Convert them to tensors.")
            labels = torch.tensor(labels, dtype=torch.long)

        if labels.dim() > 1:
            labels = labels.squeeze()
        elif labels.dim() == 0:
            labels = labels.unsqueeze(0)
            
        labels = labels.long() if labels.dtype != torch.long else labels
        #print(f"  {self.variable_name} Label sensor: Output shape = {labels.shape}, Unique value = {torch.unique(labels)}")
        return labels