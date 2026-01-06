# sensors.py
import torch
from domiknows.sensor.pytorch.learners import ModuleLearner
from models import OptimizedHierarchicalCausalGNN,CausalAwareGNN
from config import *

class HierarchicalCausalGNNLearner(ModuleLearner):
    def __init__(self, *pres, use_dag, target_variable, hierarchy_vars, device='auto'):
        input_dim = 1
        
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        
        gnn_model = OptimizedHierarchicalCausalGNN(
            input_dim, 
            HIDDEN_DIM,
            use_dag=use_dag,
            dag_edges=DAG_EDGES if use_dag else None,
            target_variable=target_variable,
            hierarchy_vars=hierarchy_vars,
            all_variables=ALL_VARIABLES
        )
        
        super().__init__(*pres, module=gnn_model, device=device)
        
        self.target_variable = target_variable
        self.hierarchy_vars = hierarchy_vars


    def forward(self, *inputs):
        batch_data = {}
        batch_size = self._get_batch_size(inputs)
        
        for i, var_name in enumerate(self.hierarchy_vars):
            if i < len(inputs) and inputs[i] is not None:
                feature = self._process_input_feature(inputs[i])
                if feature is not None:
                    batch_data[f'{var_name}_raw'] = feature

        if not batch_data:
            batch_data = self._create_default_inputs(batch_size)
        
        try:
            batch_data = self._ensure_device_consistency(batch_data)
            
            logits = self.module(batch_data)
            return self._validate_output(logits, batch_size)
            
        except Exception as e:
            print(f"❌ {self.target_variable} learner error: {e}")
            import traceback
            traceback.print_exc()
            return self._create_safe_logits(batch_size)
    
    def _process_input_feature(self, input_data):
        """
        Handling input features - using self.device (set by the parent class)
        """
        if input_data is None:
            return None
            
        if not isinstance(input_data, torch.Tensor):
            input_data = torch.tensor(input_data, dtype=torch.float32, device=self.device)
        else:
            if input_data.device != self.device:
                input_data = input_data.to(self.device)
        
        if input_data.dim() == 1:
            input_data = input_data.unsqueeze(1)
        elif input_data.dim() > 2:
            input_data = input_data.view(input_data.size(0), -1)
        
        if input_data.size(1) > 1:
            input_data = input_data[:, :1]
            
        return input_data
    
    def _ensure_device_consistency(self, batch_data):
        consistent_data = {}
        for key, tensor in batch_data.items():
            if isinstance(tensor, torch.Tensor) and tensor.device != self.device:
                consistent_data[key] = tensor.to(self.device)
            else:
                consistent_data[key] = tensor
        return consistent_data
    
    def _get_batch_size(self, inputs):
        for inp in inputs:
            if inp is not None:
                if isinstance(inp, torch.Tensor):
                    return inp.shape[0]
                elif hasattr(inp, '__len__'):
                    return len(inp)
        return 1
    
    def _create_default_inputs(self, batch_size):
        batch_data = {}
        for var_name in self.hierarchy_vars[:min(3, len(self.hierarchy_vars))]:
            default_input = torch.randn(batch_size, 1, device=self.device) * 0.01
            batch_data[f'{var_name}_raw'] = default_input
        return batch_data
    
    def _validate_output(self, logits, batch_size):
        if not isinstance(logits, torch.Tensor):
            return self._create_safe_logits(batch_size)
        
        if logits.device != self.device:
            logits = logits.to(self.device)
        
        expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
        if logits.size(-1) != expected_classes:
            print(f"⚠️ {self.target_variable} logits shape error: {logits.shape}")
            return self._create_safe_logits(batch_size)
        
        if not logits.requires_grad:
            logits = logits.clone().requires_grad_(True)
            
        return logits
    
    def _create_safe_logits(self, batch_size):
        expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
        logits = torch.zeros(batch_size, expected_classes, device=self.device)
        return logits.requires_grad_(True)
    
class SimpleGNNLearner(ModuleLearner):
    """
    A simple fully connected GNN learner for case 1
    """
    def __init__(self, *pres, target_variable, device='auto'):
        input_dim = 1  
        
        gnn_model = CausalAwareGNN(
            input_dim, 
            HIDDEN_DIM,
            use_dag=False,  
            dag_edges=None
        )
        
        super().__init__(*pres, module=gnn_model, device=device)
        self.target_variable = target_variable

    def forward(self, *inputs):
        processed_inputs = []
        for i, inp in enumerate(inputs):
            if inp is not None:
                if not isinstance(inp, torch.Tensor):
                    inp = torch.tensor(inp, dtype=torch.float32, device=self.device)
                
                if not inp.requires_grad:
                    inp = inp.clone().requires_grad_(True)
                
                processed_inputs.append(inp)
            else:
                processed_inputs.append(None)
        
        batch_data = {}
        valid_inputs = 0
        
        for i, var_name in enumerate(ALL_VARIABLES):
            if i < len(processed_inputs) and processed_inputs[i] is not None:
                input_tensor = processed_inputs[i]

                if input_tensor.dim() == 1:
                    input_tensor = input_tensor.unsqueeze(1)
                elif input_tensor.dim() > 2:
                    input_tensor = input_tensor.view(input_tensor.size(0), -1)
                    if input_tensor.size(1) > 1:
                        input_tensor = input_tensor[:, :1]
                
                batch_data[f'{var_name}_raw'] = input_tensor
                valid_inputs += 1
        
        if valid_inputs == 0:
            print(f"⚠️ Warning: {self.target_variable} The simple learner has no valid input.")
            if inputs and inputs[0] is not None:
                batch_size = inputs[0].shape[0]
            else:
                batch_size = 1
            device = next(self.module.parameters()).device
            default_input = torch.randn(batch_size, 1, device=device) * 0.01
            default_input = default_input.requires_grad_(True)
            batch_data[f'{ALL_VARIABLES[0]}_raw'] = default_input

        try:
            outputs = self.module(batch_data)
            
            if self.target_variable in outputs:
                logits = outputs[self.target_variable]
                
                if isinstance(logits, torch.Tensor):
                    expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
                    if logits.size(-1) != expected_classes:
                        print(f"⚠️ Warning: {self.target_variable} logits shape is incorrect: {logits.shape}")
                        batch_size = logits.size(0) if logits.dim() > 0 else 1
                        logits = torch.randn(batch_size, expected_classes, device=logits.device) * 0.1
                    
                    if not logits.requires_grad:
                        logits = logits.clone().requires_grad_(True)
                        
                else:
                    print(f"❌ Error: {self.target_variable} returns a non-tensor logits")
                    if inputs and inputs[0] is not None:
                        batch_size = inputs[0].shape[0]
                    else:
                        batch_size = 1
                    expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
                    device = next(self.module.parameters()).device
                    logits = torch.randn(batch_size, expected_classes, device=device) * 0.1
                    logits = logits.requires_grad_(True)
                
                return logits
            else:
                print(f"❌ Error: {self.target_variable} not in model output")
                if inputs and inputs[0] is not None:
                    batch_size = inputs[0].shape[0]
                else:
                    batch_size = 1
                expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
                device = next(self.module.parameters()).device
                logits = torch.randn(batch_size, expected_classes, device=device) * 0.1
                return logits.requires_grad_(True)
                
        except Exception as e:
            print(f"❌ {self.target_variable} Simple learner forward propagation error: {e}")
            import traceback
            traceback.print_exc()
            if inputs and inputs[0] is not None:
                batch_size = inputs[0].shape[0]
            else:
                batch_size = 1
            expected_classes = NUM_CLASSES_DICT.get(self.target_variable, 10)
            device = next(self.module.parameters()).device
            logits = torch.randn(batch_size, expected_classes, device=device) * 0.1
            return logits.requires_grad_(True)