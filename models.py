# models.py
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from config import *
from collections import deque

class CausalAwareGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_dag=True, dag_edges=None):
        super().__init__()
        self.use_dag = use_dag
        self.var_map = {v: i for i, v in enumerate(ALL_VARIABLES)}
        
        # Adjacency Matrix
        if use_dag and dag_edges:
            adj = torch.zeros(len(self.var_map), len(self.var_map))
            for u, v in dag_edges:
                if u in self.var_map and v in self.var_map:
                    adj[self.var_map[u], self.var_map[v]] = 1
            adj += torch.eye(len(self.var_map))  
        else:
            adj = torch.ones(len(self.var_map), len(self.var_map))
        self.register_buffer('adj', adj)
        
        # Input encoder
        self.input_enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # GCN Layers
        self.conv1 = gnn.GCNConv(hidden_dim, hidden_dim)
        self.conv2 = gnn.GCNConv(hidden_dim, hidden_dim)

        # Classifier
        self.classifiers = nn.ModuleDict(
            {k: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, NUM_CLASSES_DICT[k])
            ) for k in TARGET_VARIABLES if k in NUM_CLASSES_DICT}
        )
        
        self._initialize_weights()
        print(f"GNN model initialization complete, target variable: {TARGET_VARIABLES}")
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, gnn.GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, batch_data):
        """
        Forward propagation
        """
        try:
            if not batch_data:
                return self._create_default_outputs(1)
            
            first_key = list(batch_data.keys())[0]
            B = batch_data[first_key].shape[0]

            encoded = []
            for var_name in ALL_VARIABLES:
                feature_key = f'{var_name}_raw'
                if feature_key in batch_data:
                    feature = batch_data[feature_key]
                    if feature.dim() == 1:
                        feature = feature.unsqueeze(1)
                    elif feature.dim() > 2:
                        feature = feature.view(feature.size(0), -1)
                    
                    enc = self.input_enc(feature)
                    encoded.append(enc)
                else:
                    device = self.adj.device
                    enc = self.input_enc(torch.zeros(B, 1, device=device))
                    encoded.append(enc)
            
            x = torch.stack(encoded, dim=1)  # [B, N, hidden]

            x_flat = x.reshape(B * len(ALL_VARIABLES), -1)
            
            edge_index = self.adj.nonzero(as_tuple=False).t().contiguous()
            edge_index = edge_index.repeat(1, B)
            offset = torch.arange(B, device=edge_index.device) * len(ALL_VARIABLES)
            edge_index = edge_index.view(2, B, -1) + offset.view(1, B, 1)
            edge_index = edge_index.view(2, -1)
            
            x_flat = torch.relu(self.conv1(x_flat, edge_index))
            x_flat = torch.relu(self.conv2(x_flat, edge_index))
            x = x_flat.view(B, len(ALL_VARIABLES), -1)

            outputs = {}
            for var_name in TARGET_VARIABLES:
                if var_name in self.var_map and var_name in self.classifiers:
                    idx = self.var_map[var_name]
                    logits = self.classifiers[var_name](x[:, idx])
                    if not logits.requires_grad:
                        logits = logits.requires_grad_(True)
                    
                    outputs[var_name] = logits
                    
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"Warning: {var_name} output contains NaN or Inf")
            
            missing_vars = set(TARGET_VARIABLES) - set(outputs.keys())
            if missing_vars:
                print(f"Warning: Output of the following variable is missing: {missing_vars}")
                for var_name in missing_vars:
                    outputs[var_name] = self._create_default_logits(B, var_name)
            
            return outputs
            
        except Exception as e:
            print(f"GNN forward propagation error: {e}")
            import traceback
            traceback.print_exc()
            B = 1
            if batch_data and list(batch_data.keys())[0] in batch_data:
                B = batch_data[list(batch_data.keys())[0]].shape[0]
            return self._create_default_outputs(B)

    def _create_default_outputs(self, batch_size):
        outputs = {}
        for var_name in TARGET_VARIABLES:
            outputs[var_name] = self._create_default_logits(batch_size, var_name)
        return outputs

    def _create_default_logits(self, batch_size, var_name):
        if var_name in NUM_CLASSES_DICT:
            num_classes = NUM_CLASSES_DICT[var_name]
        else:
            num_classes = 10
        
        device = next(self.parameters()).device

        logits = torch.randn(batch_size, num_classes, device=device) * 0.1
        return logits
    
class OptimizedHierarchicalCausalGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_dag=True, dag_edges=None, 
                 target_variable=None, hierarchy_vars=None, all_variables=None):
        super().__init__()
        self.use_dag = use_dag
        self.target_variable = target_variable
        self.hierarchy_vars = hierarchy_vars or []
        self.all_variables = all_variables or ALL_VARIABLES
        
        self.var_map = {v: i for i, v in enumerate(self.all_variables)}
        self.target_idx = self.var_map.get(target_variable, 0)
        
        # Constructing an optimized adjacency matrix based on DAG and hierarchical variables
        self.adj = self._build_optimized_adjacency(dag_edges, hierarchy_vars)
        
        # Input encodes
        self.input_encoders = nn.ModuleDict({
            var: nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ) for var in self.all_variables
        })
        
        # GCN layers
        self.gcn_layers = nn.ModuleList([
            gnn.GCNConv(hidden_dim, hidden_dim),
            gnn.GCNConv(hidden_dim, hidden_dim),
            gnn.GCNConv(hidden_dim, hidden_dim)
        ])
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Classifier
        if target_variable in NUM_CLASSES_DICT:
            output_dim = NUM_CLASSES_DICT[target_variable]
        else:
            output_dim = 10
            
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, gnn.GCNConv):
                nn.init.kaiming_normal_(m.lin.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.MultiheadAttention):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def _build_optimized_adjacency(self, dag_edges, hierarchy_vars):
        n_vars = len(self.all_variables)
        adj = torch.zeros(n_vars, n_vars)
        
        if dag_edges and self.use_dag:
            for u, v in dag_edges:
                if u in self.var_map and v in self.var_map:
                    u_idx, v_idx = self.var_map[u], self.var_map[v]
                    adj[u_idx, v_idx] = 1.0
                    adj[v_idx, u_idx] = 0.5
            
            # Strengthen the connections between hierarchical variables
            hierarchy_indices = [self.var_map[var] for var in hierarchy_vars if var in self.var_map]
            for i in hierarchy_indices:
                for j in hierarchy_indices:
                    if i != j:
                        if adj[i, j] == 0 and adj[j, i] == 0:
                            adj[i, j] = adj[j, i] = 0.8
                        else:
                            adj[i, j] = adj[j, i] = torch.max(adj[i, j], adj[j, i]) * 1.2
            
            # Special emphasis is placed on strengthening the connection between the target variable and the hierarchical variables.
            if self.target_variable in self.var_map:
                target_idx = self.var_map[self.target_variable]
                for var_idx in hierarchy_indices:
                    if var_idx != target_idx:
                        adj[target_idx, var_idx] = adj[var_idx, target_idx] = 1.0
        else:
            adj = torch.ones(n_vars, n_vars) * 0.5
            for i in range(n_vars):
                adj[i, i] = 1.0
        
        adj = torch.clamp(adj, 0.1, 1.0)
        return adj
    
    def forward(self, batch_data):
        try:
            B = self._get_batch_size(batch_data)
            device = next(self.parameters()).device  
            
            adj = self.adj.to(device)
            
            encoded_features = []
            for var_name in self.all_variables:
                feature = self._get_variable_feature(batch_data, var_name, B, device)
                enc = self.input_encoders[var_name](feature)
                encoded_features.append(enc)
            
            x = torch.stack(encoded_features, dim=1)  # [B, N_vars, H]
            
            x_flat = x.reshape(B * len(self.all_variables), -1)
            edge_index = self._build_batch_edge_index(B, device, adj)
            
            x1 = torch.relu(self.gcn_layers[0](x_flat, edge_index))
            x2 = torch.relu(self.gcn_layers[1](x1, edge_index))
            x3 = torch.relu(self.gcn_layers[2](x2, edge_index))
            
            x_final = x3 + self.residual_mlp(x1)
            x = x_final.view(B, len(self.all_variables), -1)
            
            target_feature = x[:, self.target_idx:self.target_idx+1]
            attended_features, attention_weights = self.attention(
                target_feature, x, x
            )
            
            target_encoded = x[:, self.target_idx]
            context_encoded = attended_features.squeeze(1)
            combined_features = torch.cat([target_encoded, context_encoded], dim=1)
            
            logits = self.classifier(combined_features)
            return logits
            
        except Exception as e:
            print(f"❌ {self.target_variable} : {e}")
            return self._create_safe_logits(B if 'B' in locals() else 1, device)
    
    def _get_batch_size(self, batch_data):
        if not batch_data:
            return 1
        for key, value in batch_data.items():
            if isinstance(value, torch.Tensor):
                return value.shape[0]
        return 1
    
    def _get_variable_feature(self, batch_data, var_name, batch_size, device):
        feature_key = f'{var_name}_raw'
        if feature_key in batch_data and batch_data[feature_key] is not None:
            feature = batch_data[feature_key]
            if feature.device != device:
                feature = feature.to(device)
                
            if feature.dim() == 1:
                feature = feature.unsqueeze(-1)
            elif feature.dim() > 2:
                feature = feature.view(feature.size(0), -1)
            if feature.size(-1) > 1:
                feature = feature[:, :1]
            return feature
        else:
            return torch.randn(batch_size, 1, device=device) * 0.01
    
    def _build_batch_edge_index(self, batch_size, device, adj):
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()
        edge_index = edge_index.repeat(1, batch_size)
        offset = torch.arange(batch_size, device=device) * len(self.all_variables)
        edge_index = edge_index.view(2, batch_size, -1) + offset.view(1, batch_size, 1)
        return edge_index.view(2, -1)
    
    def _create_safe_logits(self, batch_size, device):
        if self.target_variable in NUM_CLASSES_DICT:
            num_classes = NUM_CLASSES_DICT[self.target_variable]
        else:
            num_classes = 10
        return torch.zeros(batch_size, num_classes, device=device)

class IndependentGNNTrainer:
    
    def __init__(self, use_dag=False, device='auto', case_name="independent"):
        self.use_dag = use_dag
        self.case_name = case_name
        self.device = self._setup_device(device)
        self.model = self._setup_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'test_metrics': {}
        }
        
    def _setup_device(self, device):
        if device == 'auto':
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(device)
    
    def _setup_model(self):
        input_dim = 1
        hidden_dim = HIDDEN_DIM
        
        if self.use_dag:
            model = CausalAwareGNN(
                input_dim=input_dim, 
                hidden_dim=hidden_dim,
                use_dag=True,
                dag_edges=DAG_EDGES
            )
        else:
            model = CausalAwareGNN(
                input_dim=input_dim,
                hidden_dim=hidden_dim, 
                use_dag=False
            )
        
        return model.to(self.device)
    
    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            inputs = {}
            for var_name in ALL_VARIABLES:
                raw_key = f'{var_name}_raw'
                if raw_key in batch:
                    inputs[raw_key] = batch[raw_key].to(self.device)

            outputs = self.model(inputs)

            loss = 0
            for var_name in TARGET_VARIABLES:
                if var_name in outputs and f'{var_name}_label' in batch:
                    target = batch[f'{var_name}_label'].to(self.device)
                    var_loss = self.criterion(outputs[var_name], target)
                    loss += var_loss

            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        return total_loss / batch_count if batch_count > 0 else 0
    
    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        batch_count = 0
        all_predictions = {}
        all_targets = {}
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {}
                for var_name in ALL_VARIABLES:
                    raw_key = f'{var_name}_raw'
                    if raw_key in batch:
                        inputs[raw_key] = batch[raw_key].to(self.device)
                
                outputs = self.model(inputs)
                
                batch_loss = 0
                for var_name in TARGET_VARIABLES:
                    if var_name in outputs and f'{var_name}_label' in batch:
                        target = batch[f'{var_name}_label'].to(self.device)
                        var_loss = self.criterion(outputs[var_name], target)
                        batch_loss += var_loss.item()
                        
                        pred = torch.argmax(outputs[var_name], dim=1)
                        if var_name not in all_predictions:
                            all_predictions[var_name] = []
                            all_targets[var_name] = []
                        
                        all_predictions[var_name].extend(pred.cpu().numpy())
                        all_targets[var_name].extend(target.cpu().numpy())
                
                total_loss += batch_loss
                batch_count += 1

        accuracies = {}
        for var_name in TARGET_VARIABLES:
            if var_name in all_predictions and len(all_predictions[var_name]) > 0:
                preds = np.array(all_predictions[var_name])
                targets = np.array(all_targets[var_name])
                accuracy = (preds == targets).mean()
                accuracies[var_name] = accuracy
        
        avg_accuracy = np.mean(list(accuracies.values())) if accuracies else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        return avg_loss, avg_accuracy, accuracies
    
    def predict(self, dataloader):
        self.model.eval()
        all_outputs = {}
        
        with torch.no_grad():
            for batch in dataloader:
                inputs = {}
                for var_name in ALL_VARIABLES:
                    raw_key = f'{var_name}_raw'
                    if raw_key in batch:
                        inputs[raw_key] = batch[raw_key].to(self.device)

                outputs = self.model(inputs)

                for var_name in TARGET_VARIABLES:
                    if var_name in outputs:
                        if var_name not in all_outputs:
                            all_outputs[var_name] = []
                        all_outputs[var_name].append(outputs[var_name].cpu().numpy())
        
        for var_name in all_outputs:
            all_outputs[var_name] = np.concatenate(all_outputs[var_name], axis=0)
        
        return all_outputs
    
    def plot_training_history(self, save_dir):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation loss')
        ax1.set_title(f'{self.case_name} - Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.history['val_accuracy'], 'g-', label='Validation accuracy')
        ax2.set_title(f'{self.case_name} - Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        plot_path = save_dir / f'{self.case_name}_training_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training history saved to: {plot_path}")
    
    def save_model(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f'{self.case_name}_model.pth'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'use_dag': self.use_dag,
            'history': self.history
        }, model_path)
        
        # 保存训练历史
        history_path = save_dir / f'{self.case_name}_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # 保存配置信息
        config_path = save_dir / f'{self.case_name}_config.json'
        config = {
            'case_name': self.case_name,
            'use_dag': self.use_dag,
            'device': str(self.device),
            'target_variables': TARGET_VARIABLES,
            'all_variables': ALL_VARIABLES
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Model has been saved to: {model_path}")
        print(f"Training history has been saved to: {history_path}")
        print(f"Configuration information has been saved to: {config_path}")
    
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint.get('history', self.history)
    
    def train(self, train_reader, val_reader, test_reader, epochs=EPOCHS, early_stop_patience=EARLY_STOP_PATIENCE):
        print(f"Start Training {self.case_name}...")
        print(f"Device: {self.device}")
        print(f"If use DAG: {self.use_dag}")
        print(f"Target variables: {TARGET_VARIABLES}")
        
        early_stopper = EarlyStopper(patience=early_stop_patience, mode='min')
        
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_reader)
            self.history['train_loss'].append(train_loss)
            
            val_loss, val_accuracy, val_acc_details = self.evaluate(val_reader)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            
            print(f'Epoch {epoch}/{epochs}: '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}')
            
            if early_stopper(val_loss):
                print(f'Early stopping at epoch {epoch}')
                break
        
        test_loss, test_accuracy, test_acc_details = self.evaluate(test_reader)
        self.history['test_metrics'] = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_acc_details': test_acc_details
        }
        
        print(f"  Loss(test): {test_loss:.4f}")
        print(f"  Accuracy(test): {test_accuracy:.4f}")
        for var_name, acc in test_acc_details.items():
            print(f"  {var_name}: {acc:.4f}")
        
        return self.history


class EarlyStopper:
    def __init__(self, patience=20, mode='min', min_delta=0.001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.counter = 0

    def __call__(self, val):
        if self.best is None:
            self.best = val
            return False
        
        if self.mode == 'min':
            improvement = self.best - val > self.min_delta
        else:
            improvement = val - self.best > self.min_delta
            
        if improvement:
            self.best = val
            self.counter = 0
            print(f"Early stop: Optimal value updated to {val:.4f}")
        else:
            self.counter += 1
            print(f"Early Stop: Counter {self.counter}/{self.patience}")
            
        return self.counter >= self.patience