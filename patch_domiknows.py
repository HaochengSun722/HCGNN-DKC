"""
DomiKnows Patch Files

Fixes the issue where loss and metric calculations were None

Note: In DataNodes, the label attribute is accessed using 'label/label', but the attribute of Concept is still 'label'.

"""

import torch
import torch.nn.functional as F
from domiknows.program import SolverPOIProgram, LearningBasedProgram
from domiknows.program.model.pytorch import PoiModel
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric
from domiknows.sensor.pytorch.sensors import TorchSensor
from config import *
import numpy as np
import torch
import types
from contextlib import contextmanager

import torch.nn.functional as F
import numpy as np
from domiknows.program.model.base import Mode

def detuple(*args):
    if isinstance(args, tuple) and len(args) == 1:
        return args[0]
    return args


class FixedSolverModel(PoiModel):
    def __init__(self, graph, poi=None, loss=None, metric=None, inferTypes=None, inference_with=None, device='auto'):
        super().__init__(graph, poi=poi, loss=loss, metric=metric, device=device)
        
        if inferTypes is None:
            self.inferTypes = ['ILP']
        else:
            self.inferTypes = inferTypes
            
        if inference_with is None:
            self.inference_with = []
        else:
            self.inference_with = inference_with

    def inference(self, builder):
        """Inference method has been improved to ensure that all sensors are functioning correctly."""
        for prop in self.poi:
            for sensor in prop.find(TorchSensor):
                sensor(builder)
        
        if builder.needsBatchRootDN():
            builder.addBatchRootDN()
            
        datanode = builder.getDataNode(device=self.device)
        
        for infertype in self.inferTypes:
            try:
                {
                    'ILP': lambda: datanode.inferILPResults(*self.inference_with, fun=None, epsilon=None),
                    'local/argmax': lambda: datanode.inferLocal(),
                    'local/softmax': lambda: datanode.inferLocal(),
                    'argmax': lambda: datanode.infer(),
                    'softmax': lambda: datanode.infer(),
                }[infertype]()
            except KeyError:
                print(f"Unknown infernece type: {infertype}")
        
        return builder

    def populate(self, builder, run=True):
        data_item = self.inference(builder)
        return super().populate(builder, run=False)
    
    def find_sensors(self, prop):
        """Rewrite the sensor pairing method and use get_fullname() to resolve the concept name."""
        from itertools import combinations
        
        sensors = list(prop.find(TorchSensor))
        
        if prop.name == 'logits':
            try:
                fullname = prop.get_fullname()

                parts = fullname.split('/')
                if len(parts) >= 2:
                    concept_name = parts[-2]
                    
                    if concept_name in list(self.graph):
                        concept = self.graph[concept_name]
                        if 'label' in concept:
                            label_prop = concept['label']
                            label_sensors = list(label_prop.find(TorchSensor))
                            if label_sensors:

                                for logits_sensor in sensors:
                                    for label_sensor in label_sensors:
                                        yield (logits_sensor, label_sensor)
                                return  
                else:
                    print(f"⚠️ Warning: Unable to resolve concept from full name: {fullname}")
                    
            except Exception as e:
                print(f"❌ Error: Failed to resolve concept name: {e}")
        
        for sensor1, sensor2 in combinations(sensors, r=2):
            if sensor1.label:
                target_sensor = sensor1
                output_sensor = sensor2
            elif sensor2.label:
                target_sensor = sensor2
                output_sensor = sensor1
            else:
                continue
            if output_sensor.label:
                # two targets, skip
                continue
            yield output_sensor, target_sensor


class FixedSolverPOIProgram(LearningBasedProgram):
    def __init__(self, graph, **kwargs):
        super().__init__(graph, FixedSolverModel, **kwargs)


class DomiKnowsPatcher:
    """
    DomiKnows Patch Manager
    """
    
    @staticmethod
    def create_fixed_program(graph, target_variables, **kwargs):
        """
        Create the fixed Program
        
        Args:
            graph: Concept graph
            target_variables: List of target variables
            **kwargs: Other parameters
        
        Returns:
            FixedSolverPOIProgram
        """
        poi_list = DomiKnowsPatcher.create_poi_list(graph)
        
        default_kwargs = {
            'poi': poi_list,
            'inferTypes': kwargs.get('inferTypes', ['local/softmax']),
            'loss': kwargs.get('loss', NBCrossEntropyLoss()),
            'metric': kwargs.get('metric', PRF1Tracker(DatanodeCMMetric(inferType='local/softmax'))),
            'inference_with': [graph[var_name] for var_name in target_variables if var_name in list(graph)]
        }
        
        default_kwargs.update(kwargs)
        try:
            program = FixedSolverPOIProgram(graph, **default_kwargs)
            return program
        except Exception as e:
            print(f"❌ Error: Failed to create the fixed Program: {e}")
            return DomiKnowsPatcher.create_fallback_program(graph, poi_list, **default_kwargs)
    
    @staticmethod
    def create_poi_list(graph):
        """
        Create the correct list of POIs

        Note: Use the concept's 'label' attribute, but access it in the DataNode via 'label/label'.
        
        Args:
            graph: Concept graph
            target_variables: List of target variables
        
        Returns:
            POI tuple
        """
        poi_list = []
        poi_list.append(graph['block'])
        for var_name in ALL_VARIABLES:
            if var_name in list(graph):
                # concept = graph[var_name]
                
                # if 'logits' in concept:
                #     poi_list.append(concept['logits'])
                # else:
                #     poi_list.append(concept)
                    
                # if 'label' in concept:
                #     poi_list.append(concept['label'])
                poi_list.append(graph[var_name])
        
        poi_tuple = tuple(poi_list)
        for i, poi_item in enumerate(poi_tuple):
            print(f"  {i+1}. {poi_item.fullname}")
        
        return poi_tuple
    
    @staticmethod
    def create_fallback_program(graph, poi_list, **kwargs):
        try:
            fallback_kwargs = kwargs.copy()
            fallback_kwargs.pop('inference_with', None)
            
            program = SolverPOIProgram(graph, **fallback_kwargs)
            return program
        except Exception as e:
            print(f"❌ Error: {e}")
            raise
    
    @staticmethod
    def debug_sensor_calls(program, test_batch):
        """
        Debugging Sensor Calls and Result Collection
        
        Args:
            program: Program
            test_batch: Test batch data
        
        Returns:
            DataNodeBuilder
        """
        from domiknows.graph.dataNode import DataNodeBuilder
        
        builder = DataNodeBuilder()
        builder['graph'] = program.graph
        
        for prop in program.model.poi:
            sensors = list(prop.find(TorchSensor))
            #print(f"  Property: {prop.fullname}: {len(sensors)}")
            for sensor in sensors:
                try:
                    result = sensor(builder)
                    #print(f"    Sensor: {sensor.fullname}, type: {type(result)}")
                    #if hasattr(result, 'shape'):
                        #print(f"      Shape: {result.shape}")
                except Exception as e:
                    print(f"    Sensor: {sensor.fullname}, error: {e}")
        
        return builder
    
    @staticmethod
    def validate_data_node_structure(program, test_batch, target_variables):
        """
        Verify the DataNode data structure, paying particular attention to the access method of the label attribute.
        """
        print(f"\nVerify the DataNode data structure:")
        
        try:
            datanodes = list(program.populate(dataset=[test_batch]))
            if datanodes:
                datanode = datanodes[0]
                
                for var_name in target_variables[:3]: 
                    if var_name in list(program.graph):
                        concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                        if concept_nodes:
                            print(f"  {var_name}: {len(concept_nodes)} ")
                            
                            node = concept_nodes[0]
                            attributes = node.getAttributes()
                            print(f"    Keys: {list(attributes.keys())}")
                            
                            label_direct = node.getAttribute('label')
                            label_slash = node.getAttribute('label/label')
                            
                            print(f"    getAttribute('label'): {label_direct}")
                            print(f"    getAttribute('label/label'): {label_slash}")
                            logits_test = node.getAttribute('logits')
                            
                            print(f"    getAttribute('logits'): {logits_test}")
                            print(f"    shape of getAttribute('logits'): {logits_test.shape}")
                            
                            if label_slash is not None:
                                print(f"    ✅ use 'label/label'")
                                
        except Exception as e:
            print(f"DataNode exploration error: {e}")
    
    

    @staticmethod
    def safe_training_loop(program, train_reader, val_reader, optimizer, epochs, early_stopper, target_variables):
        """
        Safe Training Loop - Solving Gradient Problems with Temporary Patches
        """
        history = {
            'train_loss': [], 
            'val_loss': [],
            'val_accuracy': [],
            'epochs_completed': 0,
        }
        
        for epoch in range(1, epochs + 1):
            print(f'\nEpoch {epoch}/{epochs}')
            
            print("Training...")
            try:
                train_loss = DomiKnowsPatcher.manual_train_epoch(
                    program, train_reader, optimizer, target_variables
                )
                
                history['train_loss'].append(train_loss)
                print(f"  Training loss: {train_loss:.4f}")
                    
            except Exception as e:
                print(f"Training Failed to: {e}")
                history['train_loss'].append(float('inf'))
            
            print("Validation...")
            try:
                val_loss, val_accuracy, val_acc_details = DomiKnowsPatcher.manual_test_epoch(
                    program, val_reader, target_variables
                )
                
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_accuracy)
                print(f"  Validation loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {val_accuracy:.4f}")
                
                for var_name, acc in val_acc_details.items():
                    print(f"    {var_name}: {acc:.4f}")
                    
            except Exception as e:
                print(f"Failed to: {e}")
                history['val_loss'].append(float('inf'))
                history['val_accuracy'].append(0.0)
            
            history['epochs_completed'] = epoch
            
            if history['val_loss'] and history['val_loss'][-1] != float('inf'):
                if early_stopper(history['val_loss'][-1]):
                    print(f'Early stopping at epoch {epoch}')
                    break
        
        return history

    @staticmethod
    def manual_train_epoch(program, train_reader, optimizer, target_variables):
        program.model.train()
        total_loss = 0
        batch_count = 0

        class GradientAwareProgram(type(program)):
            def populate_epoch(self, dataset):
                self.model.mode(Mode.POPULATE)
                self.model.reset()
                for i, data_item in enumerate(dataset):
                    _, _, *output = self.model(data_item)
                    yield detuple(*output[:1])
        
        # Temporarily replace the program class
        original_class = program.__class__
        program.__class__ = GradientAwareProgram
        
        try:
            for batch_idx, batch in enumerate(train_reader):
                try:
                    optimizer.zero_grad()

                    datanodes = list(program.populate(dataset=[batch]))
                    
                    if not datanodes:
                        continue
                        
                    datanode = datanodes[0]
                    
                    first_var_nodes = datanode.findDatanodes(select=program.graph[target_variables[0]])
                    batch_size = len(first_var_nodes) if first_var_nodes else 0
                    
                    if batch_size == 0:
                        continue

                    all_var_logits = {var_name: [] for var_name in target_variables}
                    all_var_labels = {var_name: [] for var_name in target_variables}
                    valid_variables = set()
                    
                    for var_name in target_variables:
                        if var_name in list(program.graph):
                            concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                            
                            for sample_idx in range(min(batch_size, len(concept_nodes))):
                                node = concept_nodes[sample_idx]
                                logits = node.getAttribute('logits')
                                label = node.getAttribute('label/label')
                                
                                if logits is not None and label is not None:
                                    if isinstance(logits, torch.Tensor):
                                        logits_tensor = logits
                                    else:
                                        logits_tensor = torch.tensor(logits, dtype=torch.float32)
                                    
                                    if isinstance(label, torch.Tensor):
                                        label_tensor = label
                                    else:
                                        label_tensor = torch.tensor(label, dtype=torch.long)
                                    
                                    if label_tensor.dtype != torch.long:
                                        label_tensor = label_tensor.long()
                                    
                                    if not logits_tensor.requires_grad:
                                        print(f"Warning: The logits of the batch {batch_idx} variable {var_name} instance {sample_idx} have no gradient information")
                                        continue
                                    
                                    all_var_logits[var_name].append(logits_tensor)
                                    all_var_labels[var_name].append(label_tensor)
                                    valid_variables.add(var_name)
                    
                    batch_loss = 0
                    valid_var_count = 0
                    
                    for var_name in valid_variables:
                        if all_var_logits[var_name] and all_var_labels[var_name]:
                            try:
                                var_logits = torch.stack(all_var_logits[var_name])
                                var_labels = torch.stack(all_var_labels[var_name])
                                
                                var_loss = F.cross_entropy(var_logits, var_labels)
                                batch_loss += var_loss
                                valid_var_count += 1
                                
                            except Exception as e:
                                print(f"Loss calculation for variable {var_name} failed: {e}")
                                continue
                    
                    if valid_var_count > 0 and batch_loss != 0:
                        batch_loss.backward()
                        optimizer.step()
                        
                        total_loss += batch_loss.item()
                        batch_count += 1

                        has_gradients = False
                        for name, param in program.model.named_parameters():
                            if param.requires_grad and param.grad is not None:
                                grad_norm = param.grad.norm().item()
                                if grad_norm > 0:
                                    has_gradients = True
                                    break
                        
                        if batch_idx % 5 == 0:
                            avg_batch_loss = batch_loss.item() / valid_var_count if valid_var_count > 0 else batch_loss.item()
                            grad_status = "Have gradient" if has_gradients else "No gradient"
                            print(f" Batch {batch_idx}: Total loss={batch_loss.item():.4f}, Average loss={avg_batch_loss:.4f}, "
                                f" Number of variables={valid_var_count}, Number of samples={batch_size}, {grad_status}")
                    else:
                        print(f" Batch {batch_idx}: No valid loss calculation")
                            
                except Exception as e:
                    print(f"Training batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        finally:
            # Restore the original class
            program.__class__ = original_class
        
        avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        print(f"Avg loss: {avg_loss:.4f}")
        return avg_loss

    @staticmethod
    def manual_test_epoch(program, test_reader, target_variables):
        program.model.eval()
        total_loss = 0
        batch_count = 0

        var_correct = {var_name: 0 for var_name in target_variables}
        var_total = {var_name: 0 for var_name in target_variables}
        all_predictions = {var_name: [] for var_name in target_variables}
        all_targets = {var_name: [] for var_name in target_variables}

        with torch.no_grad():
            for batch_idx, batch in enumerate(test_reader):
                try:
                    datanodes = list(program.populate(dataset=[batch]))
                    
                    if not datanodes:
                        continue
                        
                    datanode = datanodes[0]
                    first_var_nodes = datanode.findDatanodes(select=program.graph[target_variables[0]])
                    batch_size = len(first_var_nodes) if first_var_nodes else 0
                    
                    if batch_size == 0:
                        continue

                    all_var_logits = {var_name: [] for var_name in target_variables}
                    all_var_labels = {var_name: [] for var_name in target_variables}
                    valid_variables = set()
                    
                    for var_name in target_variables:
                        if var_name in list(program.graph):
                            concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                            
                            for sample_idx in range(min(batch_size, len(concept_nodes))):
                                node = concept_nodes[sample_idx]
                                logits = node.getAttribute('logits')
                                label = node.getAttribute('label/label')
                                
                                if logits is not None and label is not None:
                                    if isinstance(logits, torch.Tensor):
                                        logits_tensor = logits
                                    else:
                                        logits_tensor = torch.tensor(logits, dtype=torch.float32)
                                    
                                    if isinstance(label, torch.Tensor):
                                        label_tensor = label
                                    else:
                                        label_tensor = torch.tensor(label, dtype=torch.long)
                                    
                                    if label_tensor.dtype != torch.long:
                                        label_tensor = label_tensor.long()
                                    
                                    all_var_logits[var_name].append(logits_tensor)
                                    all_var_labels[var_name].append(label_tensor)
                                    valid_variables.add(var_name)

                    batch_loss = 0
                    valid_var_count = 0
                    
                    for var_name in valid_variables:
                        if all_var_logits[var_name] and all_var_labels[var_name]:
                            try:
                                var_logits = torch.stack(all_var_logits[var_name])
                                var_labels = torch.stack(all_var_labels[var_name])
                                
                                var_loss = F.cross_entropy(var_logits, var_labels)
                                batch_loss += var_loss.item()
                                valid_var_count += 1
                                
                                preds = torch.argmax(var_logits, dim=1)
                                all_predictions[var_name].extend(preds.cpu().numpy())
                                all_targets[var_name].extend(var_labels.cpu().numpy())
                                var_total[var_name] += len(var_labels)
                                
                            except Exception as e:
                                print(f"Testing variable {var_name} failed: {e}")
                                continue
                    
                    if valid_var_count > 0:
                        total_loss += batch_loss
                        batch_count += 1
                        
                        avg_batch_loss = batch_loss / valid_var_count if valid_var_count > 0 else batch_loss
                        print(f" Batch {batch_idx}: Total loss={batch_loss:.4f}, Average loss={avg_batch_loss:.4f}, "
                        f" Number of variables={valid_var_count}, Number of samples={batch_size}")
                        
                except Exception as e:
                    print(f"Test batch {batch_idx} failed: {e}")
                    continue
        
        accuracies = {}
        accuracy_sum = 0
        valid_vars = 0
        
        for var_name in target_variables:
            if var_name in all_predictions and len(all_predictions[var_name]) > 0:
                preds = np.array(all_predictions[var_name])
                targets = np.array(all_targets[var_name])
                accuracy = (preds == targets).mean()
                accuracies[var_name] = accuracy
                accuracy_sum += accuracy
                valid_vars += 1
                
                correct = (preds == targets).sum()
                total = len(preds)
                var_correct[var_name] = correct
                var_total[var_name] = total
        
        avg_accuracy = accuracy_sum / valid_vars if valid_vars > 0 else 0
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        
        for var_name in target_variables:
            if var_name in accuracies:
                accuracy = accuracies[var_name]
                correct = var_correct[var_name]
                total = var_total[var_name]
                print(f" {var_name}: Accuracy {accuracy:.4f} ({correct}/{total})")
        
        print(f"Average accuracy: {avg_accuracy:.4f} (target variable average)")
        print(f"Average loss: {avg_loss:.4f}")
        
        return avg_loss, avg_accuracy, accuracies

    @staticmethod
    def manual_loss_verification(program, reader, target_variables):
        """
        Manually verify the loss calculation
        Note: The label attribute in the DataNode is accessed using 'label/label'
        
        Args:
            program: Program
            reader: Data reader
            target_variables: List of target variables
        
        Returns:
            Average loss
        """
        
        program.model.train()
        total_loss = 0
        batch_count = 0
        
        for batch in reader:
            try:
                datanodes = list(program.populate(dataset=[batch]))
                if not datanodes:
                    continue
                    
                datanode = datanodes[0]
                
                batch_loss = 0
                sample_count = 0
                
                for var_name in target_variables:
                    if var_name in list(program.graph):
                        concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                        for node in concept_nodes:
                            logits = node.getAttribute('logits')
                            label = node.getAttribute('label/label') 
                            
                            if logits is not None and label is not None:
                                loss = F.cross_entropy(logits.unsqueeze(0), label.unsqueeze(0))
                                batch_loss += loss
                                sample_count += 1
                
                if sample_count > 0:
                    avg_batch_loss = batch_loss / sample_count
                    total_loss += avg_batch_loss.item()
                    batch_count += 1
                    #print(f" Batch {batch_count}: Average loss = {avg_batch_loss.item():.4f}")
                    
            except Exception as e:
                import traceback
                print(f"Manual batch verification failed: {e}")
                error_traceback = traceback.format_exc()
                print(f"Error details:\n{error_traceback}")
                break
        
        if batch_count > 0:
            avg_total_loss = total_loss / batch_count
            print(f"Manually verifying total average loss: {avg_total_loss:.4f}")
            return avg_total_loss
        else:
            return float('inf')
        
    @staticmethod
    def create_fixed_program(graph, target_variables, **kwargs):
        poi_list = DomiKnowsPatcher.create_poi_list(graph)
        
        default_kwargs = {
            'poi': poi_list,
            'inferTypes': kwargs.get('inferTypes', ['local/softmax']),
            'loss': None,  
            'metric': None,  
        }
        
        default_kwargs.update(kwargs)
        default_kwargs['loss'] = None
        default_kwargs['metric'] = None
        
        try:
            program = FixedSolverPOIProgram(graph, **default_kwargs)
            return program
        except Exception as e:
            print(f"❌ Failed to create the fixed Program: {e}")
            try:
                fallback_kwargs = default_kwargs.copy()
                program = SolverPOIProgram(graph, **fallback_kwargs)
                return program
            except Exception as e2:
                print(f"❌ Fallback Program also failed: {e2}")
                raise
    
    @staticmethod
    def check_model_health(program, target_variables):
        
        print(f"Model training mode: {program.model.training}")
        
        total_params = 0
        trainable_params = 0
        for name, param in program.model.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                grad_info = "Have gradients" if param.grad is not None else "No gradients"
                print(f"  Params {name}: requires_grad={param.requires_grad}, {grad_info}")
        
        print(f"Total parameters: {total_params}")
        print(f"Trainable parameters: {trainable_params}")
        print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}")
        
        for var_name in target_variables:
            if var_name in list(program.graph):
                node = program.graph[var_name]
                print(f"Variable {var_name}: Exists in the graph")
            else:
                print(f"Variable {var_name}: not Exists in the graph")

    @staticmethod
    def debug_computation_graph(program, batch_data, target_variables):
        
        datanodes = list(program.populate(dataset=[batch_data]))
        datanode = datanodes[0]

        for var_name in target_variables[:1]: 
            if var_name in list(program.graph):
                concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                if concept_nodes:
                    node = concept_nodes[0]
                    logits = node.getAttribute('logits')
                    
                    print(f"Variable {var_name}:")
                    print(f"  Logits requires_grad: {logits.requires_grad}")
                    print(f"  Logits grad_fn: {logits.grad_fn}")
                    print(f"  Logits is_leaf: {logits.is_leaf}")
                    
                    raw_input = node.getAttribute(f'{var_name}_raw')
                    if raw_input is not None:
                        print(f"  Input snesor requires_grad: {raw_input.requires_grad}")
                        print(f"  Input snesor grad_fn: {raw_input.grad_fn}")
                        print(f"  Input snesor is_leaf: {raw_input.is_leaf}")

    def trace_computation_graph(program, batch_data, target_variables):

        datanodes = list(program.populate(dataset=[batch_data]))
        datanode = datanodes[0]
        
        for var_name in target_variables[:1]:  
            if var_name in list(program.graph):
                concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                if concept_nodes:
                    node = concept_nodes[0]
                    
                    learner = node.concept['logits'].sensors[0]
                    
                    if hasattr(learner, 'module'):
                        model = learner.module
                        print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")
                        
                        for name, param in model.named_parameters():
                            print(f"  Param {name}: requires_grad={param.requires_grad}, shape={param.shape}")
                            break  
                            

    @staticmethod
    def check_gradients_after_training(program, target_variables):
        
        has_gradients = False
        for name, param in program.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.norm().item()
                if grad_norm > 0:
                    print(f"✅ Parameter {name}: has gradient, norm={grad_norm:.6f}")
                    has_gradients = True
                else:
                    print(f"⚠️ Parameter {name}: Gradient is zero")
            else:
                print(f"❌ Parameter {name}: No Gradient")
        
        

    @staticmethod
    def manual_inference_analysis(program, test_reader, target_variables):
        program.model.eval()
        results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_reader):
                try:
                    datanodes = list(program.populate(dataset=[batch]))
                    
                    if not datanodes:
                        continue
                        
                    datanode = datanodes[0]
                    
                    for infer_type in ['local/softmax']:
                        try:
                            if infer_type == 'local/softmax':
                                datanode.inferLocal()
                        except Exception as e:
                            print(f"Inference type {infer_type} failed: {e}")
                    
                    batch_results = {}
                    for var_name in target_variables:
                        if var_name in list(program.graph):
                            concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                            predictions = []
                            labels = []
                            
                            for node in concept_nodes:
                                logits = node.getAttribute('logits')
                                label = node.getAttribute('label/label')
                                
                                if logits is not None:
                                    pred = logits.argmax(dim=-1).item() if isinstance(logits, torch.Tensor) else torch.tensor(logits).argmax().item()
                                    predictions.append(pred)
                                if label is not None:
                                    labels.append(label.item() if isinstance(label, torch.Tensor) else label)
                            
                            if predictions and labels:
                                batch_results[var_name] = {
                                    'predictions': predictions,
                                    'labels': labels
                                }
                    
                    results[f'batch_{batch_idx}'] = batch_results
                    
                except Exception as e:
                    print(f"Inference batch {batch_idx} failed: {e}")
                    continue
        
        return results
    
