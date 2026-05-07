# experiment.py
import pdb
from domiknows.graph import Graph, Concept, Relation
from domiknows.graph import EnumConcept 
from domiknows.graph import Property
import torch.nn as nn
import torch
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import pandas as pd
from IPython.display import Image
from domiknows.program.loss import NBCrossEntropyLoss

from data_reader import UrbanCSVDataReader, UrbanFeatureSensor, UrbanLabelSensor
from sensors import HierarchicalCausalGNNLearner,SimpleGNNLearner
from constraints import add_spatial_constraints
from inference import InferenceManager
from datanode_utils import DataNodeManager
from config import *
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import TorchSensor
from models import *
from domiknows.program import IMLProgram, SolverPOIProgram
from domiknows.program.model.pytorch import SolverModel 
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyDictLoss
from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric
from collections import deque
from patch_domiknows import DomiKnowsPatcher
import time
import numpy as np
from sklearn.metrics import f1_score, recall_score

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
            print(f"Early Stop: Optimal value updated to {val:.4f}")
        else:
            self.counter += 1
            print(f"Early Stop: Counter {self.counter}/{self.patience}")
            
        return self.counter >= self.patience

class ComparativeExperiment:
    """Case1 vs. Case2 vs. Case3 vs. Case4"""
    def __init__(self, csv_file_path, program_type='base', infer_types=None, out_dir=None):
        self.csv_file_path = csv_file_path
        self.program_type = program_type
        self.infer_types = infer_types or ['local/softmax']
        self.results = {}
        self.datanode_manager = DataNodeManager()
        self.inference_manager = InferenceManager()
        self.out_dir = Path(out_dir) if out_dir else (Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.out_dir.mkdir(parents= True, exist_ok=True)
        self.device = DEVICE

    def create_experiment_case(self, case_name,batch_size, **kwargs):
        print(f"Creating Case: {case_name}")
        Graph.clear(); Concept.clear(); Relation.clear()
        with Graph(f'graph_{case_name}') as graph:
            graph['block'] = Concept(name='block')
            for var_name in ALL_VARIABLES:
                # EnumConcept values
                letters = {
                    'MABH': 'A C D E F G H I J',
                    'BN': 'A C D E F G H I J',
                    'FAR': 'A C D E F G H I J',
                    'BD': 'A C D E F G H I J',
                    'BL': 'A B C D E F G H I J',
                    'LU': 'A B E G M R S U',
                    'ABF': 'A C D E F G H I J',
                    'CL': 'A C D E F G H I J',
                    'RBS': 'A C D E F G H I J',
                    'HBN': 'A F G H I J',
                    'AHBH': 'A F G H I J',
                    'ABH': 'A B C D E F G H I J',
                    'DHBH': 'A G H I J',
                    'BLA': 'A B C D E F G H I J',
                    'BLP': 'A B C D E F G H I J',
                    'HBFR': 'A F G H',
                    'SBLU': ' '.join([str(i) for i in range(66)]), 
                    'SLR': ' '.join([str(i) for i in range(31)]), 
                }.get(var_name, 'A B C D E F G H I J')
                graph[var_name] = EnumConcept(name=var_name, values=letters.split())
            
            for var_name in ALL_VARIABLES:
                if var_name in list(graph):
                    graph['block'].contains(graph[var_name])
            print('Root key list:', list(graph))
            
            use_rules = 'rules' in case_name
            if use_rules:
                print("Establishing relationships of Concepts...")
                
                # SBLU、BL、SLR → LU
                graph['SBLU'].contains(graph['LU']) 
                graph['BL'].contains(graph['LU'])
                graph['SLR'].contains(graph['LU'])
                
                # LU → BLA, BLP, BN, BD, CL, RBS, ABF
                lu_children = ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF']
                for child in lu_children:
                    if child in list(graph):
                        graph['LU'].contains(graph[child])
                
                # BLA, BLP, BN, BD, CL, RBS, ABF → FAR
                far_parents = ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF']
                for parent in far_parents:
                    if parent in list(graph):
                        graph[parent].contains(graph['FAR'])
                
                # FAR → HBN, MABH, HBFR, ABH, AHBH, DHBH
                far_children = ['HBN', 'MABH', 'HBFR', 'ABH', 'AHBH', 'DHBH']
                for child in far_children:
                    if child in list(graph):
                        graph['FAR'].contains(graph[child])
                
                print("Conceptual relationship established")
                
                graph = add_spatial_constraints(graph)
                print('>>> Total constraints:', len(graph.logicalConstrains))

            reader, graph = self._declare_sensors(graph, case_name, batch_size, use_relations=use_rules, **kwargs)
        
        return graph, reader, case_name

    def _declare_sensors(self, graph, case_name, batch_size, use_relations=False, **kwargs):
        urban_reader = UrbanCSVDataReader(self.csv_file_path, batch_size=batch_size)
        use_dag = 'dag' in case_name or 'causal' in case_name
        rev_weight = kwargs.get('reverse_weight', 0.5)
        n_type = kwargs.get('norm_type', 'left')
        c_dag = kwargs.get('custom_dag', None)
        a_flags = kwargs.get('ablation_flags', {})

        graph['block']['index'] = UrbanFeatureSensor('block')

        input_sensors = {}
        for var_name in ALL_VARIABLES:
            if var_name not in list(graph):
                print(f"⚠️ Warning: {var_name} is not in the Graph, skip")
                continue
            
            graph[var_name]['raw_input'] = UrbanFeatureSensor(var_name)
            input_sensors[var_name] = graph[var_name]['raw_input']

        if use_relations:
            
            block_relations = [key for key in list(graph) if key.startswith('block-contains-') and not key.endswith('.reversed')]
            print(f"Found {len(block_relations)} 'block' relationships")
            
            for rel_name in block_relations:
                try:
                    # Regular expressions - Matching format: block-contains-{index}-{var_name}
                    import re
                    pattern = r'^block-contains-(\d+)-([A-Z]+)$'
                    match = re.match(pattern, rel_name)
                    
                    if match:
                        index = match.group(1)
                        child_name = match.group(2) 
                        
                        if child_name in list(graph):
                            def create_block_connection(child_data, parent_data):
                                child_batch_size = child_data.size(0)  
                                parent_batch_size = parent_data.size(0) 
                                
                                # Create the Matrix: [child_batch_size, parent_batch_size, 1]
                                connections = torch.zeros(child_batch_size, parent_batch_size, 1, device=child_data.device)
                                connections[:, 0, 0] = 1 
                                
                                return connections
                            
                            graph[child_name][rel_name] = EdgeSensor(
                                graph[child_name]['raw_input'],
                                graph['block']['index'],
                                relation=graph[rel_name],
                                forward=create_block_connection
                            )
                            
                        else:
                            print(f"⚠️ Skip {rel_name}: Child concept {child_name} does not exist")
                    else:
                        print(f"⚠️ Unable to resolve 'block' relation name format: {rel_name}")
                        
                except Exception as e:
                    print(f"❌ Failed to mount 'block' relationship {rel_name}: {e}")

            other_relations = [key for key in list(graph) if 'contains' in key and not key.endswith('.reversed') and 'block' not in key]
            print(f"Found {len(other_relations)} other containment relationships")
            
            for rel_name in other_relations:
                try:
                    import re
                    pattern = r'^([A-Z]+)-contains-(\d+)-([A-Z]+)$'
                    match = re.match(pattern, rel_name)
                    
                    if match:
                        parent_name = match.group(1)
                        index = match.group(2)
                        child_name = match.group(3)
                        
                        if parent_name in list(graph) and child_name in list(graph):
                            def create_connection_matrix(child_data, parent_data):
                                child_batch_size = child_data.size(0)
                                parent_batch_size = parent_data.size(0)
                                return torch.eye(child_batch_size, parent_batch_size, device=child_data.device).unsqueeze(-1)
                            
                            graph[child_name][rel_name] = EdgeSensor(
                                graph[child_name]['raw_input'],
                                graph[parent_name]['raw_input'],
                                relation=graph[rel_name],
                                forward=create_connection_matrix
                            )
                            
                        else:
                            print(f"⚠️ Skip {rel_name}: Parent concept {parent_name} or child concept {child_name} does not exist")
                    else:
                        print(f"⚠️ Unable to resolve relation name format: {rel_name}")
                        
                except Exception as e:
                    print(f"❌ Mounting relationship {rel_name} failed: {e}")

        print("Creating a target variable learner...")
        use_rules = use_relations
        
        hierarchy_map = self._build_hierarchy_map(graph, use_rules)
        
        self.validate_hierarchy_structure(hierarchy_map)

        for var_name in TARGET_VARIABLES:
            try:
                if 'dag' in case_name or 'causal' in case_name:
                    if use_rules and var_name in hierarchy_map:
                        hierarchy_vars = hierarchy_map[var_name]
                        print(f"📊 {var_name}'s hierarchy variables: {hierarchy_vars}")
                    else:
                        hierarchy_vars = ALL_VARIABLES
                        print(f"📊 {var_name} Use all variables as input")
                    
                    required_inputs = [input_sensors[var] for var in hierarchy_vars if var in input_sensors]
                    
                    if not required_inputs:
                        required_inputs = [input_sensors[var] for var in ALL_VARIABLES if var in input_sensors]
                        hierarchy_vars = ALL_VARIABLES
                    
                    graph[var_name]['logits'] = HierarchicalCausalGNNLearner(
                        *required_inputs, 
                        use_dag=use_dag,
                        target_variable=var_name,
                        hierarchy_vars=hierarchy_vars,
                        reverse_weight=rev_weight,  
                        norm_type=n_type,           
                        custom_dag=c_dag,          
                        ablation_flags=a_flags      
                    )
                    
                else:
                    required_inputs = [input_sensors[var] for var in ALL_VARIABLES if var in input_sensors]
                    
                    graph[var_name]['logits'] = SimpleGNNLearner(
                        *required_inputs,
                        target_variable=var_name
                    )
                    
                graph[var_name]['label'] = UrbanLabelSensor(var_name)

            except Exception as e:
                print(f"❌ Failed to mount learner for target variable {var_name}: {e}")
        
        print(f"\n🔍 Architecture Validation:")
        print(f"✅ Input Sensors: {len(input_sensors)}/{len(ALL_VARIABLES)} variables")
        print(f"✅ Relation Sensors: {'Enabled' if use_relations else 'Disabled'}")
        print(f"✅ Learner Type: {'Hierarchical Causal GNN' if ('dag' in case_name or 'causal' in case_name) else 'Simple GNN'}")
        
        return urban_reader, graph

    # ---------- Run a single experiment ----------
    def run_experiment(self, case_name, batch_size=BATCH_SIZE, epochs=EPOCHS,**kwargs):
        print(f"\n{'='*60}")
        print(f"Start running the experiment: {case_name}")
        print(f"Program type: {self.program_type}")
        print(f"Inference type: {self.infer_types}")
        print(f"{'='*60}")

        start = time.time()
        try:
            train_csv, val_csv, test_csv = self.split_csv(self.out_dir / case_name / 'split')

            train_reader = UrbanCSVDataReader(train_csv, batch_size=batch_size)
            val_reader   = UrbanCSVDataReader(val_csv,   batch_size=batch_size)
            test_reader  = UrbanCSVDataReader(test_csv,  batch_size=batch_size)
            
            if case_name in ['case2_dag_causal_gnn', 'case4_pure_gnn']:
                print(f"Using a GNN trainer for {case_name}")
                
                use_dag = 'dag' in case_name
                trainer = IndependentGNNTrainer(use_dag=use_dag, case_name=case_name)
                
                history = trainer.train(train_reader, val_reader, test_reader, epochs=epochs)
                
                case_dir = self.out_dir / case_name
                trainer.save_model(case_dir)
                trainer.plot_training_history(case_dir)
                
                predictions = trainer.predict(test_reader)
                
                results = {
                    'training_history': history,
                    'test_loss': history['test_metrics']['test_loss'],
                    'test_accuracy': history['test_metrics']['test_accuracy'],
                    'test_acc_details': history['test_metrics']['test_acc_details'],
                    'predictions': predictions,
                    'trainer': trainer,
                    'execution_time': time.time() - start,
                    'program_type': 'independent',
                    'inferTypes': ['local/softmax']
                }
                
                self.results[case_name] = results
                print(f"\nExperiment {case_name} complete! Time taken: {time.time() - start:.2f} seconds")
                return results
            
            else:
                print(f"Using the DomiKnows architecture for {case_name}")
                my_kwargs = kwargs if kwargs else getattr(self, 'kwargs', {})
                a_flags = my_kwargs.get('ablation_flags', {})
                graph, reader, _ = self.create_experiment_case(case_name, batch_size,**my_kwargs)
                
                print(f"\n🔍 Graph Structure Verification:")
                print(f"Number of Graph Concepts: {len(list(graph.concepts))}")
                print(f"Graph Root Keys: {list(graph)}")
                
                from config import PROGRAM_CONFIGS
                
                base_case_key = None
                for key in PROGRAM_CONFIGS.keys():
                    if case_name.startswith(key):
                        base_case_key = key
                        break
                        
                if base_case_key:
                    program_kwargs = PROGRAM_CONFIGS[base_case_key].copy()
                    if 'model_kwargs' in program_kwargs:
                        optimal_kwargs = program_kwargs.pop('model_kwargs')
                        program_kwargs.update(optimal_kwargs)
                        print(f"🌟 Discovering and injecting optimal architecture hyperparameters: {optimal_kwargs}")

                    print(f"🔍 Matched base configuration: {base_case_key} -> type: {program_kwargs['program_type']}")
                else:
                    print(f"⚠️ No matching base configuration found, defaulting to base type")
                    program_kwargs = {'program_type': 'base', 'inferTypes': ['local/softmax']}
                
                program_kwargs.update(my_kwargs) 
                
                from patch_domiknows import DomiKnowsPatcher
                program = DomiKnowsPatcher.create_fixed_program(
                    graph=graph,
                    target_variables=TARGET_VARIABLES,
                    case_name=case_name,
                    **program_kwargs
                )
                
                print(f"\n{'*'*40}")
                print(f"🛠️ The underlying model parameters for this experiment have been confirmed:")
                print(f" - Program type: {program.__class__.__name__}")
                print(f" - Reverse causal weight: {program_kwargs.get('reverse_weight', 0.5)}")
                print(f" - Row normalization method: {program_kwargs.get('norm_type', 'left')}")
                print(f" - Semantic Beta: {program_kwargs.get('beta', 1.0) if not a_flags.get('wo_rules', False) else 0.0}")
                if a_flags:
                    print(f" - Enable architecture ablation: {list(a_flags.keys())}")
                print(f"{'*'*40}\n")
                print(f"✅ Program created successfully: {program.__class__.__name__}")
                print(f"Inference type: {program_kwargs.get('inferTypes',[])}")


                print("\nStarting manual model training...")

                optimizer = torch.optim.Adam(
                    program.model.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY
                )
                
                early_stopper = EarlyStopper(
                                patience=EARLY_STOP_PATIENCE, 
                                mode='max', 
                                min_delta=0.001
                            )
                
                # ================= Start Training =================
                print(f"Start Train {case_name}...")
                def optimizer_factory(params):
                    return torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
                    
                try:
                    train_kwargs = {
                        'training_set': train_reader,
                        'valid_set': val_reader,
                        'test_set': test_reader, 
                        'device': self.device,   
                        'train_epoch_num': epochs,
                        'Optim': optimizer_factory, 
                        'early_stopper': early_stopper
                    }
                    
                    if hasattr(program, 'cmodel') and program.cmodel is not None:
                        train_kwargs['c_warmup_iters'] = 240 
                        a_flags = my_kwargs.get('ablation_flags', {})
                        current_beta = program_kwargs.get('beta', 1.0)
                        if a_flags.get('wo_rules', False):
                            print("⚠️ [Ablation Experiment] The Semantic Loss weight (beta) has been forcibly set to 0.0 to achieve physical isolation!")
                            program.beta = 0.0  
                        else:
                            program.beta = current_beta
                            print(f"💡 Logical rules activated! Warm-up Batch Number: {train_kwargs['c_warmup_iters']}, Semantic Weights (Beta): {program.beta}")
                    else:
                        print("⚠️ The current model is the base model, with no logical constraints, and uses the standard training mode.")

                    start_time = time.time()

                    history = program.train(**train_kwargs)

                    end_time = time.time()
                    training_time_seconds = end_time - start_time
                    training_time_formatted = time.strftime("%H:%M:%S", time.gmtime(training_time_seconds))
                    
                    total_params = sum(p.numel() for p in program.model.parameters())
                    trainable_params = sum(p.numel() for p in program.model.parameters() if p.requires_grad)
                    print(f"Total training time: {training_time_formatted}")
                    print(f"Total model parameters: {total_params:,} Trainable parameters: {trainable_params:,})")
                    
                except Exception as e:
                    print(f"❌ Training loop crashed: {e}")
                    import traceback
                    traceback.print_exc()
                    raise 

                try:
                    from patch_domiknows import DomiKnowsPatcher
                    test_loss, test_acc, test_acc_details, inference_results = DomiKnowsPatcher.final_test_and_inference(
                        program, test_reader, TARGET_VARIABLES
                    )
                    
                    all_preds = {var: [] for var in TARGET_VARIABLES}
                    all_labels = {var: [] for var in TARGET_VARIABLES}

                    for batch_key, batch_data in inference_results.items():
                        if isinstance(batch_data, dict):
                            for var_name in TARGET_VARIABLES:
                                if var_name in batch_data:
                                    all_preds[var_name].extend(batch_data[var_name]['predictions'])
                                    all_labels[var_name].extend(batch_data[var_name]['labels'])

                    test_f1_details = {}
                    test_recall_details = {}

                    for var_name in TARGET_VARIABLES:
                        if len(all_preds[var_name]) > 0:
                            y_pred = np.array(all_preds[var_name])
                            y_true = np.array(all_labels[var_name])
                            
                            test_f1_details[var_name] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
                            test_recall_details[var_name] = float(recall_score(y_true, y_pred, average='macro', zero_division=0))

                    final_test_f1 = float(np.mean(list(test_f1_details.values()))) if test_f1_details else 0.0
                    final_test_recall = float(np.mean(list(test_recall_details.values()))) if test_recall_details else 0.0
                    # ======================================================================================

                    history['final_test_loss'] = test_loss
                    history['final_test_accuracy'] = test_acc
                    history['test_acc_details'] = test_acc_details
                    
                    history['final_test_f1_macro'] = final_test_f1
                    history['final_test_recall_macro'] = final_test_recall
                    history['test_f1_details'] = test_f1_details
                    history['test_recall_details'] = test_recall_details
                    
                    history['model_parameters_count'] = total_params
                    history['trainable_parameters_count'] = trainable_params
                    history['training_time_seconds'] = training_time_seconds
                    history['training_time_formatted'] = training_time_formatted
                    
                    print(f"Final Test Loss: {test_loss:.4f} | Accuracy: {test_acc:.4f} | F1: {final_test_f1:.4f} | Recall: {final_test_recall:.4f}")
                        
                except Exception as e:
                    print(f"❌ Final test failed: {e}")
                    inference_results = {'error': str(e)}
                
                config = {
                    'case_name': case_name,
                    'program_type': 'fixed_solver',
                    'inferTypes': program_kwargs['inferTypes'],
                    'target_variables': TARGET_VARIABLES,
                    'all_variables': ALL_VARIABLES,
                    'use_dag': 'dag' in case_name or 'causal' in case_name,
                    'use_rules': 'rules' in case_name,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'learning_rate': LEARNING_RATE,
                    'hidden_dim': HIDDEN_DIM,
                    'device': str(getattr(program, 'device', 'cpu')),
                    'created_time': datetime.now().isoformat()
                }

                if hasattr(self, 'kwargs'):
                    config['supplementary_test_params'] = self.kwargs

                # ================= Save model and config =================
                config = getattr(self, 'config', {}) if hasattr(self, 'config') else {}
                config.update({
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'case_name': case_name,
                    'applied_program_kwargs': program_kwargs 
                })
                try:
                    self.save_domiknows_model(
                        program=program,
                        case_name=case_name,
                        save_dir=self.out_dir / case_name,
                        history=history,
                        config=config
                    )
                    
                    inference_path = self.out_dir / case_name / f'{case_name}_inference.json'
                    with open(inference_path, 'w') as f:
                        serializable_inference = {}
                        for key, value in inference_results.items():
                            if isinstance(value, (int, float, str, bool, type(None))):
                                serializable_inference[key] = value
                            elif isinstance(value, (list, tuple)):
                                serializable_inference[key] = [float(x) if isinstance(x, (int, float)) else str(x) for x in value]
                            elif isinstance(value, dict):
                                serializable_inference[key] = {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in value.items()}
                            else:
                                serializable_inference[key] = str(value)
                        
                        json.dump(serializable_inference, f, indent=2)
                    
                    print(f"Inference results have been saved to: {inference_path}")
                    
                except Exception as save_e:
                    print(f"Failed to save model or result: {save_e}")
                    try:
                        model_path = self.out_dir / case_name / 'trained_model_state.pth'
                        model_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(program.model.state_dict(), model_path)
                        print(f"Model state has been saved to: {model_path}")
                    except Exception as fallback_e:
                        print(f"Failed to save on fallback: {fallback_e}")

                self.results[case_name] = {
                    'training_history': history,
                    'inference_results': inference_results,
                    'program': program,
                    'graph': graph,
                    'execution_time': time.time() - start,
                    'program_type': 'fixed_solver',
                    'inferTypes': program_kwargs['inferTypes'],
                    'config': config 
                }

                print(f"\nExperiment {case_name} completed! Time taken: {time.time() - start:.2f} seconds")
                
                return self.results[case_name]

        except Exception as e:
            print(f"Experiment Case {case_name} Failed: {e}")
            import traceback
            traceback.print_exc()
            self.results[case_name] = {
                'error': str(e), 
                'execution_time': time.time() - start
            }
            return None

    def run_all_experiments(self, batch_size=32, epochs=50):
        for case in EXPERIMENT_CASES:
            self.run_experiment(case, batch_size, epochs)
        return self.results

    def analyze_experiment_results(self):
        print(f"\n{'='*80}")
        print(f"Experimental Comparison")
        print(f"{'='*80}")
        analysis = {}
        
        for case, res in self.results.items():
            if 'error' in res:
                print(f"\n{case}: Failed - {res['error']}")
                continue
            
            if res.get('program_type') == 'independent':
                test_loss = res.get('test_loss', 0)
                test_accuracy = res.get('test_accuracy', 0)
                test_acc_details = res.get('test_acc_details', {})
            else:
                hist = res.get('training_history', {})
                test_loss = hist.get('final_test_loss', 0)
                test_accuracy = hist.get('final_test_accuracy', 0)
                test_acc_details = hist.get('test_acc_details', {})
            
            print(f"\n{case}:")
            print(f"Final loss: {test_loss:.4f}")
            print(f"Final accuracy: {test_accuracy:.4f}")
            print(f"Execution time: {res.get('execution_time', 0):.2f} seconds")
            
            if test_acc_details:
                for var_name, acc in test_acc_details.items():
                    print(f"    {var_name}: {acc:.4f}")
            
            analysis[case] = {
                'final_loss': test_loss, 
                'final_accuracy': test_accuracy, 
                'execution_time': res.get('execution_time', 0),
                'acc_details': test_acc_details
            }
        
        return analysis
    
    def save_detailed_results(self, filename='detailed_experiment_results.json'):
        filename = self.out_dir / filename
        out = {k: {'program_type': v.get('program_type'), 'infer_types': v.get('infer_types'),
                   'execution_time': v.get('execution_time'),
                   'training_history': {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                                        for kk, vv in v.get('training_history', {}).items()},
                   'inference_results': v.get('inference_results')}
               for k, v in self.results.items() if 'error' not in v}
        filename.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nDetailed experimental results have been saved to: {filename.resolve()}")

    def generate_comparison_report(self, filename='experiment_comparison_report.json'):
        analysis = self.analyze_experiment_results()
        report = {
            'experiment_summary': {
                'total_cases': len(EXPERIMENT_CASES),
                'successful_cases': len([r for r in self.results.values() if 'error' not in r]),
                'failed_cases': len([r for r in self.results.values() if 'error' in r]),
                'total_execution_time': sum(r.get('execution_time', 0) for r in self.results.values())
            },
            'performance_comparison': analysis
        }
        out_file = self.out_dir / filename
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nComparison experiment report has been generated: {out_file.resolve()}")
        return report
    
    def split_csv(self, case_dir: Path):
        df = pd.read_csv(self.csv_file_path)
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1 / 0.8, random_state=42)
        case_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(case_dir / 'train.csv', index=False)
        val.to_csv(case_dir / 'val.csv', index=False)
        test.to_csv(case_dir / 'test.csv', index=False)
        return case_dir / 'train.csv', case_dir / 'val.csv', case_dir / 'test.csv'
    
    #========================Default Function===============================
    def validate_program_configuration(self, program, graph, poi_list, train_reader, case_name):
        print(f"\n{'='*80}")
        print(f"🔍 Start verifying the Program configuration for experiment {case_name}")
        print(f"{'='*80}")

        print("\n 1. Program:")
        print(f"   Program type: {type(program)}")
        print(f"   Program name: {program.__class__.__name__}")

        print("\n 2. Model:")
        if hasattr(program, 'model'):
            model = program.model
            print(f"   Model type: {type(model)}")
            print(f"   Model name: {model.__class__.__name__}")
            
            model_attrs_to_check = [
                'poi', 'loss', 'metric', 'inferTypes', 'inference_with', 
                'device', 'graph', 'training', 'build'
            ]
            
            for attr in model_attrs_to_check:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    print(f"   - {attr}: {value}")
                else:
                    print(f"   - {attr}: Not exist")
        else:
            print("   Model not exist!")
            return False

        print("\n 3. POI:")
        if hasattr(model, 'poi'):
            poi = model.poi
            print(f"   POI type: {type(poi)}")
            print(f"   POI length: {len(poi) if poi else 0}")
            
            if poi:
                for i, item in enumerate(poi):
                    print(f"   - POI[{i}]: {type(item)} - {item}")
                    if hasattr(item, 'name'):
                        print(f"     Name: {item.name}")
                    if hasattr(item, 'fullname'):
                        print(f"     Fullname: {item.fullname}")
            else:
                print("   POI is empty!")
        else:
            print("   POI attributes not exist!")

        print("\n 4. Inference types:")
        if hasattr(model, 'inferTypes'):
            infer_types = model.inferTypes
            print(f"   InferTypes: {infer_types}")
            valid_infer_types = ['ILP', 'local/argmax', 'local/softmax', 'argmax', 'softmax']
            for infer_type in infer_types:
                if infer_type in valid_infer_types:
                    print(f"   ✅ {infer_type}: Valided")
                else:
                    print(f"   ❌ {infer_type}: Not valided")
        else:
            print("   inferTypes not exist!")

        print("\n 5. Loss function:")
        if hasattr(model, 'loss'):
            loss = model.loss
            print(f"   Loss function: {loss}")
            if isinstance(loss, torch.nn.Module):
                print(f"   ✅ Loss function: Valided")
            elif loss is None:
                print(f"   ⚠️ Loss function: None")
            else:
                print(f"   ❓ Loss function unknown: {type(loss)}")
        else:
            print("   ❌ Loss function not exist!")

        print("\n 6. Evaluation:")
        if hasattr(model, 'metric'):
            metric = model.metric
            print(f"   Evaluation: {metric}")
            if isinstance(metric, dict):
                print(f"   ✅ Metric is dict type")
                for key, value in metric.items():
                    print(f"     - {key}: {value}")
            elif metric is None:
                print(f"   ⚠️ Metric is None")
            else:
                print(f"   ❓ Metric is {type(metric)}")
        else:
            print("   ❌ metric not exist!")

        print("\n 7. Graph structure:")
        if hasattr(model, 'graph'):
            graph = model.graph
            
            if hasattr(graph, 'concepts'):
                concepts = graph.concepts
                print(f"   Graph length: {len(concepts)}")
                for name, concept in list(concepts.items())[:5]: 
                    print(f"     - {name}: {type(concept)}")
            else:
                print(f" ⚠️ Unable to retrieve the list of concepts for the graph")
        else:
            print("   ❌ graph attributes not exist!")

        print("\n 8. Device:")
        if hasattr(model, 'device'):
            device = model.device
            print(f"   Device config: {device}")
        else:
            print("   ⚠️ device attribute not exist")

        print("\n 9. Sensor:")
        try:
            from domiknows.sensor.pytorch.sensors import TorchSensor
            sensors = list(graph.get_sensors(TorchSensor))
            print(f"   Find {len(sensors)} sensors")
            
            for i, sensor in enumerate(sensors[:10]):  
                print(f"   - Sensor[{i}]: {sensor}")
                if hasattr(sensor, 'concept'):
                    print(f"     Related concept: {sensor.concept}")
                if hasattr(sensor, 'label'):
                    print(f"     Label sensor: {sensor.label}")
        except Exception as e:
            print(f"   ⚠️ Sensor error: {e}")

        print(f"\n{'='*80}")
        print("Program config evaluation complete")
        print(f"{'='*80}")

        return True
    

    def _get_loss_value(self, loss_obj):
        if loss_obj is None:
            return float('inf')
        
        if isinstance(loss_obj, tuple):
            if len(loss_obj) > 0:
                first_element = loss_obj[0]
                if hasattr(first_element, 'item'):
                    return first_element.item()
                elif isinstance(first_element, (int, float)):
                    return float(first_element)
                else:
                    return self._get_loss_value(first_element)
            else:
                return float('inf')
        
        if hasattr(loss_obj, 'item'):
            return loss_obj.item()
        elif hasattr(loss_obj, 'value'):
            return loss_obj.value()
        else:
            try:
                return float(loss_obj)
            except (TypeError, ValueError):
                return float('inf')

    def _get_metric_value(self, metric_obj):
        if metric_obj is None:
            return 0.0
        
        if hasattr(metric_obj, 'item'):
            return metric_obj.item()
        elif hasattr(metric_obj, 'value'):
            return metric_obj.value()
        elif hasattr(metric_obj, 'copy'):
            metric_value = metric_obj.copy()
            if isinstance(metric_value, dict):
                for key in ['accuracy', 'acc', 'f1']:
                    if key in metric_value:
                        return metric_value[key]
                for val in metric_value.values():
                    if isinstance(val, (int, float)):
                        return val
            return 0.0
        else:
            try:
                return float(metric_obj)
            except (TypeError, ValueError):
                return 0.0

    def _plot_domiknows_training(self, case_name, program, save_dir):
        if case_name not in self.results:
            return
        
        result = self.results[case_name]
        history = result.get('training_history', {})
        
        if not history.get('train_loss') or not history.get('val_loss'):
            print(f"Unable to plot a graph for {case_name}: Training history data is missing")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training loss')
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation loss')
        ax1.set_title(f'{case_name} - Loss curve')
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
        
        plot_path = save_dir / f'{case_name}_training_plot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"DomiKnows training charts have been saved to: {plot_path}")

    def _build_hierarchy_map(self, graph, use_rules):
        hierarchy_map = {}
        
        if not use_rules:
            for var_name in TARGET_VARIABLES:
                hierarchy_map[var_name] = ALL_VARIABLES
            return hierarchy_map
        
        hierarchy_tree = self._build_hierarchy_tree()
        
        hierarchy_config = {
            
            'LU': {'depth': 1, 'description': 'Use parent variables'},
            
            'BLA': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'BLP': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'BN': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'BD': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'CL': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'RBS': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'ABF': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            
            'FAR': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            
            'HBN': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'MABH': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'HBFR': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'ABH': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'AHBH': {'depth': 2, 'description': 'Use grandparent and parent variables'},
            'DHBH': {'depth': 2, 'description': 'Use grandparent and parent variables'},
        }
        
        for target_var in TARGET_VARIABLES:
            if target_var in hierarchy_config:
                depth = hierarchy_config[target_var]['depth']
                hierarchy_vars = self._collect_hierarchy_vars(target_var, hierarchy_tree, depth)
                hierarchy_map[target_var] = hierarchy_vars
                #print(f"📊 {target_var}: {hierarchy_config[target_var]['description']} - {len(hierarchy_vars)} variables")
            else:
                hierarchy_map[target_var] = ALL_VARIABLES
                print(f"⚠️ {target_var}: Hierarchy depth not configured, using all variables")
        
        #hierarchy_map = {key: [item for item in value_list if item != key] for key, value_list in hierarchy_map.items()}
        #hierarchy_map = self._add_same_level_variables(hierarchy_map, hierarchy_tree)
        return hierarchy_map
    
    def _add_same_level_variables(self, hierarchy_map, hierarchy_tree):
        
        level_groups = {
            'level1': ['LU'],
            'level2': ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF'],
            'level3': ['FAR'],
            'level4': ['HBN', 'MABH', 'HBFR', 'ABH', 'AHBH', 'DHBH']
        }
        
        var_to_level = {}
        for level, vars_list in level_groups.items():
            for var in vars_list:
                var_to_level[var] = level
        
        level_expansion_config = {
            'BN': level_groups['level2'],
            'BLA': level_groups['level2'], 
            'BLP': level_groups['level2'],
            'BD': level_groups['level2'],
            'CL': level_groups['level2'],
            'RBS': level_groups['level2'],
            'ABF': level_groups['level2'],
            
            'FAR': level_groups['level4'],
            
            'MABH': level_groups['level4'],
            'HBN': level_groups['level4'],
            'HBFR': level_groups['level4'],
            'ABH': level_groups['level4'],
            'AHBH': level_groups['level4'],
            'DHBH': level_groups['level4'],
        }
        
        expanded_hierarchy_map = {}
        for target_var, current_vars in hierarchy_map.items():
            expanded_vars = set(current_vars)
            
            if target_var in level_expansion_config:
                vars_to_add = level_expansion_config[target_var]
                for var in vars_to_add:
                    if var != target_var:  
                        expanded_vars.add(var)
            
            expanded_hierarchy_map[target_var] = list(expanded_vars)
        
        for target_var, vars_list in expanded_hierarchy_map.items():
            original_vars = set(hierarchy_map[target_var])
            expanded_vars = set(vars_list)
            added_vars = expanded_vars - original_vars
            
            if added_vars:
                print(f"{target_var}: Added {len(added_vars)} sibling variables: {list(added_vars)}")
        
        return expanded_hierarchy_map

    def _build_hierarchy_tree(self):
        hierarchy_tree = {}
        
        # SBLU、BL、SLR → LU
        hierarchy_tree['LU'] = ['BL', 'SLR','SBLU']
        
        # LU → BLA, BLP, BN, BD, CL, RBS, ABF
        hierarchy_tree['BLA'] = ['LU']
        hierarchy_tree['BLP'] = ['LU']
        hierarchy_tree['BN'] = ['LU']
        hierarchy_tree['BD'] = ['LU']
        hierarchy_tree['CL'] = ['LU']
        hierarchy_tree['RBS'] = ['LU']
        hierarchy_tree['ABF'] = ['LU']
        
        # BLA, BLP, BN, BD, CL, RBS, ABF → FAR
        hierarchy_tree['FAR'] = ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF']
        
        # FAR → HBN, MABH, HBFR, ABH, AHBH, DHBH
        hierarchy_tree['HBN'] = ['FAR']
        hierarchy_tree['MABH'] = ['FAR']
        hierarchy_tree['HBFR'] = ['FAR']
        hierarchy_tree['ABH'] = ['FAR']
        hierarchy_tree['AHBH'] = ['FAR']
        hierarchy_tree['DHBH'] = ['FAR']
        
        # Root
        hierarchy_tree['BL'] = []
        hierarchy_tree['SBLU'] = []
        hierarchy_tree['SLR'] = []
        
        return hierarchy_tree

    def save_domiknows_model(self, program, case_name, save_dir, history, config):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = save_dir / f'{case_name}_model.pth'
        try:
            torch.save({
                'model_state_dict': program.model.state_dict(),
                'training_history': history,
                'config': config,
                'program_type': 'domiknows',
                'case_name': case_name,
                'saved_time': datetime.now().isoformat()
            }, model_path)
        except Exception as e:
            print(f"❌ Save model state error: {e}")
            torch.save(program.model.state_dict(), model_path)
        
        history_path = save_dir / f'{case_name}_history.json'
        try:
            json_history = {
                'train_loss': [float(x) if x != float('inf') else None for x in history.get('train_loss', [])],
                'val_loss': [float(x) if x != float('inf') else None for x in history.get('val_loss', [])],
                'val_accuracy': [float(x) for x in history.get('val_accuracy', [])],
                'epochs_completed': history.get('epochs_completed', 0),
                'final_test_loss': float(history.get('final_test_loss', 0)) if history.get('final_test_loss') != float('inf') else None,
                'final_test_accuracy': float(history.get('final_test_accuracy', 0)),
                'test_acc_details': {k: float(v) for k, v in history.get('test_acc_details', {}).items()},
                'final_test_f1_macro': float(history.get('final_test_f1_macro', 0.0)),
                'final_test_recall_macro': float(history.get('final_test_recall_macro', 0.0)),
                'test_f1_details': {k: float(v) for k, v in history.get('test_f1_details', {}).items()},
                'test_recall_details': {k: float(v) for k, v in history.get('test_recall_details', {}).items()},
                'model_parameters_count': history.get('model_parameters_count', 0),
                'trainable_parameters_count': history.get('trainable_parameters_count', 0),
                'training_time_seconds': float(history.get('training_time_seconds', 0.0)),
                'training_time_formatted': str(history.get('training_time_formatted', ""))
            }
            
            with open(history_path, 'w') as f:
                json.dump(json_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Save model history error: {e}")
        
        config_path = save_dir / f'{case_name}_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Save model config error: {e}")
        
        detailed_path = save_dir / 'detailed_experiment_results.json'
        try:
            detailed_results = {
                'case_name': case_name,
                'training_history': json_history, 
                'config': config,
                'saved_time': datetime.now().isoformat(),
                'metrics_summary': {
                    'best_val_accuracy': max(history.get('val_accuracy', [0])),
                    'final_test_accuracy': history.get('final_test_accuracy', 0),
                    'final_test_f1_macro': history.get('final_test_f1_macro', 0),
                    'final_test_recall_macro': history.get('final_test_recall_macro', 0),
                    'final_test_loss': history.get('final_test_loss', float('inf'))
                }
            }
            
            with open(detailed_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Save detailed results error:: {e}")

    def _collect_hierarchy_vars(self, target_var, hierarchy_tree, depth):
        if depth <= 0:
            return [target_var]
        
        from collections import deque
        
        collected_vars = {target_var: None}
        queue = deque([(target_var, 0)])
        
        while queue:
            current_var, current_depth = queue.popleft()
            if current_depth >= depth:
                continue
            parents = hierarchy_tree.get(current_var, [])
            for parent in parents:
                if parent not in collected_vars:
                    collected_vars[parent] = None  
                    queue.append((parent, current_depth + 1))
        
        return list(collected_vars.keys())
