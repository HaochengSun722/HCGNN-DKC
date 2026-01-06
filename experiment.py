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
from config import *
from domiknows.sensor.pytorch.relation_sensors import EdgeSensor
from domiknows.sensor.pytorch.sensors import TorchSensor
from models import *
# from domiknows.program import IMLProgram, SolverPOIProgram
# from domiknows.program.model.pytorch import SolverModel 
# from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyDictLoss
# from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric
from collections import deque
from patch_domiknows import DomiKnowsPatcher

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
            print(f"Early stop: Timer {self.counter}/{self.patience}")
            
        return self.counter >= self.patience

class ComparativeExperiment:
    """Comparative Experiment Runner"""
    def __init__(self, csv_file_path, program_type='base', infer_types=None, out_dir=None):
        self.csv_file_path = csv_file_path
        self.program_type = program_type
        self.infer_types = infer_types or ['local/softmax']
        self.results = {}
        self.out_dir = Path(out_dir) if out_dir else (Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S'))
        self.out_dir.mkdir(parents= True, exist_ok=True)


    # ---------- Creating experimental cases ----------
    def create_experiment_case(self, case_name,batch_size):
        print(f"Creating experimental cases: {case_name}")
        Graph.clear(); Concept.clear(); Relation.clear()
        with Graph(f'graph_{case_name}') as graph:
            graph['block'] = Concept(name='block')
            for var_name in ALL_VARIABLES:
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

            print('Concept list:', list(graph))
            
            # 2. Conceptual relationships are established only in cases where rule constraints are used.
            use_rules = 'rules' in case_name
            if use_rules:
                print("Construct ontological structure...")
                
                # Establish the contains relationship according to the hierarchical structure.
                graph['SBLU'].contains(graph['LU']) 
                graph['BL'].contains(graph['LU'])
                graph['SLR'].contains(graph['LU'])
                
                # LU ——BLA, BLP, BN, BD, CL, RBS, ABF 
                lu_children = ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF']
                for child in lu_children:
                    if child in list(graph):
                        graph['LU'].contains(graph[child])
                
                # BLA, BLP, BN, BD, CL, RBS, ABF —— FAR 
                far_parents = ['BLA', 'BLP', 'BN', 'BD', 'CL', 'RBS', 'ABF']
                for parent in far_parents:
                    if parent in list(graph):
                        graph[parent].contains(graph['FAR'])
                
                # FAR——HBN, MABH, HBFR, ABH, AHBH, DHBH
                far_children = ['HBN', 'MABH', 'HBFR', 'ABH', 'AHBH', 'DHBH']
                for child in far_children:
                    if child in list(graph):
                        graph['FAR'].contains(graph[child])
                
                print("Conceptual relationship established")
                
                # 3. Injecting normative domain knowledge constraints
                graph = add_spatial_constraints(graph)
                print('>>> General constraints have been injected:', len(graph.logicalConstrains))

            # 4. Connect sensors - All sensors are directly connected under EnumConcept
            reader, graph = self._declare_sensors(graph, case_name, batch_size, use_relations=use_rules)
        
        return graph, reader, case_name

    # ---------- 3. For connect sensors used ----------
    def _declare_sensors(self, graph, case_name, batch_size, use_relations=False):
        urban_reader = UrbanCSVDataReader(self.csv_file_path, batch_size=batch_size)
        use_dag = 'dag' in case_name or 'causal' in case_name


        graph['block']['index'] = UrbanFeatureSensor('block')

        input_sensors = {}
        for var_name in ALL_VARIABLES:
            if var_name not in list(graph):
                print(f"⚠️ Erro: {var_name} skip, not in cuurent graph")
                continue
                
            graph[var_name]['raw_input'] = UrbanFeatureSensor(var_name)
            input_sensors[var_name] = graph[var_name]['raw_input']

        if use_relations:
            
            block_relations = [key for key in list(graph) if key.startswith('block-contains-') and not key.endswith('.reversed')]
            
            for rel_name in block_relations:
                try:
                    # Use regular expressions to precisely parse relation names
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
                            print(f"⚠️ Skip{rel_name}: Child Concept {child_name} not exist")
                    else:
                        print(f"⚠️ Error in: {rel_name}(block relationship name format)")
                        
                except Exception as e:
                    print(f"❌ Failed in {rel_name} : {e}")

            other_relations = [key for key in list(graph) if 'contains' in key and not key.endswith('.reversed') and 'block' not in key]
            
            for rel_name in other_relations:
                try:
                    # Use regular expressions to precisely parse relation names
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
                            print(f"⚠️ Skip {rel_name}: Parent concept- {parent_name} or Child concept {child_name} not exist")
                    else:
                        print(f"⚠️ Error in: {rel_name}(relationship name format)")
                        
                except Exception as e:
                    print(f"❌ Failed in {rel_name} : {e}")

        # 3. Create independent HCGNN learners for each target variable
        print("Create target variables learner...")
        use_rules = use_relations
        
        hierarchy_map = self._build_hierarchy_map(graph, use_rules)

        for var_name in TARGET_VARIABLES:
            try:
                if 'dag' in case_name or 'causal' in case_name:
                    # case3: HCGNN-DKC
                    if use_rules and var_name in hierarchy_map:
                        hierarchy_vars = hierarchy_map[var_name]
                        print(f"{var_name} - Hierarchical: {hierarchy_vars}")
                    else:
                        hierarchy_vars = ALL_VARIABLES
                        print(f"{var_name} - All variables(this variable not in hierarchy)")
                    
                    required_inputs = [input_sensors[var] for var in hierarchy_vars if var in input_sensors]
                    
                    if not required_inputs:
                        required_inputs = [input_sensors[var] for var in ALL_VARIABLES if var in input_sensors]
                        hierarchy_vars = ALL_VARIABLES
                    
                    graph[var_name]['logits'] = HierarchicalCausalGNNLearner(
                        *required_inputs, 
                        use_dag=use_dag,
                        target_variable=var_name,
                        hierarchy_vars=hierarchy_vars
                    )
                    print(f"✅ Complete: {var_name} for HCGNN learner")
                    
                else:
                    # case1
                    required_inputs = [input_sensors[var] for var in ALL_VARIABLES if var in input_sensors]
                    
                    graph[var_name]['logits'] = SimpleGNNLearner(
                        *required_inputs,
                        target_variable=var_name
                    )
                    print(f"✅ Complete: {var_name} for simple GNN learner")
                
                graph[var_name]['label'] = UrbanLabelSensor(var_name)

            except Exception as e:
                print(f"❌ Failed in {var_name} for learner: {e}")
        
        # Validate the Architecture
        print(f"\n🔍 Architecture Validation:")
        print(f"✅ Input Sensors: {len(input_sensors)}/{len(ALL_VARIABLES)} variables")
        print(f"✅ Relationship Sensors: {'Enabled' if use_relations else 'Disabled'}")
        print(f"✅ Learner Type: {'Hierarchical Causal GNN' if ('dag' in case_name or 'causal' in case_name) else 'Simple GNN'}")
        
        return urban_reader, graph


    def run_experiment(self, case_name, batch_size=BATCH_SIZE, epochs=EPOCHS):
        print(f"\n{'='*60}")
        print(f"Start Case: {case_name}")
        print(f"Program Type: {self.program_type}")
        print(f"Inference Type: {self.infer_types}")
        print(f"{'='*60}")

        start = time.time()
        try:
            train_csv, val_csv, test_csv = self.split_csv(self.out_dir / case_name / 'split')

            train_reader = UrbanCSVDataReader(train_csv, batch_size=batch_size)
            val_reader   = UrbanCSVDataReader(val_csv,   batch_size=batch_size)
            test_reader  = UrbanCSVDataReader(test_csv,  batch_size=batch_size)
            
            # ================= For case 2 and case 4 =================
            if case_name in ['case2_dag_causal_gnn', 'case4_pure_gnn']:
                print(f"Use IndependentGNN for {case_name}")
                
                use_dag = 'dag' in case_name
                trainer = IndependentGNNTrainer(use_dag=use_dag, case_name=case_name)
                
                history = trainer.train(train_reader, val_reader, test_reader, epochs=epochs)
                
                case_dir = self.out_dir / case_name
                trainer.save_model(case_dir)
                trainer.plot_training_history(case_dir)
                
                predictions = trainer.predict(test_reader)
                
                # Return results (in the same format as the DomiKnows cases)
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
                print(f"\n {case_name} Finished! TRAINING TIME: {time.time() - start:.2f} s")
                return results
                
            # ================= For case1 and case3(DomiKnows) =================
            else:
                print(f"Use DomiknowS for {case_name}")
                graph, reader, _ = self.create_experiment_case(case_name, batch_size)
                
                # Verify the graph structure before creating the program.
                # print(f"\n🔍 Graph Structure Verification:")
                # print(f"Number of Graph Concepts: {len(list(graph.concepts))}")
                # print(f"Graph Keys: {list(graph)}")
                
                from patch_domiknows import DomiKnowsPatcher
                
                # Configure Program parameters
                program_kwargs = {
                    'inferTypes': ['local/softmax'],
                }
                
                if 'rules' in case_name and 'dag' in case_name:
                    # case 3
                    program_kwargs.update({
                        'loss': None,  # Manually calculate the loss
                        'metric': None,  
                    })
                else:
                    # case 1
                    program_kwargs.update({
                        'loss': None,  
                        'metric': None,  
                    })
                
                # Create Program
                program = DomiKnowsPatcher.create_fixed_program(
                    graph=graph,
                    target_variables=TARGET_VARIABLES,
                    **program_kwargs
                )
                
                print(f"✅ Program created successfully: {program.__class__.__name__}")
                print(f" Inference type: {program_kwargs['inferTypes']}")
                print(f" Loss function: Manually calculated")
                
                # ================= Data Validation and Debugging =================
                #test_batch = next(iter(train_reader))
                
                # Validate DataNode
                #DomiKnowsPatcher.validate_data_node_structure(program, test_batch, TARGET_VARIABLES)
                
                # Debug Sensor
                #DomiKnowsPatcher.debug_sensor_calls(program, test_batch)
                
                # ================= MANUAL TRAINING LOOP =================
                print("\nStart Manually Training...")

                optimizer = torch.optim.Adam(
                    program.model.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY
                )
                
                early_stopper = EarlyStopper(patience=EARLY_STOP_PATIENCE, mode='min')
                
                # Debug
                #DomiKnowsPatcher.debug_computation_graph(program, test_batch, TARGET_VARIABLES)
                #DomiKnowsPatcher.check_model_health(program, TARGET_VARIABLES)
                #DomiKnowsPatcher.trace_computation_graph(program, test_batch, TARGET_VARIABLES)
                #pdb.set_trace()

                history = DomiKnowsPatcher.safe_training_loop(
                    program=program,
                    train_reader=train_reader,
                    val_reader=val_reader,
                    optimizer=optimizer,
                    epochs=epochs,
                    early_stopper=early_stopper,
                    target_variables=TARGET_VARIABLES
                )

                #DomiKnowsPatcher.check_gradients_after_training(program, TARGET_VARIABLES)
                
                # ================= FINAL TEST =================
                print("\n Start Final Testing...")
                try:
                    test_loss, test_accuracy, test_acc_details = DomiKnowsPatcher.manual_test_epoch(
                        program, test_reader, TARGET_VARIABLES
                    )
                    history['final_test_loss'] = test_loss
                    history['final_test_accuracy'] = test_accuracy
                    history['test_acc_details'] = test_acc_details
                    print(f"Final loss (test): {test_loss:.4f}")
                    print(f"Final accuracy (test): {test_accuracy:.4f}")
                    
                    for var_name, acc in test_acc_details.items():
                        print(f"  {var_name}: {acc:.4f}")
                        
                except Exception as e:
                    print(f"Failed in: {e}")
                    history['final_test_loss'] = float('inf')
                    history['final_test_accuracy'] = 0.0
                    history['test_acc_details'] = {}
                
                # ================= Inference =================
                print("\nStart Inference...")
                inference_results = {}
                try:
                    inference_results = DomiKnowsPatcher.manual_inference_analysis(
                        program, test_reader, TARGET_VARIABLES
                    )
                except Exception as e:
                    print(f"Failed in: {e}")
                    inference_results = {'error': str(e)}
                
                # ================= CONFIG INFO =================
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

                # ================= SAVE MODEL =================
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
                    
                    print(f"Inference results: {inference_path}")
                    
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

                print(f"\n {case_name} Finished! TRAINING TIME: {time.time() - start:.2f} s")
                
                try:
                    self._plot_domiknows_training(case_name, program, self.out_dir / case_name)
                except Exception as e:
                    print(f"Failed to draw DomiKnows training graph: {e}")
                
                return self.results[case_name]

        except Exception as e:
            print(f"{case_name} failed in: {e}")
            import traceback
            traceback.print_exc()
            self.results[case_name] = {
                'error': str(e), 
                'execution_time': time.time() - start
            }
            return None

    # ---------- FOUR EXPERIMENT ----------
    def run_all_experiments(self, batch_size=32, epochs=50):
        for case in EXPERIMENT_CASES:
            self.run_experiment(case, batch_size, epochs)
        return self.results

    def analyze_experiment_results(self):
        print(f"\n{'='*80}")
        print(f"🔍 Experimental Comparison and Analysis Results")
        print(f"{'='*80}")
        analysis = {}
        
        for case, res in self.results.items():
            if 'error' in res:
                print(f"\n{case}: failed in - {res['error']}")
                continue
            
            # Unified processing of results for different case types
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
            'performance_comparison': analysis,
            'knowledge_contribution': self.analyze_knowledge_contribution()
        }
        out_file = self.out_dir / filename
        out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"\nComparison experiment report has been generated: {out_file.resolve()}")
        return report

    def analyze_knowledge_contribution(self):
        baseline = 'case4_pure_gnn'
        if baseline not in self.results or 'error' in self.results[baseline]:
            return 
        base_acc = self.results[baseline]['training_history'].get('test_accuracy', [0])[-1] \
                   if isinstance(self.results[baseline]['training_history'].get('test_accuracy'), list) else 0
        contrib = {}
        for case, res in self.results.items():
            if case == baseline or 'error' in res:
                continue
            acc = res['training_history'].get('test_accuracy', [0])[-1] \
                  if isinstance(res['training_history'].get('test_accuracy'), list) else 0
            sources = []
            if 'rules' in case:
                sources.append('rules')
            if 'dag' in case or 'causal' in case:
                sources.append('dag')
            contrib[case] = {
                'knowledge_sources': sources,
                'improvement': acc - base_acc,
                'baseline_accuracy': base_acc,
                'case_accuracy': acc
            }
        return contrib
    
    def split_csv(self, case_dir: Path):
        df = pd.read_csv(self.csv_file_path)
        train_val, test = train_test_split(df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_val, test_size=0.1 / 0.8, random_state=42)
        case_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(case_dir / 'train.csv', index=False)
        val.to_csv(case_dir / 'val.csv', index=False)
        test.to_csv(case_dir / 'test.csv', index=False)
        return case_dir / 'train.csv', case_dir / 'val.csv', case_dir / 'test.csv'
    
    
    #========================following is the Validation code and default function.===============================
    def validate_program_configuration(self, program, graph, poi_list, train_reader, case_name):
        """
        Verify that all parameters of the program are configured correctly.
        """
        print(f"\n{'='*80}")
        print(f" Start verifying the Program configuration for experiment {case_name}")
        print(f"{'='*80}")

        print("\n 1. Program :")
        print(f"   Program type: {type(program)}")
        print(f"   Program typename: {program.__class__.__name__}")

        print("\n 2. Model config:")
        if hasattr(program, 'model'):
            model = program.model
            print(f"   Model type: {type(model)}")
            print(f"   Model typename: {model.__class__.__name__}")
            
            model_attrs_to_check = [
                'poi', 'loss', 'metric', 'inferTypes', 'inference_with', 
                'device', 'graph', 'training', 'build'
            ]
            
            for attr in model_attrs_to_check:
                if hasattr(model, attr):
                    value = getattr(model, attr)
                    print(f"   - {attr}: {value}")
                else:
                    print(f"   - {attr}: ❌ not exist")
        else:
            print("   ❌ Model not exist!")
            return False

        print("\n3. POI config:")
        if hasattr(model, 'poi'):
            poi = model.poi
            print(f"   POI type: {type(poi)}")
            print(f"   POI length: {len(poi) if poi else 0}")
        else:
            print("   ❌ POI not exist!")

        print("\n4. Inference type config:")
        if hasattr(model, 'inferTypes'):
            infer_types = model.inferTypes
            print(f"   Inference type: {infer_types}")
            valid_infer_types = ['ILP', 'local/argmax', 'local/softmax', 'argmax', 'softmax']
            for infer_type in infer_types:
                if infer_type in valid_infer_types:
                    print(f"   ✅ {infer_type}: valid")
                else:
                    print(f"   ❌ {infer_type}: unvalid")
        else:
            print("   ❌ inferTypes not exist!")

        print("\n5. Loss config:")
        if hasattr(model, 'loss'):
            loss = model.loss
            print(f"   Loss function: {loss}")
            if isinstance(loss, torch.nn.Module):
                print(f"   ✅ valid")
            elif loss is None:
                print(f"   ⚠️ Loss = None")
            else:
                print(f"   ❓ unvalid: {type(loss)}")
        else:
            print("   ❌ loss not exist!")

        print("\n 6. Graph config:")
        if hasattr(model, 'graph'):
            graph = model.graph
            
            if hasattr(graph, 'concepts'):
                concepts = graph.concepts
                print(f"   length: {len(concepts)}")
                for name, concept in list(concepts.items())[:5]:  
                    print(f"     - {name}: {type(concept)}")
            else:
                print(f"   ⚠️ graph not have concepts")
        else:
            print("   ❌ graph not exist!")

        print("\n7. Device config:")
        if hasattr(model, 'device'):
            device = model.device
            print(f"   Divice: {device}")
        else:
            print("   ⚠️ device not exist")

        print("\n8. Sensor:")
        try:
            from domiknows.sensor.pytorch.sensors import TorchSensor
            sensors = list(graph.get_sensors(TorchSensor))
            print(f"   find {len(sensors)} sensors")
            
            for i, sensor in enumerate(sensors[:10]): 
                print(f"   - Sensor[{i}]: {sensor}")
                if hasattr(sensor, 'label'):
                    print(f"     Label Sensor: {sensor.label}")
        except Exception as e:
            print(f"   ⚠️ Sensor validation failed to: {e}")

        print("\n9. Data reader:")
        try:
            sample_data = next(iter(train_reader))
            print(f"   Sample data type: {type(sample_data)}")
            if isinstance(sample_data, dict):
                print(f"   Sample data keys: {list(sample_data.keys())}")
                for key, value in sample_data.items():
                    if hasattr(value, 'shape'):
                        print(f"     - {key}: shape{value.shape}")
                    else:
                        print(f"     - {key}: {type(value)}")
        except Exception as e:
            print(f"   ❌ Failded in: {e}")

        print("\n10. Model parms:")
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"   Total params: {total_params:,}")
            print(f"   Trainable params: {trainable_params:,}")
            print(f"   Frozen params: {total_params - trainable_params:,}")
            
            if trainable_params == 0:
                print("   ⚠️ Warning: No params!")
        except Exception as e:
            print(f"   Failed in: {e}")

        print(f"\n{'='*80}")
        print("🔍 Program Config validation complete!")
        print(f"{'='*80}")

        return True
    

    def _get_loss_value(self, loss_obj):
        """
        Unified acquisition of loss values ​​- processing DomiKnows output format
        """
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
        """
        Unified acquisition of metric values
        """
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
        """
        Plot training history (DomiKnowS cases)
        """
        if case_name not in self.results:
            return
        
        result = self.results[case_name]
        history = result.get('training_history', {})
        
        if not history.get('train_loss') or not history.get('val_loss'):
            
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
        
        print(f"DomiKnows Training history saved: {plot_path}")

    def _build_hierarchy_map(self, graph, use_rules):
        hierarchy_map = {}
        
        if not use_rules:
            for var_name in TARGET_VARIABLES:
                hierarchy_map[var_name] = ALL_VARIABLES
            return hierarchy_map
        
        hierarchy_tree = self._build_hierarchy_tree()
        
        hierarchy_config = {
            
            'LU': {'depth': 1, 'description': 'Use Parent Variables'},
            
            'BLA': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'BLP': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'BN': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'BD': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'CL': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'RBS': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'ABF': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            
            'FAR': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            
            'HBN': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'MABH': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'HBFR': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'ABH': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'AHBH': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
            'DHBH': {'depth': 2, 'description': 'Use Parent+Grandparent Variables'},
        }
        
        for target_var in TARGET_VARIABLES:
            if target_var in hierarchy_config:
                depth = hierarchy_config[target_var]['depth']
                hierarchy_vars = self._collect_hierarchy_vars(target_var, hierarchy_tree, depth)
                hierarchy_map[target_var] = hierarchy_vars
                #print(f" {target_var}: {hierarchy_config[target_var]['description']} - {len(hierarchy_vars)}")
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
            print(f"✅ The complete model state has been saved to: {model_path}")
        except Exception as e:
            print(f"❌ Failed to save complete model state: {e}")

            torch.save(program.model.state_dict(), model_path)
            print(f"✅ Model weights have been saved to: {model_path}")

        history_path = save_dir / f'{case_name}_history.json'
        try:
            json_history = {
                'train_loss': [float(x) if x != float('inf') else None for x in history.get('train_loss', [])],
                'val_loss': [float(x) if x != float('inf') else None for x in history.get('val_loss', [])],
                'val_accuracy': [float(x) for x in history.get('val_accuracy', [])],
                'epochs_completed': history.get('epochs_completed', 0),
                'final_test_loss': float(history.get('final_test_loss', 0)) if history.get('final_test_loss') != float('inf') else None,
                'final_test_accuracy': float(history.get('final_test_accuracy', 0)),
                'test_acc_details': {k: float(v) for k, v in history.get('test_acc_details', {}).items()}
            }
            
            with open(history_path, 'w') as f:
                json.dump(json_history, f, indent=2, ensure_ascii=False)
            print(f"✅ Training history has been saved to: {history_path}")
        except Exception as e:
            print(f"❌ Failed to save training history: {e}")
        
        config_path = save_dir / f'{case_name}_config.json'
        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            print(f"✅ Configuration information has been saved to: {config_path}")
        except Exception as e:
            print(f"❌ Failed to save configuration information: {e}")
        
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
                    'final_test_loss': history.get('final_test_loss', float('inf'))
                }
            }
            
            with open(detailed_path, 'w') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            print(f"✅ Detailed experimental results have been saved to: {detailed_path}")
        except Exception as e:
            print(f"❌ Failed to save detailed experimental results: {e}")

    def _collect_hierarchy_vars(self, target_var, hierarchy_tree, depth):
        if depth <= 0:
            return [target_var]
        
        collected_vars = set([target_var])
        
        # Use BFS to collect ancestors at a specified depth.
        queue = deque([(target_var, 0)])
        
        while queue:
            current_var, current_depth = queue.popleft()
            
            if current_depth >= depth:
                continue
                
            parents = hierarchy_tree.get(current_var, [])
            for parent in parents:
                if parent not in collected_vars:
                    collected_vars.add(parent)
                    queue.append((parent, current_depth + 1))
        
        return list(collected_vars)
