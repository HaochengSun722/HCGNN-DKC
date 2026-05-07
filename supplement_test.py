# supplement_test.py
import argparse
import random
import json
from pathlib import Path
from datetime import datetime
from config import DAG_EDGES, ALL_VARIABLES
from experiment import ComparativeExperiment

def is_acyclic(edges, nodes):
    adj = {n:[] for n in nodes}
    for u, v in edges:
        adj[u].append(v)
    visited = {n: 0 for n in nodes} # 0: unvisited, 1: visiting, 2: visited
    def dfs(node):
        if visited[node] == 1: return False # Cycle detected
        if visited[node] == 2: return True
        visited[node] = 1
        for neighbor in adj[node]:
            if not dfs(neighbor): return False
        visited[node] = 2
        return True
    return all(dfs(n) for n in nodes)

def drop_dag_edges(edges, drop_ratio=0.2):
    keep_num = int(len(edges) * (1 - drop_ratio))
    return random.sample(edges, keep_num)

def modify_dag_edges(edges, nodes, modify_ratio=0.2):
    new_edges = list(edges)
    num_to_modify = int(len(edges) * modify_ratio)
    
    for _ in range(num_to_modify):
        if not new_edges: break
        idx = random.randrange(len(new_edges))
        new_edges.pop(idx)
        for _ in range(10):
            u, v = random.sample(nodes, 2)
            if (u, v) not in new_edges:
                new_edges.append((u, v))
                if is_acyclic(new_edges, nodes):
                    break
                else:
                    new_edges.pop() 
    return new_edges

class SupplementRunner:
    def __init__(self, base_out_dir="runs/supplementary_tests"):
        self.base_dir = Path(base_out_dir) / datetime.now().strftime('%Y%m%d-%H%M')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = "data/1061samples_int.csv"

    def run_single_test(self, test_group, test_name, case_base="case3_rules_dag_causal_gnn", kwargs_dict=None):
        if kwargs_dict is None:
            kwargs_dict = {}
        print(f"\n{'='*60}\nStart suppement test[{test_group}]: {test_name}\n{'='*60}")
        out_dir = self.base_dir / test_group
        
        exp = ComparativeExperiment(csv_file_path=self.csv_path, out_dir=out_dir)
        
        safe_test_name = test_name.replace('.', '_')
        
        actual_case_base = case_base

        unique_case_name = f"{actual_case_base}_{safe_test_name}"
        
        exp.kwargs = kwargs_dict

        try:
            result = exp.run_experiment(
                case_name=unique_case_name, 
                batch_size=32, 
                epochs=100,
                **kwargs_dict
            )
            
            acc = result['training_history'].get('final_test_accuracy', 0.0)
            print(f"✅[{test_name}] finish! Final accuracy: {acc:.4f}")
            
            if test_group not in self.results_summary:
                self.results_summary[test_group] = {}
            self.results_summary[test_group][test_name] = acc
            
        except Exception as e:
            print(f"❌ [{test_name}] faild: {e}")
            import traceback
            traceback.print_exc()

def run_task_1(runner):
    print(">>> Task 1: Sensitivity Analysis of Inverse Weights <<<")
    weights =[0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
    for w in weights:
        runner.run_single_test("1_ReverseWeight", f"weight_{w}", kwargs_dict={'reverse_weight': w})

def run_task_2(runner):
    print(">>> Task 2: Sensitivity Analysis of Three Normalization Methods <<<")
    norms = ['left', 'right', 'symmetric'
             ]
    for n in norms:
        runner.run_single_test("2_Normalization", f"norm_{n}", kwargs_dict={'norm_type': n})

def run_task_3(runner):
    print(">>> Task 3: Sensitivity Analysis of Expert DAG <<<")
    tests = {
        'Expert_DAG': DAG_EDGES,
        'Ontology_Only_DAG': [], 
        'Random_DAG_1': modify_dag_edges([], ALL_VARIABLES, modify_ratio=1.0),
        'Drop_20pct_Edges': drop_dag_edges(DAG_EDGES, 0.2),
        'Modify_20pct_Edges': modify_dag_edges(DAG_EDGES, ALL_VARIABLES, 0.2)
    }
    for name, dag in tests.items():
        runner.run_single_test("3_DAG_Sensitivity", name, kwargs_dict={'custom_dag': dag})

def run_task_4(runner):
    print(">>> Task 4: Ablation Experiment <<<")
    ablations = {
        'Full_HCGNN_DKC': {},
        'wo_Independent_MLP': {'wo_mlp': True}
        ,'wo_Residual_Bridge': {'wo_residual': True},
        'wo_Attention': {'wo_attention': True},
        'wo_Ontology_OEM': {'wo_ontology': True}
        ,'wo_Semantic_Rules': {'wo_rules': True}  
    }
    for name, flags in ablations.items():
        runner.run_single_test("4_Comprehensive_Ablation", name, kwargs_dict={'ablation_flags': flags} )

def run_task_5(runner):
    print(">>> Task 5: Sensitivity Analysis of Semantic Loss Weights (Beta) <<<")
    betas =[0.0,0.1, 0.3, 0.5, 0.7, 0.9]
    for b in betas:
        runner.run_single_test(
            test_group="5_SemanticBetaWeight", 
            test_name=f"beta_{b}",
            kwargs_dict={'beta': b}
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, choices=[0, 1, 2, 3, 4, 5], required=True, 
                        help="Select task to run: 0(All Tasks), 1(Weight), 2(Norm), 3(DAG), 4(Ablation), 5(Beta)")
    args = parser.parse_args()
    
    runner = SupplementRunner()
    
    if args.task == 1 or args.task == 0: 
        print("\\n▶️ Start Task 1...")
        run_task_1(runner)
    if args.task == 2 or args.task == 0: 
        print("\\n▶️ Start Task 2...")
        run_task_2(runner)
    if args.task == 3 or args.task == 0: 
        print("\\n▶️ Start Task 3...")
        run_task_3(runner)
    if args.task == 4 or args.task == 0: 
        print("\\n▶️ Start Task 4...")
        run_task_4(runner)
    if args.task == 5 or args.task == 0: 
        print("\\n▶️ Start Task 5...")
        run_task_5(runner)
        
    if args.task == 0:
        print("\\n All Tasks completed！")