# config.py
from pathlib import Path
import torch
from domiknows.program.loss import NBCrossEntropyLoss, NBCrossEntropyDictLoss
from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric

# ---------- BASE ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
torch.manual_seed(SEED)

# ---------- PATH ----------
DATA_ROOT = Path('data')
CSV_FILE_PATH = DATA_ROOT / '1061samples_int.csv'

# ---------- VARIABLES ----------
ALL_VARIABLES = ['BLA', 'BLP', 'MABH', 'BN', 'FAR', 'BD',
                 'HBN', 'AHBH', 'ABH', 'DHBH', 'ABF', 'CL',
                 'RBS', 'HBFR', 'BL', 'LU', 'SBLU', 'SLR']

TARGET_VARIABLES = ['MABH', 'BN', 'FAR', 'BD', 'HBN', 'AHBH',
                    'ABH', 'DHBH', 'ABF', 'CL', 'RBS', 'HBFR']

NUM_CLASSES_DICT = {
    'MABH': 9, 'BN': 9, 'FAR': 9, 'BD': 9, 'HBN': 6, 'AHBH': 6,
    'ABH': 10, 'DHBH': 5, 'ABF': 9, 'CL': 9, 'RBS': 9, 'HBFR': 4,
    'BL': 10, 'LU': 8, 'SBLU': 61, 'SLR': 31
}

DAG_EDGES = [
    ('BLA', 'BLP'), ('MABH', 'DHBH'), ('BN', 'BD'),
    ('SLR', 'HBFR'), ('SLR', 'RBS'), ('SLR', 'FAR'),
    ('BD', 'ABF'), ('BD', 'CL'), ('FAR', 'BLA'),
    ('FAR', 'HBN'), ('FAR', 'BN'), ('FAR', 'AHBH'), ('FAR', 'BD'),
    ('FAR', 'DHBH'), ('ABH', 'DHBH'), ('LU', 'BLA'), ('LU', 'BN'),
    ('LU', 'FAR'), ('LU', 'CL'), ('LU', 'RBS'),
    ('CL', 'RBS'), ('CL', 'HBFR'), ('BL', 'FAR'), ('BL', 'DHBH'),
    ('SBLU', 'LU'), ('BL', 'MABH'), ('LU', 'ABF'), ('FAR', 'ABH'),
    ('BL', 'SBLU'), ('BL', 'HBN')
]

# TRAIN
BATCH_SIZE = 32
EPOCHS = 200
HIDDEN_DIM = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY    = 1e-5
EARLY_STOP_PATIENCE = 20

# INFERENCE
INFER_TYPES = ['local/softmax', 'local/argmax', 'ILP', 'GBI']
PROGRAM_TYPES = {'base': 'SolverPOIProgram',
                 'iml': 'IMLProgram',
                 'primal_dual': 'PrimalDualProgram',
                 'gbi': 'GBIProgram'}

# ILP / GBI CONFIG
ILP_CONFIG = {'epsilon': 1e-5, 'minimizeObjective': False}

# 4 CASE EXPERIMENTS
EXPERIMENT_CASES = [
    'case1_rules_pure_gnn',
    'case2_dag_causal_gnn',
    'case3_rules_dag_causal_gnn'
    ,'case4_pure_gnn'
]

# DEFALUT CONFIG
PROGRAM_CONFIGS = {
    'case1_rules_pure_gnn': {
        'program_type': 'iml',
        'inferTypes': ['ILP', 'local/softmax']
    },
    'case2_dag_causal_gnn': {
        'program_type': 'base', 
        'inferTypes': ['local/softmax']
    },
    'case3_rules_dag_causal_gnn': {
        'program_type': 'iml',
        'inferTypes': ['ILP', 'local/softmax']
    },
    'case4_pure_gnn': {
        'program_type': 'base',
        'inferTypes': ['local/softmax'] 
    }
}