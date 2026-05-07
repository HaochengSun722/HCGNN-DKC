# main.py
import argparse
import logging
import coloredlogs
import torch
import os
import random
import numpy as np
from pathlib import Path
from datetime import datetime

from experiment import ComparativeExperiment
from config import (CSV_FILE_PATH, BATCH_SIZE, EPOCHS,
                    PROGRAM_TYPES, INFER_TYPES, DEVICE, SEED,
                    EXPERIMENT_CASES) 

coloredlogs.install(level=logging.INFO,
                    fmt='%(asctime)s %(levelname)s %(message)s')

def _set_deterministic_mode(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"🔒 Deterministic Mode activated, Seed: {seed}")

def get_args():
    p = argparse.ArgumentParser(description="HCGNN-DKC Main Experiments (Case 1-4)")
    p.add_argument('--program', default='base', choices=list(PROGRAM_TYPES.keys()), 
                   help="Default program type if not specified in PROGRAM_CONFIGS")
    p.add_argument('--infer', nargs='+', default=['local/softmax'], choices=INFER_TYPES)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    
    p.add_argument('--case', type=str, default='all', 
                   help="Specify a case to run (e.g., 'case3_rules_dag_causal_gnn') or 'all'")
                   
    p.add_argument('--out_dir', type=Path,
                   default=Path('runs') / f"main_experiment_{datetime.now().strftime('%Y%m%d-%H%M')}")
    return p.parse_args()

def main():
    args = get_args()
    _set_deterministic_mode(SEED)
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Using device: %s', DEVICE)
    logging.info('Output Directory: %s', args.out_dir)

    exp = ComparativeExperiment(
        csv_file_path=CSV_FILE_PATH,
        program_type=args.program,
        infer_types=args.infer,
        out_dir=args.out_dir
    )
    
    try:
        if args.case == 'all':
            logging.info(f"🚀 Start runing the full main experiment comparison (Cases: {len(EXPERIMENT_CASES)})...")
            exp.run_all_experiments(batch_size=args.batch_size, epochs=args.epochs)
        else:
            if args.case in EXPERIMENT_CASES:
                logging.info(f"Start a single experiment: {args.case}")
                exp.run_experiment(case_name=args.case, batch_size=args.batch_size, epochs=args.epochs)
            else:
                logging.error(f"Unknown Case Name: {args.case}. Please check EXPERIMENT_CASES in config.py.")
                return
        
        logging.info("Starting to generate comparative analysis report...")
        exp.analyze_experiment_results()   
        exp.save_detailed_results()     
        exp.generate_comparison_report() 
        
        logging.info('All experiments completed! Results saved to: %s', args.out_dir.resolve())
        
    except Exception:
        logging.exception('❌ Experiment failed')
        raise

if __name__ == '__main__':
    main()