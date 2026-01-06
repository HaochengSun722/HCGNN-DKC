# main.py
import argparse, logging, coloredlogs, torch
from pathlib import Path
from datetime import datetime
from experiment import ComparativeExperiment
from config import (CSV_FILE_PATH, BATCH_SIZE, EPOCHS,
                    PROGRAM_TYPES, INFER_TYPES, DEVICE, SEED)

coloredlogs.install(level=logging.INFO,
                    fmt='%(asctime)s %(levelname)s %(message)s')
def _set_deterministic_mode(seed=42):
    import random
    import numpy as np
    import torch
    import os
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--program', default='base', choices=list(PROGRAM_TYPES.keys()))
    p.add_argument('--infer', nargs='+', default=['local/softmax'], choices=INFER_TYPES)
    p.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    p.add_argument('--epochs', type=int, default=EPOCHS)
    p.add_argument('--out_dir', type=Path,
                   default=Path('runs') / datetime.now().strftime('%Y%m%d-%H%M%S'))
    return p.parse_args()

def main():
    #_set_deterministic_mode()
    args = get_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    logging.info('Using device: %s', DEVICE)
    logging.info('Random seed: %s', SEED)

    exp = ComparativeExperiment(csv_file_path=CSV_FILE_PATH,
                                program_type=args.program,
                                infer_types=args.infer,
                                out_dir=args.out_dir
                                )
    try:
        exp.run_all_experiments(batch_size=args.batch_size, epochs=args.epochs)
        exp.analyze_experiment_results()      
        exp.save_detailed_results()           
        exp.generate_comparison_report()      
    except Exception:
        logging.exception('Experiment failed')
        raise
    logging.info('All done, results saved to %s', args.out_dir.resolve())

if __name__ == '__main__':
    main()