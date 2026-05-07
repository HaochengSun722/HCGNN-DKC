"""
DomiKnowS Patch Files
"""

import torch
import torch.nn.functional as F
import numpy as np
import types
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from domiknows.program.model.pytorch import SolverModel,PoiModel
from domiknows.program.loss import NBCrossEntropyLoss
from domiknows.program.metric import PRF1Tracker, DatanodeCMMetric
from domiknows.sensor.pytorch.sensors import TorchSensor
from domiknows.program.lossprogram import PrimalDualProgram
from domiknows.program.model_program import POIProgram, IMLProgram
from config import *
from domiknows.program.model.base import Mode


def get_len(dataset):
    try: return len(dataset)
    except TypeError: return None

class NativeSemanticLoss:
    SEMANTIC_SCALE = 20.0 

    @staticmethod
    def get_groups(probs):
        g = {}
        if 'BL' in probs:
            g['central'] = probs['BL'][:,[0,1,2,3]].sum(dim=1)
            g['edge'] = probs['BL'][:,[7,8,9]].sum(dim=1)
        if 'MABH' in probs:
            g['MABH_low'] = probs['MABH'][:, [0,1,2]].sum(dim=1)
            g['MABH_med'] = probs['MABH'][:,[3,4,5]].sum(dim=1)
            g['MABH_hig'] = probs['MABH'][:,[6,7,8]].sum(dim=1)
        if 'ABH' in probs:
            g['ABH_low'] = probs['ABH'][:, [0,1,2]].sum(dim=1)
            g['ABH_med'] = probs['ABH'][:, [3,4,5]].sum(dim=1)
            g['ABH_hig'] = probs['ABH'][:,[6,7,8,9]].sum(dim=1)
        if 'AHBH' in probs:
            g['AHBH_low'] = probs['AHBH'][:, [0,1,2]].sum(dim=1)
            g['AHBH_hig'] = probs['AHBH'][:,[3,4,5]].sum(dim=1)
        if 'DHBH' in probs:
            g['DHBH_low'] = probs['DHBH'][:, [0,1]].sum(dim=1)
            g['DHBH_hig'] = probs['DHBH'][:, [2,3,4]].sum(dim=1)
        if 'BD' in probs:
            g['BD_low'] = probs['BD'][:, [0,1,2]].sum(dim=1)
            g['BD_med'] = probs['BD'][:,[3,4,5]].sum(dim=1)
            g['BD_hig'] = probs['BD'][:,[6,7,8]].sum(dim=1)
        if 'FAR' in probs:
            g['FAR_low'] = probs['FAR'][:, [0,1,2]].sum(dim=1)
            g['FAR_med'] = probs['FAR'][:, [3,4,5]].sum(dim=1)
            g['FAR_hig'] = probs['FAR'][:, [6,7,8]].sum(dim=1)
        if 'ABF' in probs:
            g['ABF_low'] = probs['ABF'][:, [0,1,2]].sum(dim=1)
            g['ABF_med'] = probs['ABF'][:, [3,4,5]].sum(dim=1)
            g['ABF_hig'] = probs['ABF'][:, [6,7,8]].sum(dim=1)
        if 'CL' in probs:
            g['CL_low'] = probs['CL'][:, [0,1,2]].sum(dim=1)
            g['CL_med'] = probs['CL'][:, [3,4,5]].sum(dim=1)
            g['CL_hig'] = probs['CL'][:,[6,7,8]].sum(dim=1)
        if 'RBS' in probs:
            g['RBS_low'] = probs['RBS'][:, [0,1,2]].sum(dim=1)
            g['RBS_med'] = probs['RBS'][:, [3,4,5]].sum(dim=1)
            g['RBS_hig'] = probs['RBS'][:, [6,7,8]].sum(dim=1)
        if 'HBFR' in probs:
            g['HBFR_low'] = probs['HBFR'][:, [0,1]].sum(dim=1)
            g['HBFR_med'] = probs['HBFR'][:, [2]].sum(dim=1)
            g['HBFR_hig'] = probs['HBFR'][:, [3]].sum(dim=1)
        if 'HBN' in probs:
            g['HBN_low'] = probs['HBN'][:,[0,1,2]].sum(dim=1)
            g['HBN_med'] = probs['HBN'][:,[3]].sum(dim=1)
            g['HBN_hig'] = probs['HBN'][:, [4,5]].sum(dim=1)
        if 'BN' in probs:
            g['BN_low'] = probs['BN'][:, [0,1,2]].sum(dim=1)
            g['BN_med'] = probs['BN'][:, [3,4,5]].sum(dim=1)
            g['BN_hig'] = probs['BN'][:, [6,7,8]].sum(dim=1)
        if 'BLA' in probs:
            g['BLA_small'] = probs['BLA'][:, [0,1,2]].sum(dim=1)
            g['BLA_med'] = probs['BLA'][:,[3,4,5]].sum(dim=1)
            g['BLA_large'] = probs['BLA'][:,[6,7,8]].sum(dim=1)
        if 'LU' in probs:
            g['public'] = probs['LU'][:, 0]
            g['commercial'] = probs['LU'][:, 1]
            g['water'] = probs['LU'][:, 2]
            g['green'] = probs['LU'][:, 3]
            g['industrial'] = probs['LU'][:, 4]
            g['residential'] = probs['LU'][:, 5]
            g['utility'] = probs['LU'][:, 7]
        return g

    @staticmethod
    def T_and(*args):
        res = args[0]
        for a in args[1:]: res = torch.clamp(res + a - 1.0, 0.0, 1.0)
        return res

    @staticmethod
    def T_or(*args):
        res = args[0]
        for a in args[1:]: res = torch.clamp(res + a, 0.0, 1.0)
        return res

    @staticmethod
    def T_imp(a, b): return torch.clamp(1.0 - a + b, 0.0, 1.0)

    @staticmethod
    def T_not(a): return 1.0 - a

    @classmethod
    def compute(cls, probs):
        g = cls.get_groups(probs)
        losses =[]

        def add_rule(val, weight):
            if val is not None:
                violation = torch.relu(1.0 - val)
                losses.append(violation.mean() * weight)

        try:
            if all(k in g for k in ['MABH_hig', 'BD_hig', 'FAR_hig']):
                add_rule(cls.T_imp(cls.T_and(g['MABH_hig'], g['BD_hig']), g['FAR_hig']), 0.75)
            if all(k in g for k in['ABH_hig', 'FAR_med', 'FAR_hig']):
                add_rule(cls.T_imp(g['ABH_hig'], cls.T_or(g['FAR_med'], g['FAR_hig'])), 0.70)
            if all(k in g for k in ['MABH_hig', 'FAR_low']):
                add_rule(cls.T_imp(g['MABH_hig'], cls.T_not(g['FAR_low'])), 0.80)
            if all(k in g for k in['FAR_hig', 'BD_low']):
                add_rule(cls.T_imp(g['FAR_hig'], cls.T_not(g['BD_low'])), 0.75)
            if all(k in g for k in ['BN_hig', 'BD_med', 'BD_hig']):
                add_rule(cls.T_imp(g['BN_hig'], cls.T_or(g['BD_med'], g['BD_hig'])), 0.70)
            if all(k in g for k in ['RBS_hig', 'CL_hig']):
                add_rule(cls.T_imp(g['RBS_hig'], g['CL_hig']), 0.65)
            
            if all(k in g for k in['central', 'MABH_hig', 'FAR_hig', 'ABH_hig']):
                add_rule(cls.T_imp(g['central'], cls.T_and(g['MABH_hig'], g['FAR_hig'], g['ABH_hig'])), 0.80)
            if all(k in g for k in['green', 'water', 'FAR_low', 'BD_low', 'ABF_low']):
                eco = cls.T_or(g['green'], g['water'])
                add_rule(cls.T_imp(eco, cls.T_and(g['FAR_low'], g['BD_low'], g['ABF_low'])), 0.95)
            if all(k in g for k in['commercial', 'HBFR_low', 'RBS_hig']):
                add_rule(cls.T_imp(g['commercial'], cls.T_and(g['HBFR_low'], g['RBS_hig'])), 0.75)
            if all(k in g for k in['BLA_large', 'BD_med', 'BD_low', 'RBS_med']):
                add_rule(cls.T_imp(g['BLA_large'], cls.T_and(cls.T_or(g['BD_med'], g['BD_low']), g['RBS_med'])), 0.75)
                
        except Exception as e:
            print(f"⚠️ Semantic Loss Calculation Error: {e}")

        if len(losses) > 0:
            return torch.stack(losses).sum() * cls.SEMANTIC_SCALE
        
        return None

def patch_model_inference(model):
    def new_inference(self, builder):
        for prop in self.poi:
            for sensor in prop.find(TorchSensor):
                sensor(builder)
        if builder.needsBatchRootDN():
            builder.addBatchRootDN()
        datanode = builder.getDataNode(device=self.device)
        if self.mode_ == Mode.TRAIN:
            return builder
        for infertype in self.inferTypes:
            try:
                if infertype in ['local/argmax', 'local/softmax']:
                    datanode.inferLocal()
            except Exception: pass
        return builder
    model.inference = types.MethodType(new_inference, model)
    return model

class TensorBoardPrimalDualProgram(PrimalDualProgram):
    def __init__(self, graph, Model, case_name="default", **kwargs):
        super().__init__(graph, Model, **kwargs)
        self.case_name = case_name
        self.writer = None
        self.early_stopper = None
        self.history = {'train_loss':[], 'val_loss': [], 'val_accuracy':[]}
        self.target_vars = TARGET_VARIABLES
        self.all_vars = ALL_VARIABLES
        
    def train(self, training_set, valid_set=None, test_set=None, early_stopper=None, **kwargs):
        log_dir = os.path.join("logs", f"tb_{self.case_name}_{datetime.now().strftime('%m%d_%H%M')}")
        self.writer = SummaryWriter(log_dir=log_dir)
        self.early_stopper = early_stopper
        print(f"TensorBoard logs saved to: {log_dir}")
        res = super().train(training_set, valid_set=valid_set, test_set=test_set, **kwargs)
        self.writer.close()
        return self.history

    def _extract_probs_and_task_loss(self, datanode):
        total_manual_loss =[]
        probs = {}
        
        for var_name in self.all_vars:
            if var_name in list(self.graph):
                concept_nodes = datanode.findDatanodes(select=self.graph[var_name])
                if not concept_nodes: continue
                
                var_logits_list = []
                var_labels_list =[]
                
                for node in concept_nodes:
                    logits = node.getAttribute('logits')
                    label = node.getAttribute('label/label')
                    if label is None:
                        label = node.getAttribute('label')
                        
                    if label is not None:
                        if logits is not None:
                            var_logits_list.append(logits)
                        lbl_val = label.item() if hasattr(label, 'item') else label
                        var_labels_list.append(lbl_val)
                
                if not var_labels_list: continue
                
                device = var_logits_list[0].device if var_logits_list else self.device
                labels_tensor = torch.tensor(var_labels_list, dtype=torch.long, device=device)
                num_classes = NUM_CLASSES_DICT.get(var_name, 10)
                
                if var_name in self.target_vars and var_logits_list:
                    logits_tensor = torch.stack(var_logits_list) # [Batch, C]
                    l = F.cross_entropy(logits_tensor, labels_tensor)
                    total_manual_loss.append(l)
                    probs[var_name] = F.softmax(logits_tensor, dim=1) 
                
                elif var_name not in self.target_vars:
                    probs[var_name] = F.one_hot(labels_tensor, num_classes=num_classes).float()
                    
        mloss = torch.stack(total_manual_loss).mean() if total_manual_loss else None
        return mloss, probs

    def train_epoch(self, dataset, c_warmup_iters=10, c_session={}, **kwargs):
        self.model.mode(Mode.TRAIN)
        iter_cnt = c_session.get('iter', 0)
        self.model.train()
        self.model.reset()
        
        epoch_task_loss = 0.0
        batch_count = 0

        for data in dataset:
            if self.opt is not None: self.opt.zero_grad()
            
            results = self.model(data)
            datanode = results[2] if len(results) > 2 else None
            if datanode is None: continue

            mloss, probs_dict = self._extract_probs_and_task_loss(datanode)
            if mloss is None: mloss = torch.tensor(0.0, device=self.device, requires_grad=True)
            
            loss = mloss
            task_loss_val = mloss.item()
            closs_val = 0.0

            current_beta = getattr(self, 'beta', 1.0)
            
            if iter_cnt == c_warmup_iters:
                print(f"\n[Epoch {self.epoch}] Warmup over! Semantic Loss officially kicks in! Current Beta = {current_beta}")

            if iter_cnt >= c_warmup_iters and probs_dict and current_beta > 0:
                closs_tensor = NativeSemanticLoss.compute(probs_dict)
                if closs_tensor is not None and closs_tensor.requires_grad:
                    closs_val = closs_tensor.item()
                    loss = mloss + current_beta * closs_tensor 
                else:
                    if iter_cnt % 50 == 0:
                        print(" Warning: NativeSemanticLoss returns None, no rules were triggered in the current batch.")
            
            if self.opt is not None and getattr(loss, 'requires_grad', False):
                loss.backward()
                self.opt.step()
                iter_cnt += 1

            if self.writer is not None and self.epoch is not None:
                d_len = get_len(dataset)
                global_step = (self.epoch - 1) * d_len + batch_count if d_len else getattr(self, '_g_step', 0) + 1
                self._g_step = global_step
                    
                self.writer.add_scalar('Batch_Loss/Train_Task', task_loss_val, global_step)
                if closs_val > 0:
                    self.writer.add_scalar('Batch_Loss/Train_Semantic', closs_val, global_step)

            epoch_task_loss += task_loss_val
            batch_count += 1
            yield (loss, results[1], datanode, results[3] if len(results) > 3 else None)

        if batch_count > 0:
            avg_loss = epoch_task_loss / batch_count
            self.history['train_loss'].append(avg_loss)
            if self.writer is not None and self.epoch is not None:
                self.writer.add_scalar('Epoch_Loss/Train_Total', avg_loss, self.epoch)

        c_session['iter'] = iter_cnt

    def _manual_accuracy(self, datanode):
        correct, total = 0, 0
        var_details = {}
        for var_name in self.target_vars:
            if var_name in list(self.graph):
                concept_nodes = datanode.findDatanodes(select=self.graph[var_name])
                v_correct, v_total = 0, 0
                for node in concept_nodes:
                    logits = node.getAttribute('logits')
                    label = node.getAttribute('label/label')
                    if logits is not None and label is not None:
                        pred = logits.argmax(dim=-1)
                        if pred.item() == label.item():
                            v_correct += 1
                            correct += 1
                        v_total += 1
                        total += 1
                if v_total > 0:
                    var_details[var_name] = v_correct / v_total
        return correct, total, var_details

    def call_epoch(self, name, dataset, epoch_fn, **kwargs):
        from tqdm import tqdm
        desc = name if self.epoch is None else f'Epoch {self.epoch} {name}'
        
        if name == 'Training':
            for _ in tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset) or 0, desc=desc): pass
        else:
            total_val_loss = 0.0
            total_correct, total_samples, val_batches = 0, 0, 0
            
            for result in tqdm(epoch_fn(dataset, **kwargs), total=get_len(dataset) or 0, desc=desc):
                datanode = result[2] if len(result) > 2 else None
                if datanode is not None:
                    l, _ = self._extract_probs_and_task_loss(datanode)
                    if l is not None:
                        total_val_loss += l.item()
                    c, t, _ = self._manual_accuracy(datanode)
                    total_correct += c
                    total_samples += t
                    val_batches += 1
                        
            avg_val_loss = total_val_loss / val_batches if val_batches > 0 else float('inf')
            val_acc = total_correct / total_samples if total_samples > 0 else 0.0
            
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_accuracy'].append(val_acc)
            
            if self.writer is not None and self.epoch is not None:
                self.writer.add_scalar('Epoch_Loss/Validation', avg_val_loss, self.epoch)
                self.writer.add_scalar('Accuracy/Validation', val_acc, self.epoch)
            
            print(f"[{name}] Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f} ({total_correct}/{total_samples})")
            if self.early_stopper and self.early_stopper(val_acc):
                print(f"Triggered early stop! Validation accuracy has not improved after consecutive rounds of {self.early_stopper.patience}.")
                self.stop = True

class DomiKnowsPatcher:
    @staticmethod
    def create_fixed_program(graph, target_variables, case_name="default", **kwargs):
        poi_list = [graph['block']]
        for var_name in ALL_VARIABLES:
            if var_name in list(graph): poi_list.append(graph[var_name])
                
        default_kwargs = {
            'poi': tuple(poi_list),
            'inferTypes': kwargs.get('inferTypes',['local/softmax']),
            'loss': NBCrossEntropyLoss(),
            'metric': PRF1Tracker(DatanodeCMMetric(inferType='local/softmax')),
        }
        default_kwargs.update(kwargs)
        program_type = default_kwargs.pop('program_type', 'base')
        
        if program_type == 'primal_dual':
            from domiknows.program.model.pytorch import PoiModel
            program = TensorBoardPrimalDualProgram(graph, Model=PoiModel, case_name=case_name, **default_kwargs)
            print("Complete PrimalDualProgram")
        else:
            program = POIProgram(graph, **default_kwargs)
            print("Complete POIProgram")
            
        program.model = patch_model_inference(program.model)
        return program

    @staticmethod
    def final_test_and_inference(program, test_reader, target_variables):
        program.model.eval()
        total_loss, total_correct, total_samples, batch_count = 0.0, 0, 0, 0
        var_correct = {var: 0 for var in target_variables}
        var_total = {var: 0 for var in target_variables}
        inference_results = {}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_reader):
                results = program.model(batch)
                datanode = results[2] if len(results) > 2 else None
                if datanode is None: continue
                
                if isinstance(program, TensorBoardPrimalDualProgram):
                    l, _ = program._extract_probs_and_task_loss(datanode)
                    if l is not None:
                        total_loss += l.item()
                        batch_count += 1
                
                batch_results = {}
                for var_name in target_variables:
                    if var_name in list(program.graph):
                        concept_nodes = datanode.findDatanodes(select=program.graph[var_name])
                        preds, labels, probs = [], [],[]
                        for node in concept_nodes:
                            logits = node.getAttribute('logits')
                            label = node.getAttribute('label/label')
                            if logits is not None:
                                pred = logits.argmax(dim=-1).item()
                                prob = F.softmax(logits, dim=-1).cpu().numpy().tolist()
                                preds.append(pred)
                                probs.append(prob)
                            if label is not None:
                                lbl = label.item()
                                labels.append(lbl)
                                if logits is not None and pred == lbl:
                                    var_correct[var_name] += 1
                                    total_correct += 1
                                var_total[var_name] += 1
                                total_samples += 1
                        
                        if preds and labels:
                            batch_results[var_name] = {
                                'predictions': preds, 
                                'labels': labels,
                                'probabilities': probs
                            }
                
                inference_results[f'batch_{batch_idx}'] = batch_results

        test_loss = total_loss / batch_count if batch_count > 0 else float('inf')
        test_acc = total_correct / total_samples if total_samples > 0 else 0.0
        test_acc_details = {k: (var_correct[k]/var_total[k] if var_total[k]>0 else 0.0) for k in target_variables}
        
        return test_loss, test_acc, test_acc_details, inference_results