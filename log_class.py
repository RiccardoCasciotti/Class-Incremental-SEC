import json
import os 
import numpy as np

class Log():
    def __init__(self, experiment_path):
        self.experiment_path = experiment_path
        self.logs = {}
        self.current_task = None
        self.res_types = ["train", "val", "eval"]
    
    def check_res_type(self, res_type):
        if res_type not in self.res_types:
            raise Exception(f"Unknown result type, choose from: {self.res_types}")
        return
    
    def check_current(self):
        if self.current_task == None or self.current_task not in self.logs: 
            raise Exception("No current task given in the Log!")
        return

    def set_loss(self, loss, res_type, task=None):
        # gets the current_task, and modifies inside of it the *task* loss field
        self.check_current()
        self.check_res_type(res_type)
        if task == None: 
            task = self.current_task
        if task not in self.logs[self.current_task]:
            self.logs[self.current_task][task] = {}
        self.logs[self.current_task][task][f"{res_type}_loss"] = loss
        return

    def set_mAp(self, mAp, res_type, task=None):
         #  gets the current_task, and modifies inside of it the *task* mAp field
        self.check_current()
        self.check_res_type(res_type)
        if task == None: 
            task = self.current_task
        if task not in self.logs[self.current_task]:
            self.logs[self.current_task][task] = {}
        self.logs[self.current_task][task][f"{res_type}_mAp"] = mAp
        return
    
    def set_current_task(self, task): 
        #  the internal field so the logger knows at which tasks it is for the training.
        self.current_task = task
        self.logs[self.current_task] = {}
        self.check_current()
        return
    
    def get_full_logs(self): 
        #  return full dict.
        self.check_current()
        return self.logs
        
    def get_task_logs(self, task):
        self.check_current()
        return self.logs[task]

    def log_task_on_file(self, task):
        self.check_current()
        with open(f"{self.experiment_path}/T{task}_log.json", "w") as f: 
            json.dump(self.logs[task], f)
        return

    def log_full_on_file(self):
        self.check_current()
        with open(f"{self.experiment_path}/Full_experiment_logs.json", "w") as f: 
            json.dump(self.logs, f)
        return
    
    def log_full_on_vector(self):
        self.check_current()
        n_tasks = len(self.logs)
        log_vector = np.empty((n_tasks, n_tasks, 2, 3))
        for source_task in self.logs.keys():
            for evalued_task in self.logs[source_task].keys():
                mAps = [0.0, 0.0, 0.0]
                losses = [0.0, 0.0, 0.0]
                for metric in self.logs[source_task][evalued_task].keys():
                    if metric == "train_mAp":
                        mAps[0] = self.logs[source_task][evalued_task][metric]
                    elif metric == "val_mAp":
                        mAps[1] = self.logs[source_task][evalued_task][metric]
                    elif metric == "eval_mAp":
                        mAps[2] = self.logs[source_task][evalued_task][metric]
                    
                    elif metric == "train_loss":
                        losses[0] = self.logs[source_task][evalued_task][metric]
                    elif metric == "val_loss":
                        losses[1] = self.logs[source_task][evalued_task][metric]
                    elif metric == "eval_loss":
                        losses[2] = self.logs[source_task][evalued_task][metric]
                
                log_vector[source_task][evalued_task][0] = mAps
                log_vector[source_task][evalued_task][1] = losses
        np.save(f"{self.experiment_path}/metrics_vector.npy", log_vector)
        print(np.load(f"{self.experiment_path}/metrics_vector.npy"))
        return