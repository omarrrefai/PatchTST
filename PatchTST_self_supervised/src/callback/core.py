
__all__ = ['Callback', 'SetupLearnerCB', 'GetPredictionsCB', 'GetTestCB' ]


""" 
Callback lists:
    > before_fit
        - before_epoch
            + before_epoch_train                
                ~ before_batch_train
                ~ after_batch_train                
            + after_epoch_train

            + before_epoch_valid                
                ~ before_batch_valid
                ~ after_batch_valid                
            + after_epoch_valid
        - after_epoch
    > after_fit

    - before_predict        
        ~ before_batch_predict
        ~ after_batch_predict          
    - after_predict

"""

from ..basics import *
import torch

DTYPE = torch.float32

class Callback(GetAttr): 
    _default='learner'


class SetupLearnerCB(Callback): 
    def __init__(self):        
        self.device = default_device(use_cuda=True)

    def before_batch_train(self): self._to_device()
    def before_batch_valid(self): self._to_device()
    def before_batch_predict(self): self._to_device()
    def before_batch_test(self): self._to_device()

    def _to_device(self):
        batch = to_device(self.batch, self.device)        
        if self.n_inp > 1: xb, yb = batch
        else: xb, yb = batch, None        
        self.learner.batch = xb, yb
        
    def before_fit(self): 
        "Set model to cuda before training"                
        self.learner.model.to(self.device)
        self.learner.device = self.device                        


class GetPredictionsCB(Callback):
    def __init__(self):
        super().__init__()

    def before_predict(self):
        self.preds = []        
    
    def after_batch_predict(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)

    def after_predict(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()

         

class GetTestCB(Callback):
    def __init__(self):
        super().__init__()

    def before_test(self):
        self.preds, self.targets = [], []        
    
    def after_batch_test(self):        
        # append the prediction after each forward batch           
        self.preds.append(self.pred)
        self.targets.append(self.yb)

    def after_test(self):           
        self.preds = torch.concat(self.preds)#.detach().cpu().numpy()
        self.targets = torch.concat(self.targets)#.detach().cpu().numpy()

        # inside GetTestCB.after_test()
        dl = getattr(self.learner, 'dl', None)
        use_real_scale = getattr(self.learner, 'eval_on_real_scale', False)
        scaler = getattr(getattr(dl, 'dataset', object()), 'scaler', None)

        # Align target to pred horizon for convenience
        if self.targets.shape[1] != self.preds.shape[1]:
            self.targets = self.targets[:, -self.preds.shape[1]:, :]

        if use_real_scale and scaler is not None:
            from ..utils import inverse_scale_tensor
            self.preds = inverse_scale_tensor(self.preds, scaler)
            self.targets = inverse_scale_tensor(self.targets, scaler)
