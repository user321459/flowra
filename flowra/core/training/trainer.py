"""
FLOWRA Trainer

Main training loop with support for dynamic refinement.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Any
from tqdm import tqdm

from flowra.core.training.callbacks import TrainingCallback, EarlyStoppingCallback


class FlowraTrainer:
    """
    Trainer for FLOWRA adapted models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adapters: Dict[str, nn.Module],
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        loss_fn: Optional[nn.Module] = None,
        callbacks: Optional[List[TrainingCallback]] = None,
        device: torch.device = None
    ):
        self.model = model
        self.adapters = adapters
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model.to(self.device)
        
        # Setup optimizer
        if optimizer is None:
            adapter_params = []
            for adapter in adapters.values():
                adapter_params.extend(p for p in adapter.parameters() if p.requires_grad)
            self.optimizer = torch.optim.AdamW(adapter_params, lr=1e-4)
        else:
            self.optimizer = optimizer
        
        self.scheduler = scheduler
        self.loss_fn = loss_fn or nn.CrossEntropyLoss()
        self.callbacks = callbacks or []
        
        self.global_step = 0
        self.current_epoch = 0
        self.training_history: List[Dict] = []
    
    def train(self, epochs: int = 3, max_steps: Optional[int] = None):
        """Run training loop."""
        self.model.train()
        
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_loss = 0.0
            
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            progress = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch in progress:
                if max_steps and self.global_step >= max_steps:
                    break
                
                for callback in self.callbacks:
                    callback.on_step_begin(self, self.global_step)
                
                loss = self._training_step(batch)
                epoch_loss += loss
                
                for callback in self.callbacks:
                    callback.on_step_end(self, self.global_step, loss)
                
                # Check early stopping
                for callback in self.callbacks:
                    if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                        print("Early stopping triggered")
                        return
                
                progress.set_postfix({'loss': loss})
                self.global_step += 1
            
            # Epoch end
            metrics = {'train_loss': epoch_loss / len(self.train_loader)}
            
            if self.val_loader:
                val_metrics = self.evaluate()
                metrics.update(val_metrics)
            
            self.training_history.append(metrics)
            
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)
            
            print(f"Epoch {epoch+1}: {metrics}")
        
        for callback in self.callbacks:
            callback.on_train_end(self)
    
    def _training_step(self, batch) -> float:
        """Single training step."""
        if isinstance(batch, (tuple, list)):
            inputs = batch[0].to(self.device)
            targets = batch[1].to(self.device) if len(batch) > 1 else None
        else:
            inputs = batch.to(self.device)
            targets = None
        
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        
        if targets is not None:
            if outputs.dim() == 3:
                loss = self.loss_fn(outputs[:, :-1].reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
            else:
                loss = self.loss_fn(outputs, targets)
        else:
            if outputs.dim() == 3:
                loss = self.loss_fn(outputs[:, :-1].reshape(-1, outputs.size(-1)), outputs[:, 1:].argmax(-1).reshape(-1))
            else:
                loss = outputs.mean()
        
        loss.backward()
        self.optimizer.step()
        
        if self.scheduler:
            self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self.val_loader:
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0].to(self.device)
                    targets = batch[1].to(self.device) if len(batch) > 1 else None
                else:
                    inputs = batch.to(self.device)
                    targets = None
                
                outputs = self.model(inputs)
                
                if targets is not None:
                    if outputs.dim() == 3:
                        loss = self.loss_fn(outputs[:, :-1].reshape(-1, outputs.size(-1)), targets[:, 1:].reshape(-1))
                    else:
                        loss = self.loss_fn(outputs, targets)
                else:
                    loss = outputs.mean()
                
                total_loss += loss.item()
        
        self.model.train()
        
        metrics = {'eval_loss': total_loss / len(self.val_loader)}
        
        for callback in self.callbacks:
            callback.on_evaluate(self, metrics)
        
        return metrics
