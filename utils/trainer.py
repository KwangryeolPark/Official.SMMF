import os
import pickle
import torch
from datetime import datetime
from transformers import get_scheduler
from tqdm.auto import tqdm
from .seed import set_seed
from .optim import get_optim
from .dataset import get_dataset
from .model import get_model

class Trainer(object):
    def __init__(self, args):
        
        set_seed(args)
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_ds, self.eval_ds = get_dataset(args)
        self.model = get_model(args).to(self.device)
        self.optimizer = get_optim(args, self.model)
        self.scheduler = get_scheduler('cosine_with_restarts', self.optimizer, 100, args.num_epochs * len(self.train_ds))
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.results = {
            'train_loss': [],
            'train_acc': [],
            'eval_loss': [],
            'eval_acc': [],
            'args': args
        }
        self.args = args
        
    def train(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_acc = 0.0
        progress_bar = tqdm(self.train_ds, leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            self.optimizer.zero_grad()
            
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            loss.backward()
            
            self.optimizer.step()
            self.scheduler.step()
            
            progress_bar.set_description(
                f'Epoch {epoch} Loss: {loss.item():.4f} Acc: {acc.item():.4f} lr: {self.optimizer.param_groups[0]["lr"]:.4f}'
            )
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            self.results['train_loss'].append(loss.item())
            self.results['train_acc'].append(acc.item())
        
        avg_loss = total_loss / len(self.train_ds)
        avg_acc = total_acc / len(self.train_ds)
        
        return avg_loss, avg_acc
    
    @torch.no_grad()
    def evaluate(self, epoch):
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        progress_bar = tqdm(self.eval_ds, leave=True)
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            acc = (outputs.argmax(dim=1) == targets).float().mean()
            
            progress_bar.set_description(
                f'Epoch {epoch} Loss: {loss.item():.4f} Acc: {acc.item():.4f}'
            )
            
            total_loss += loss.item()
            total_acc += acc.item()
            
            self.results['eval_loss'].append(loss.item())
            self.results['eval_acc'].append(acc.item())
        
        avg_loss = total_loss / len(self.eval_ds)
        avg_acc = total_acc / len(self.eval_ds)
        
        return avg_loss, avg_acc
    
    def fit(self):
        for epoch in range(self.args.num_epochs):
            train_loss, train_acc = self.train(epoch)
            eval_loss, eval_acc = self.evaluate(epoch)
            print(
                f'Epoch {epoch} train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} eval_loss: {eval_loss:.4f} eval_acc: {eval_acc:.4f}'
            )
        return self.results
    
    def finish(self):
        self.save_results()
    
    def save_results(self):
        if os.path.exists('results') == False:
            os.mkdir('results')
        now = str(datetime.now()).replace(' ', '-').replace(':', '-').replace('.', '-')
        file_dir = f'results/{now}.pkl'
        
        with open(file_dir, 'wb') as f:
            pickle.dump(self.results, f)
