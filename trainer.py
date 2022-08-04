import torch
import torch.nn as nn
import torch.nn.functional as F

import tqdm
import os

from utils import whiten_split

SS_SCHEDULE_15=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}]
SS_SCHEDULE_30=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}]
SS_SCHEDULE_50=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}, {'set_size':(25,50), 'steps':10000}]
SS_SCHEDULE_75=[{'set_size':(1,5), 'steps':20000}, {'set_size':(3,10), 'steps':5000}, {'set_size':(8,15), 'steps':5000}, {'set_size':(10,30), 'steps':5000}, {'set_size':(25,50), 'steps':5000}, {'set_size':(50,75), 'steps':5000}]
SS_SCHEDULES={15:SS_SCHEDULE_15, 30:SS_SCHEDULE_30, 75:SS_SCHEDULE_75}

class SetSizeScheduler():
    def __init__(self, schedule, step_mult=1):
        self.schedule=schedule
        self.N = sum([entry['steps'] for entry in schedule])
        self.step_mult=step_mult

    def get_set_size(self, iter_id):
        if iter_id >= 0:    #return last set size for iter_id -1
            step=0
            for entry in self.schedule:
                step += entry['steps']
                if self.step_mult * iter_id < step:
                    return entry['set_size']
        return self.schedule[-1]['set_size']    #fallback for now

class Trainer():
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=None,
            eval_every=500, save_every=2000, criterion=nn.BCEWithLogitsLoss(), scheduler=None, checkpoint_dir=None, ss_schedule=-1):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_args = train_args
        self.eval_args = eval_args
        self.device = device
        self.logger = logger
        self.eval_every = eval_every
        self.save_every = save_every
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.ss_schedule = SetSizeScheduler(SS_SCHEDULES[ss_schedule]) if ss_schedule > 0 else None

    def save_checkpoint(self, step, metrics):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(), 
            'scheduler': self.scheduler, 
            'step': step, 
            'metrics': metrics
        }
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        torch.save(save_dict, checkpoint_path)

    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        load_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(load_dict['scheduler'])
        step, metrics = load_dict['step'], load_dict['metrics']
        return step, metrics
    
    def train(self, train_steps, val_steps, test_steps):
        all_metrics={
            'train/loss': [],
        }

        def _log(name, value, step=-1):
            if name not in all_metrics:
                all_metrics[name] = []
            all_metrics[name].append(value)
            if self.logger is not None and step >= 0:
                self.logger.add_scalar(name, value, step)

        initial_step=0

        if self.checkpoint_dir is not None:
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
            else:
                checkpoint_path = os.path.join(self.checkpoint_dir, "checkpoint.pt")
                if os.path.exists(checkpoint_path):
                    initial_step, metrics = self.load_checkpoint()

        avg_loss = 0
        loss_fct = nn.BCEWithLogitsLoss()
        for i in tqdm.tqdm(range(initial_step, train_steps)):
            if self.ss_schedule is not None:
                set_size = self.ss_schedule.get_set_size(i)
                self.train_args['data_kwargs']['set_size'] = set_size
                self.eval_args['data_kwargs']['set_size'] = set_size

            loss = self.train_step(i, train_steps, self.train_dataset)
            
            _log('train/loss', loss, i)

            if i > initial_step:
                if self.eval_every > 0 and val_steps > 0 and self.val_dataset is not None and i % self.eval_every == 0:
                    val_metrics = self.evaluate(val_steps, self.val_dataset)
                    for k, v in val_metrics.items():
                        key = "val/"+k
                        _log(key, v, i)

                if self.checkpoint_dir is not None and i % self.save_every == 0:
                    self.save_checkpoint(i, all_metrics)

        if self.test_dataset is not None:
            test_metrics = evaluate(test_steps, self.test_dataset)
            for k, v in test_metrics.items():
                key = "test/"+k
                _log(key, v, -1)


        return all_metrics
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), target = dataset(args['batch_size'], **args['data_kwargs'])

        out = self.model(X.to(self.device),Y.to(self.device))
        loss = self.criterion(out.squeeze(-1), target.to(self.device))
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, steps, dataset):
        args = self.eval_args
        n_correct = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), target = self.train_dataset(args['batch_size'], **args['data_kwargs'])
                out = self.model(X.to(self.device),Y.to(self.device)).squeeze(-1)
                n_correct += torch.eq((out > 0), target.to(self.device)).sum().item()
        
        return {'acc': n_correct / (args['batch_size'] * steps)}


class CaptionTrainer(Trainer):
    def train_step(self, i, steps, dataset):
        def to_device(batch, device):
            if isinstance(batch, dict):
                batch['inputs'] =  {k:v.to(self.device) for k,v in batch['inputs'].items()}
            else:
                batch = batch.to(self.device)
            return batch

        args = self.train_args
        (X,Y), target = dataset(args['batch_size'], **args['data_kwargs'])

        out = self.model(to_device(X, self.device), to_device(Y, self.device))
        loss = self.criterion(out.squeeze(-1), target.to(self.device))
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()


class CountingTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=None,
            eval_every=500, save_every=2000, poisson=False, scheduler=None, checkpoint_dir=None, ss_schedule=-1):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=logger,
            eval_every=eval_every, save_every=save_every, criterion=poisson_loss if poisson else nn.MSELoss(), scheduler=scheduler, 
            checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.poisson=poisson
    
    def evaluate(self, steps, dataset):
        n_correct = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), target = dataset(batch_size, **data_kwargs)
                out = self.model(X.to(self.device),Y.to(self.device)).squeeze(-1)
                if self.poisson:
                    out = torch.exp(out)
                target = target.to(self.device).int()
                n_correct += torch.logical_or(torch.eq(out.ceil(), target), torch.eq(out.ceil()-1, target)).sum().item()
        return {'acc':n_correct / (batch_size * steps)}


class MetaDatasetTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=None,
            save_every=2000, episode_classes=100, episode_datasets=5, episode_length=250, scheduler=None, checkpoint_dir=None, 
            ss_schedule=-1):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            logger=logger, save_every=save_every, criterion=nn.BCEWithLogitsLoss(), scheduler=scheduler, 
            checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.episode_classes = episode_classes
        self.episode_datasets = episode_datasets
        self.episode_length = episode_length
    
    def train(self, train_steps, val_steps, test_steps):
        metrics={
            'train/loss': [],
        }

        def _log(name, value, step=-1):
            if name not in all_metrics:
                all_metrics[name] = []
            all_metrics[name].append(value)
            if self.logger is not None and step >= 0:
                self.logger.add_scalar(name, value, step)

        initial_step=0

        if self.checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            else:
                initial_step, metrics = self.load_checkpoint()

        avg_loss = 0
        n_episodes = math.ceil((train_steps - initial_step) / self.episode_length)
        step = initial_step
        for _ in tqdm.tqdm(range(n_episodes)):
            train_episode = self.train_dataset.get_episode(self.episode_classes, self.episode_datasets)
            for i in range(episode_length):
                if self.ss_schedule is not None:
                    set_size = self.ss_schedule.get_set_size(step)
                    self.train_args['data_kwargs']['set_size'] = set_size
                    self.eval_args['data_kwargs']['set_size'] = set_size

                loss = self.train_step(step, train_steps, train_episode)
                
                _log('train/loss', loss, step)

                step += 1

                if step > initial_step:
                    if step % save_every == 0:
                        self.save_checkpoint(step, metrics)
                    
                    if step >= train_steps:
                        break
            else:
                val_episode = self.val_dataset.get_episode(self.episode_classes, self.episode_datasets)
                val_metrics = self.evaluate(val_steps, val_episode)
                for k, v in val_metrics.items():
                    key = "val/"+k
                    _log(key, v, step)
                continue
            break
             
        if self.test_dataset is not None:
            episode = self.test_dataset.get_episode(self.episode_classes, self.episode_datasets)
            test_metrics = self.evaluate(val_steps, val_episode)
            for k, v in test_metrics.items():
                key = "test/"+k
                _log(key, v, -1)

        return metrics

    def evaluate(self, steps, dataset):
        args = self.eval_args
        n_correct = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), target = self.train_dataset(args['batch_size'], **args['data_kwargs'])
                out = self.model(X.to(self.device),Y.to(self.device)).squeeze(-1)
                n_correct += torch.eq((out > 0), target.to(self.device)).sum().item()
        
        return n_correct / (args['batch_size'] * steps)


class StatisticalDistanceTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, 
            label_fct, exact_loss, baselines, logger=None, save_every=2000, eval_every=500, scheduler=None, 
            checkpoint_dir=None, ss_schedule=-1):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            save_every=save_every, criterion=criterion, scheduler=scheduler, logger=logger,
            checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.exact_loss = exact_loss
        self.baselines = baselines
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        if self.exact_loss:
            X, theta = dataset(args['batch_size'], **args['sample_kwargs'])
            labels = self.label_fct(*theta, X=X[0], **args['label_kwargs']).squeeze(-1)
        else:
            X = dataset(args['batch_size'], **args['sample_kwargs'])
            if args['normalize'] == 'scale-linear':
                X, avg_norm = normalize_sets(*X)
            labels = self.label_fct(*X, **args['label_kwargs'])
        if args['normalize'] == 'scale-inv':
            X, avg_norm = normalize_sets(*X)
        elif args['normalize'] == 'whiten':
            X = whiten_split(*X)
        
        out = self.model(*X).squeeze(-1)

        loss = self.criterion(out, labels)
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, steps, dataset):
        args = self.eval_args
        model_losses=[]
        baseline_losses={baseline:[] for baseline in self.baselines.keys()}
        with torch.no_grad():
            for i in tqdm.tqdm(range(steps)):
                if self.exact_loss:
                    X, theta = dataset(args['batch_size'], **args['sample_kwargs'])
                    labels = self.label_fct(*theta, X=X[0], **args['label_kwargs']).squeeze(-1)
                else:
                    X = dataset(args['batch_size'], **args['sample_kwargs'])
                    labels = self.label_fct(*X, **args['label_kwargs'])
                
                for baseline_name, baseline_fct in self.baselines.items():
                    baseline_out = baseline_fct(*X).squeeze(-1)
                    baseline_loss = self.criterion(out, labels)
                    baseline_losses[baseline_name].append(baseline_loss.item())

                if not self.exact_loss and args['normalize'] == 'scale-linear':
                    X, avg_norm = normalize_sets(*X)
                    labels = self.label_fct(*X, **args['label_kwargs'])

                if args['normalize'] == 'scale-inv':
                    X, avg_norm = normalize_sets(*X)
                elif args['normalize'] == 'whiten':
                    X = whiten_split(*X)
                
                out = self.model(*X).squeeze(-1)
                loss = self.criterion(out, labels)
                model_losses.append(loss.item())

        metrics = {'loss': sum(model_losses)/len(model_losses)}
        for baseline_name, baseline_losses in baseline_losses.items():
            key = baseline_name + '/loss'
            metrics[key] = sum(baseline_losses)/len(baseline_losses)
        return metrics


import math

class DonskerVaradhanTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, criterion, label_fct, 
            logger=None, save_every=2000, eval_every=500, scheduler=None, checkpoint_dir=None, ss_schedule=-1, split_inputs=True):
        super().__init__(model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, logger=logger,
            save_every=save_every, criterion=criterion, scheduler=scheduler, checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.label_fct = label_fct
        self.split_inputs = split_inputs

    @staticmethod
    def _KL_estimate(X, Y):
        return X.sum(dim=1)/X.size(1) - Y.logsumexp(dim=1) + math.log(Y.size(1))
    
    def train_step(self, i, steps, dataset):
        args = self.train_args
        (X,Y), _ = dataset(args['batch_size'], **args['sample_kwargs'])
        if args['normalize'] == 'whiten':
            X,Y = whiten_split(X,Y)

        X, Y = X.to(self.device),Y.to(self.device)
        if self.split_inputs:
            X0,X1 = X.chunk(2, dim=1)
            Y0,Y1 = Y.chunk(2, dim=1)
        else:
            X0,X1 = X,X
            Y0,Y1 = Y,Y

        X_out, Y_out = self.model(X0, Y0, X1, Y1)
        loss = -1* self._KL_estimate(X_out, Y_out).mean()
        loss.backward()

        if (i+1) % args['grad_steps'] == 0 or i == (steps - 1):
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
        
        return loss.item()

    def evaluate(self, steps, dataset):
        args = self.eval_args
        avg_loss = 0
        with torch.no_grad():
            for i in range(steps):
                (X,Y), theta = self.train_dataset(args['batch_size'], **args['sample_kwargs'])
                KL_true = self.label_fct(*theta, X=X, **args['label_kwargs']).squeeze(-1)
                if args['normalize'] == 'whiten':
                    X,Y = whiten_split(X,Y)
                
                X, Y = X.to(self.device),Y.to(self.device)
                if self.split_inputs:
                    X0,X1 = X.chunk(2, dim=1)
                    Y0,Y1 = Y.chunk(2, dim=1)
                else:
                    X0,X1 = X,X
                    Y0,Y1 = Y,Y

                X_out, Y_out = self.model(X0, Y0, X1, Y1)
                KL_out = self._KL_estimate(X_out, Y_out)

                avg_loss += self.criterion(KL_out, KL_true)
            avg_loss /= steps
        
        return {"criterion":avg_loss}


#
#   Pretraining for image encoders
#

class Pretrainer():
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, device, batch_size, eval_every=-1):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.batch_size = batch_size
        self.eval_every = eval_every
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, steps):
        metrics={
            'train/loss': [],
            'val/acc': []
        }
        step = 0
        avg_loss = 0
        while step < steps:
            loader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.batch_size)
            for batch, targets in loader:
                optimizer.zero_grad()

                out = self.model(batch.to(device))
                loss = self.criterion(out, targets.to(device))
                loss.backward()
                self.optimizer.step()

                avg_loss += loss.item()
                metrics['train/loss'].append(loss.item())

                step += 1

                if step % self.eval_every == 0 and step > 0 and self.eval_every > 0:
                    avg_loss /= self.eval_every
                    acc = self.evaluate(self.val_dataset)
                    metrics['val/acc'].append(acc)
                    print("Step: %d\tTraining Loss: %f\t Eval Acc: %f" % (step, avg_loss, acc))
                    avg_loss = 0

                if step >= steps:
                    break

        if self.test_dataset is not None:
            acc = self.evaluate(self.test_dataset)
            metrics['test/acc'] = acc
            
        return metrics
    
    def evaluate(self, dataset):
        eval_loader = DataLoader(dataset, shuffle=True, batch_size=self.batch_size)
        acc = 0
        for batch, targets in eval_loader:
            with torch.no_grad():
                out = self.model(batch.cuda())
                acc += out.argmax(dim=-1).eq(targets.cuda()).sum().item()
        acc /= len(dataset)
        return acc