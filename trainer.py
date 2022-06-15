import torch
import torch.nn as nn



class Trainer():
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device
            eval_every=500, save_every=2000, criterion=nn.BCEWithLogitsLoss(), scheduler=None, checkpoint_dir=None, ss_schedule=None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.train_args = train_args
        self.eval_args = eval_args
        self.device = device
        self.eval_every = eval_every
        self.save_every = save_every
        self.criterion = criterion
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.ss_schedule = ss_schedule

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

    def load_checkpoint(self)
        checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")
        load_dict = torch.load(checkpoint_path)
        self.model.load_state_dict(load_dict['model'])
        self.optimizer.load_state_dict(load_dict['optimizer'])
        self.scheduler.load_state_dict(load_dict['scheduler'])
        step, metrics = load_dict['step'], load_dict['metrics']
        return step, metrics
    
    def train(self, train_steps, val_steps, test_steps):
        metrics={
            'train/loss': [],
            'val/acc': []
        }
        initial_step=0

        if self.checkpoint_dir is not None:
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            else:
                initial_step, metrics = self.load_checkpoint()

        avg_loss = 0
        loss_fct = nn.BCEWithLogitsLoss()
        for i in tqdm.tqdm(range(initial_step, train_steps)):
            if self.ss_schedule is not None:
                set_size = self.ss_schedule.get_set_size(i)
                self.train_args['data_kwargs']['set_size'] = set_size
                self.eval_args['data_kwargs']['set_size'] = set_size

            loss = self.train_step(i, steps, self.train_dataset)
            
            avg_loss += loss
            metrics['train/loss'].append(loss)

            if i > initial_step:
                if i % self.eval_every == 0:
                    acc = self.evaluate(eval_steps, self.val_dataset)
                    metrics['val/acc'].append(acc)
                    avg_loss /= eval_every
                    print("Step: %d\tLoss: %f\tAccuracy: %f" % (i, avg_loss, acc))
                    avg_loss = 0

                if i % self.save_every == 0:
                    self.save_checkpoint()

        if self.test_dataset is not None:
            test_acc = evaluate(test_steps, self.test_dataset)
            metrics['test/acc'] = test_acc

        return metrics
    
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
        
        return n_correct / (args['batch_size'] * steps)


class CountingTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            eval_every=500, save_every=2000, poisson=False, scheduler=None, checkpoint_dir=None, ss_schedule=None):
        super(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            eval_every=eval_every save_every=save_every, criterion=poisson_loss if poisson else nn.MSELoss(), scheduler=scheduler, 
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
        return n_correct / (batch_size * steps)


class MetaDatasetTrainer(Trainer):
    def __init__(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device, save_every=2000, episode_classes=100, episode_datasets=5, episode_length=250, scheduler=None, checkpoint_dir=None, ss_schedule=None):
        super(self, model, optimizer, train_dataset, val_dataset, test_dataset, train_args, eval_args, device,
            save_every=save_every, criterion=nn.BCEWithLogitsLoss(), scheduler=scheduler, 
            checkpoint_dir=checkpoint_dir, ss_schedule=ss_schedule)
        self.episode_classes = episode_classes
        self.episode_datasets = episode_datasets
        self.episode_length = episode_length
    
    def train(self, train_steps, val_steps, test_steps):
        metrics={
            'train/loss': [],
            'val/acc': []
        }
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
            episode = self.train_dataset.get_episode(self.episode_classes, self.episode_datasets)
            for i in range(episode_length):
                if self.ss_schedule is not None:
                    set_size = self.ss_schedule.get_set_size(step)
                    self.train_args['data_kwargs']['set_size'] = set_size
                    self.eval_args['data_kwargs']['set_size'] = set_size

                loss = self.train_step(step, steps, episode)
                
                avg_loss += loss
                metrics['train/loss'].append(loss)

                step += 1

                if step > initial_step:
                    if step % save_every == 0:
                        self.save_checkpoint()
                    
                    if step >= steps:
                        break
            else:
                episode = self.val_dataset.get_episode(self.episode_classes, self.episode_datasets)
                acc = self.evaluate(val_steps, episode)
                metrics['val/acc'].append(acc)
                avg_loss /= eval_every
                print("Step: %d\tLoss: %f\tAccuracy: %f" % (step, avg_loss, acc))
                avg_loss = 0
                continue
            break
             
        if self.test_dataset is not None:
            episode = self.test_dataset.get_episode(self.episode_classes, self.episode_datasets)
            test_acc = self.evaluate(test_steps, self.test_dataset)
            metrics['test/acc'] = test_acc

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
            'train/loss': []
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