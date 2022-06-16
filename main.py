

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

#import apex

import argparse
import os
import shutil
from datetime import date

from tasks import TASKS
from builders import SET_MODEL_BUILDERS


def parse_args():
    parser = argparse.ArgumentParser()
    # File paths
    parser.add_argument('run_name', type=str)
    parser.add_argument('--run_id', type=int)
    parser.add_argument('--basedir', type=str, default="runs")
    parser.add_argument('--dataset_dir', type=str, default='./data')
    parser.add_argument('--checkpoint_dir', type=str, default="/checkpoint/kaselby")
    parser.add_argument('--checkpoint_name', type=str, default=None)

    # Run config
    parser.add_argument('--model', type=str, default='mst', choices=SET_MODEL_BUILDERS.keys())
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--task', type=str)

    # Training args
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--grad_steps', type=int, default=1)
    parser.add_argument('--set_size', type=int, nargs=2, default=[6,10])
    parser.add_argument('--ss_schedule', type=int, choices=[-1, 15, 30, 50, 75], default=-1)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every', type=int, default=2000)
    parser.add_argument('--train_steps', type=int, default=5000)
    parser.add_argument('--val_steps', type=int, default=200)
    parser.add_argument('--test_steps', type=int, default=500)
    parser.add_argument('--use_amp', action="store_true")
    parser.add_argument('--use_apex', action="store_true")
    
    # Model args
    parser.add_argument('--num_blocks', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--decoder_layers', type=int, default=1)
    parser.add_argument('--ln', action='store_true')
    parser.add_argument('--weight_sharing', type=str, choices=['none', 'cross', 'sym'], default='none')

    # Pretraining args
    parser.add_argument('--pretrain_steps', type=int, default=0)
    parser.add_argument('--pretrain_lr', type=float, default=3e-4)

    # Counting args
    parser.add_argument('--poisson', action='store_true')
    parser.add_argument('--val_split', type=float, default=0.1)

    # Alignment args
    parser.add_argument('--text_model', type=str, choices=['bert', 'ft'], default='bert')
    parser.add_argument('--img_model', type=str, choices=['resnet', 'cnn'], default='resnet')   #also for md
    parser.add_argument('--embed_path', type=str, default="cc.en.300.bin")
    parser.add_argument('--embed_dim', type=int, default=300)
    parser.add_argument('--overlap_mult', type=int, default=-1)

    # Distinguishability args
    parser.add_argument('--episode_classes', type=int, default=100)
    parser.add_argument('--episode_datasets', type=int, default=5)
    parser.add_argument('--episode_length', type=int, default=500)
    parser.add_argument('--p_dl', type=float, default=0.3)
    parser.add_argument('--n', type=int, default=8)     # also for stat
    parser.add_argument('--md_path', type=str, default="/ssd003/projects/meta-dataset")

    # Statistical distance args
    parser.add_argument('--normalize', type=str, choices=('none', 'scale-linear', 'scale-inv', 'whiten'))
    parser.add_argument('--scaling', type=float, default=0.5)
    parser.add_argument('--blur', type=float, default=0.05)
    parser.add_argument('--equi', action='store_true')
    parser.add_argument('--vardim', action='store_true')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, args.dataset, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    log_dir = os.path.join(run_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    args_file = os.path.join(run_dir, "args.pt")
    if not os.path.exists(args_file):
        torch.save({'id':args.run_id, 'args':args}, args_file)
        #write_log(args.logfile, args.run_name, args.run_id)

    checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    resume = os.path.exists(checkpoint_file)

    device = torch.device("cuda")

    task = TASKS[args.task](args)
    train_dataset, val_dataset, test_dataset = task.build_dataset()

    if task.pretraining_task is not None and not resume and args.pretrain_steps > 0:
        pretraining_task = task.pretraining_task(args)
        pretraining_model = pretrain_task.build_model().to(device)
        pretraining_opt = torch.optim.Adam(pretrain_model.parameters(), lr=args.pretrain_lr)
        pretrainer = pretrain_task.build_trainer(pretrain_model, pretrain_opt, train_dataset, val_dataset, None, device)

        pretrain_metrics = pretrainer.train(args.pretrain_steps)
        pretrained_model = pretraining_model._modules['0']
    else:
        pretrained_model = None

    model = task.build_model(pretrained_model=pretrained_model).to(device)

    if torch.cuda.device_count() > 1:
        n_gpus = torch.cuda.device_count()
        print("Using", n_gpus, "GPUs.")
        model = nn.DataParallel(model)
        args.batch_size *= n_gpus

    train_dataset, val_dataset, test_dataset = task.build_dataset()
    metrics = task.build_metrics(device)

    if args.use_apex:
        pass
        #opt = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    logger = SummaryWriter(log_dir)

    trainer = task.build_trainer(model, opt, None, train_dataset, val_dataset, test_dataset, device)
    all_metrics = trainer.train(args.steps, args.val_steps, args.save_every, args.eval_every)
    
    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir, "model.pt"))
    torch.save({'metrics':all_metrics, 'args':args}, os.path.join(run_dir, "out.pt"))
    
    shutil.move(os.path.join(args.checkpoint_dir, "checkpoint.pt"), os.path.join(run_dir, "last_checkpoint.pt"))



