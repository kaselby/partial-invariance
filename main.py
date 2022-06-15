

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import apex

import argparse
import os
import shutil
from datetime import date

from tasks import TASKS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_name', type=str)
    parser.add_argument('--run_id', type=int, default=-1)
    parser.add_argument('--logfile', type=str, default="exp_logs.pt")
    parser.add_argument('--n', type=int, default=2)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--latent_size', type=int, default=16)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--disc_blocks', type=int, default=4)
    parser.add_argument('--gen_blocks', type=int, default=4)
    parser.add_argument('--steps', type=int, default=100000)
    parser.add_argument('--val_steps', type=int, default=250)
    parser.add_argument('--save_every', type=int, default=10000)
    parser.add_argument('--eval_every', type=int, default=2000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--reference_size', type=int, nargs=2, default=[500,750])
    parser.add_argument('--candidate_size', type=int, nargs=2, default=[200,300])
    parser.add_argument('--output_layers', type=int, default=1)
    parser.add_argument('--weight_sharing', type=str, choices=['none', 'sym', 'cross'], default='none')
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--basedir', type=str, default="runs")
    parser.add_argument('--l_smooth', type=float, default=0.1)
    parser.add_argument('--p_flip', type=float, default=0.05)
    parser.add_argument('--warmup_steps', type=int, default=150000)
    parser.add_argument('--anneal_steps', type=int, default=500000)
    parser.add_argument('--noise_dim', type=int, default=1)
    parser.add_argument('--normalize', action="store_true")
    parser.add_argument('--blur', default=0.05)
    parser.add_argument('--task', type=str, choices=('gmm', 'mnist', 'omniglot', 'base-mnist'), default='gmm')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--nu', type=float, default=3)
    parser.add_argument('--mu0', type=float, default=0)
    parser.add_argument('--s0', type=float, default=0.2)
    parser.add_argument('--mu_scale', type=float, default=8)
    parser.add_argument('--pretrained', default=None)
    parser.add_argument('--no_encoder', action='store_true')
    parser.add_argument('--conv_mode', type=str, choices=('none', 'seq', 'trans'), default='none')
    parser.add_argument('--model_type', type=str, default='base')
    parser.add_argument('--criterion', type=str, choices=('bce', 'wgan', 'hinge'), default='bce')
    parser.add_argument('--lambda_gp', type=float, default=10.0)
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--gen_sample_size', type=int, nargs=2, default=(4, 4))
    parser.add_argument('--spectral_norm', action="store_true")
    parser.add_argument('--specnorm_param', action="store_true")
    parser.add_argument('--gradient_penalty', action="store_true")
    parser.add_argument('--r1', action="store_true")
    parser.add_argument('--cycle_loss', action="store_true")
    parser.add_argument('--bn', action="store_true")
    parser.add_argument('--add_img_noise', action="store_true")
    parser.add_argument('--save_gen_samples', action="store_true")
    parser.add_argument('--lambda_cyc', type=float, default=1.0)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--imgsize', type=int, default=-1)
    parser.add_argument('--gaussian_noise', type=float, default=0)
    parser.add_argument('--use_amp', action="store_true")
    parser.add_argument('--use_apex', action="store_true")
    return parser.parse_args()

def write_log(logfile, run_name, run_id):
    with open(logfile, 'a') as writer:
        writer.write(str(date.today())+"\t"+run_name+"\t"+str(run_id)+"\n")

if __name__ == '__main__':
    args = parse_args()

    run_dir = os.path.join(args.basedir, args.run_name)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    log_dir = os.path.join(run_dir, "logs")
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    args_file = os.path.join(run_dir, "args.pt")
    if not os.path.exists(args_file):
        torch.save({'id':args.run_id, 'args':args}, args_file)
        write_log(args.logfile, args.run_name, args.run_id)

    checkpoint_file = os.path.join(args.checkpoint_dir, 'checkpoint.pt')
    resume = os.path.exists(checkpoint_file)

    device = torch.device("cuda")

    task = TASKS[args.task](args)
    train_dataset, val_dataset, test_dataset = task.build_dataset()

    if task.pretraining_task is not None and not resume:
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
    training_args, eval_args = task.build_trainer_args()
    metrics = task.build_metrics(device)

    if args.use_apex:
        opt = apex.optimizers.FusedAdam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    logger = SummaryWriter(log_dir)

    trainer = task.build_trainer(model, opt, train_dataset, val_dataset, test_dataset, device)
    all_metrics = trainer.train(args.steps, args.val_steps, args.save_every, args.eval_every)
    
    model_out = model._modules['module'] if torch.cuda.device_count() > 1 else model
    torch.save(model_out, os.path.join(run_dir, "model.pt"))
    torch.save({'metrics':all_metrics, 'args':args}, os.path.join(run_dir, "out.pt"))
    
    shutil.move(os.path.join(args.checkpoint_dir, "checkpoint.pt"), os.path.join(run_dir, "last_checkpoint.pt"))



