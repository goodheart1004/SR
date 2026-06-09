import argparse
import os
import time
from collections import defaultdict

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from arguments import train_parser
from data import ProcessedDSMDataset
from losses import get_loss
from model import GADBase
from utils import new_log, seed_all, to_device


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        self.dataloaders = self.get_dataloaders(args)

        seed_all(args.seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = GADBase(
            args.feature_extractor,
            Npre=args.Npre,
            Ntrain=args.Ntrain,
            guide_channels=ProcessedDSMDataset.guide_channels,
        ).to(self.device)

        self.experiment_folder, self.args.expN, self.args.randN = new_log(
            os.path.join(args.save_dir, 'DSM'),
            args
        )
        self.args.experiment_folder = self.experiment_folder
        self.writer = SummaryWriter(log_dir=self.experiment_folder)

        if not args.no_opt:
            self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.w_decay)
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        else:
            self.optimizer = None
            self.scheduler = None

        self.epoch = 0
        self.iter = 0
        self.train_stats = defaultdict(lambda: np.nan)
        self.val_stats = defaultdict(lambda: np.nan)
        self.best_rmse_loss = np.inf

        if args.resume is not None:
            self.resume(path=args.resume)

    def __del__(self):
        writer = getattr(self, 'writer', None)
        if writer is not None:
            writer.close()

    def train(self):
        with tqdm(range(self.epoch, self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_rmse=np.nan, validation_rmse=np.nan, best_rmse=np.nan)
            for _ in tnr:
                self.train_epoch(tnr)

                if (self.epoch + 1) % self.args.val_every_n_epochs == 0:
                    self.validate()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')

                if self.args.lr_scheduler == 'step' and not self.args.no_opt:
                    self.scheduler.step()
                    self.writer.add_scalar('log_lr', np.log10(self.scheduler.get_last_lr()[0]), self.epoch)

                self.epoch += 1

    def train_epoch(self, tnr=None):
        self.train_stats = defaultdict(float)

        self.model.train()
        log_interval = min(self.args.logstep_train, len(self.dataloaders['train']))

        with tqdm(self.dataloaders['train'], leave=False) as inner_tnr:
            inner_tnr.set_postfix(training_rmse=np.nan)
            for i, sample in enumerate(inner_tnr):
                sample = to_device(sample, self.device)

                if not self.args.no_opt:
                    self.optimizer.zero_grad()

                output = self.model(sample, train=True)
                loss, loss_dict = get_loss(output, sample, self.args.loss)

                if torch.isnan(loss):
                    raise RuntimeError('detected NaN loss')

                for key, value in loss_dict.items():
                    self.train_stats[key] += value.detach().cpu().item() if torch.is_tensor(value) else value

                if self.epoch > 0 or not self.args.skip_first:
                    if not self.args.no_opt:
                        loss.backward()

                        if self.args.gradient_clip > 0.:
                            clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)

                        self.optimizer.step()

                self.iter += 1

                if (i + 1) % log_interval == 0:
                    self.train_stats = {key: value / log_interval for key, value in self.train_stats.items()}

                    inner_tnr.set_postfix(training_rmse=self.train_stats['rmse_loss'])
                    if tnr is not None:
                        tnr.set_postfix(
                            training_rmse=self.train_stats['rmse_loss'],
                            validation_rmse=self.val_stats['rmse_loss'],
                            best_rmse=self.best_rmse_loss
                        )

                    for key, value in self.train_stats.items():
                        self.writer.add_scalar('train/' + key, value, self.iter)

                    self.train_stats = defaultdict(float)

    def validate(self):
        self.val_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for sample in tqdm(self.dataloaders['val'], leave=False):
                sample = to_device(sample, self.device)

                output = self.model(sample)
                _, loss_dict = get_loss(output, sample, self.args.loss)

                for key, value in loss_dict.items():
                    self.val_stats[key] += value.detach().cpu().item() if torch.is_tensor(value) else value

            self.val_stats = {key: value / len(self.dataloaders['val']) for key, value in self.val_stats.items()}

            for key, value in self.val_stats.items():
                self.writer.add_scalar('val/' + key, value, self.epoch)

            if self.val_stats['rmse_loss'] < self.best_rmse_loss:
                self.best_rmse_loss = self.val_stats['rmse_loss']
                if self.args.save_model in ['best', 'both']:
                    self.save_model('best')

    @staticmethod
    def get_dataloaders(args):
        data_args = {
            'crop_size': args.crop_size,
            'in_memory': args.in_memory,
            'max_rotation_angle': args.max_rotation,
            'do_horizontal_flip': not args.no_flip,
            'scaling': args.scaling
        }

        datasets = {
            'train': ProcessedDSMDataset(args.data_dir, **data_args, split='train', crop_deterministic=False),
            'val': ProcessedDSMDataset(args.data_dir, **data_args, split='val', crop_deterministic=True),
        }

        return {
            'train': DataLoader(
                datasets['train'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=True,
                drop_last=False
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False,
                drop_last=False
            )
        }

    def save_model(self, prefix=''):
        checkpoint = {
            'model': self.model.state_dict(),
            'epoch': self.epoch + 1,
            'iter': self.iter,
            'best_rmse_loss': self.best_rmse_loss,
        }
        if not self.args.no_opt:
            checkpoint['optimizer'] = self.optimizer.state_dict()
            checkpoint['scheduler'] = self.scheduler.state_dict()
        torch.save(checkpoint, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))

    def resume(self, path):
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if not self.args.no_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.epoch = checkpoint['epoch']
        self.iter = checkpoint['iter']
        self.best_rmse_loss = checkpoint.get('best_rmse_loss', np.inf)

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
