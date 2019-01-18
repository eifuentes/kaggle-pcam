import json
import os
import random
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import BinaryAccuracy, Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dataset import PCamDataset, calculate_statistics
from model import WideResNetBinaryClassifier


def create_summary_writer(model, dloader, logdir):
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        print('failed to import tensorboardX')
        return None
    writer = SummaryWriter(log_dir=logdir)
    x, y = next(iter(dloader))
    try:
        writer.add_graph(model, x)
    except Exception as e:
        print(f'failed to save model graph: {e}')
    return writer


def fmt_metric(mode, name, value, step):
    metric = {
        'metric': str(mode) + '_' + str(name),
        'value': round(value, 5),
        'step': int(step)
    }
    return json.dumps(metric)


def log_metric(mode, name, value, step, writer=None):
    print(fmt_metric(mode, name, value, step))
    if writer:
        writer.add_scalar(f'{mode}/{name}', value, step)


def save_experiment_params(artifacts_dir, experiment_params):
    with open(os.path.join(artifacts_dir, 'experiment.json'), 'w') as fp:
        experiment_params = {p: experiment_params[p] for p in experiment_params
                             if isinstance(experiment_params[p], (str, int, float, list, tuple, bool))}
        json.dump(experiment_params, fp, sort_keys=True, indent=4)
    print('saved experiment params')


def run(datadir, outdir, validation_size=0.10, batch_size=32,
        max_epochs=10, lr=1e-3, beta1=0.5, beta2=0.9,
        num_workers=32, seed=None,
        log_iter_interval=20, logdir=None):

    # experiment params
    experiment_params = locals()
    print(experiment_params)

    # setup and device specific config
    seed = seed if seed else randint(1, 1000)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
        device = torch.device('cuda')
        non_blocking = True
        dloader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device('cpu')
        non_blocking = False
        dloader_args = {'num_workers': num_workers, 'pin_memory': False}
    experiment_params['seed'] = seed
    experiment_params['non_blocking'] = non_blocking
    print(f'using device: {device.type}')
    print(f'set random seed to {seed}')

    # load training labels and calculate class info
    label_filename = Path(datadir, 'train_labels.csv')
    label_df = pd.read_csv(label_filename)
    num_samples = label_df['label'].shape[0]
    label_cnts = label_df['label'].value_counts()
    print(f'Negative Class ({100*label_cnts[0]/num_samples:.3f}%): {label_cnts[0]}')
    print(f'Postive Class ({100*label_cnts[1]/num_samples:.3f}%): {label_cnts[1]}')
    print(f'Number of Samples: {num_samples}')

    # create datasets, verify integrity of dataset, and calculate statistics
    training_dset = PCamDataset(datadir, 'train',
                                label_filename,
                                transform=vtransforms.ToTensor())
    num_val_samples = int(np.floor(len(training_dset) * validation_size))
    num_train_samples = len(training_dset) - num_val_samples
    train_dset, val_dset = random_split(training_dset,
                                        [num_train_samples, num_val_samples])
    train_mean, train_std = calculate_statistics(train_dset,
                                                 batch_size, num_workers)
    val_mean, val_std = calculate_statistics(val_dset,
                                             batch_size, num_workers)
    experiment_params['normalization'] = {
        'mean': train_mean,
        'std': train_std
    }
    print(f'normalization mean: {train_mean} | std: {train_std}')

    # create data preprocessing pipeline for train/test
    train_dtransform = vtransforms.Compose([
        vtransforms.ColorJitter(brightness=0.05, contrast=0.05),
        vtransforms.RandomAffine(
            degrees=(180), translate=(0.05, 0.05),
            scale=(0.8, 1.2), shear=0.05, resample=PIL.Image.BILINEAR
        ),
        vtransforms.RandomHorizontalFlip(),
        vtransforms.RandomVerticalFlip(),
        vtransforms.ToTensor(),
        vtransforms.Normalize(train_mean, train_std)
    ])
    test_dtransform = vtransforms.Compose([
        vtransforms.ToTensor(),
        vtransforms.Normalize(train_mean, train_std)
    ])
    training_dset.transform = train_dtransform

    # assemble train, validation, test dataloaders
    train_dloader = DataLoader(train_dset, batch_size=batch_size,
                               shuffle=True, **dloader_args)
    val_dloader = DataLoader(val_dset, batch_size=batch_size,
                             shuffle=False, **dloader_args)

    # setup model, optimizer, tensorboard writer, trainers and evaluators
    model = WideResNetBinaryClassifier(in_channels=3,
                                       num_groups=4, num_blocks_per_group=3,
                                       channel_scale_factor=6, init_num_channels=16,
                                       dropout_proba=0.1, residual_scale_factor=0.2)
    writer = create_summary_writer(model, train_dloader, logdir)
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    trainer = create_supervised_trainer(model, optimizer,
                                        F.binary_cross_entropy,
                                        device=device,
                                        non_blocking=non_blocking)
    eval_metrics = {
        'accuracy': BinaryAccuracy(),
        'bce': Loss(F.binary_cross_entropy),
        'roc': ROC_AUC()
    }
    evaluator = create_supervised_evaluator(model,
                                            metrics=eval_metrics,
                                            device=device,
                                            non_blocking=non_blocking)

    # attach handlers for stdout and tensorboard
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        niter = (engine.state.iteration - 1) % len(train_dloader) + 1
        if niter % log_iter_interval == 0:
            log_metric('training', 'loss', engine.state.output, engine.state.iteration, writer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_dloader)
        metrics = evaluator.state.metrics
        avg_accuracy, avg_bce, avg_roc = metrics['accuracy'], metrics['bce'],  metrics['roc']
        log_metric('training', 'bce', avg_bce, engine.state.iteration, writer)
        log_metric('training', 'roc', avg_roc, engine.state.iteration, writer)
        log_metric('training', 'accuracy', avg_accuracy, engine.state.iteration, writer)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_dloader)
        metrics = evaluator.state.metrics
        avg_accuracy, avg_bce, avg_roc = metrics['accuracy'], metrics['bce'],  metrics['roc']
        log_metric('validation', 'bce', avg_bce, engine.state.iteration, writer)
        log_metric('validation', 'roc', avg_roc, engine.state.iteration, writer)
        log_metric('validation', 'accuracy', avg_accuracy, engine.state.iteration, writer)

    # save experiment run params
    save_experiment_params(outdir, experiment_params)

    # conduct training
    trainer.run(train_dloader, max_epochs=max_epochs)

    # save model and its assets
    torch.save(model.state_dict(), os.path.join(outdir, 'model.pt'))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PCam Training Script')
    parser.add_argument('--datadir', type=str, default='/pcam')
    parser.add_argument('--outdir', type=str, default='/output')
    parser.add_argument('--val-size', type=float, default=0.10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--num-workers', type=int, default=32)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--log-iter-interval', type=int, default=20)
    parser.add_argument('--logdir', type=str, default='logs')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.logdir, exist_ok=True)

    run(args.datadir, args.outdir, args.val_size, args.batch_size,
        args.max_epochs, args.lr, args.beta1, args.beta2,
        args.num_workers, args.seed,
        args.log_iter_interval, args.logdir)
