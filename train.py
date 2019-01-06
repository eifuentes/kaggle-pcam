import random
from pathlib import Path
from random import randint

import numpy as np
import pandas as pd
import PIL
import torch
import torch.nn.functional as F
import torchvision.transforms as vtransforms
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.metrics import ROC_AUC
from ignite.engine import (Events, create_supervised_evaluator,
                           create_supervised_trainer)
from ignite.metrics import BinaryAccuracy, Loss, RunningAverage
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from dataset import PCamDataset, calculate_statistics
from model import WideResNetBinaryClassifier


def run(data_dir, validation_size=0.10, batch_size=64,
        max_epochs=10, lr=1e-3, beta1=0.9, beta2=0.999,
        num_workers=8, seed=None):
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

    # load training labels and calculate class info
    label_filename = Path(data_dir, 'train_labels.csv')
    label_df = pd.read_csv(label_filename)
    num_samples = label_df['label'].shape[0]
    label_cnts = label_df['label'].value_counts()
    print(f'Negative Class ({100*label_cnts[0]/num_samples:.3f}%): {label_cnts[0]}')
    print(f'Postive Class ({100*label_cnts[1]/num_samples:.3f}%): {label_cnts[1]}')
    print(f'Number of Samples: {num_samples}')

    # create datasets, verify integrity of dataset, and calculate statistics
    training_dset = PCamDataset(data_dir, 'train',
                                label_filename,
                                transform=vtransforms.ToTensor())
    num_val_samples = int(np.floor(len(training_dset) * validation_size))
    num_train_samples = len(training_dset) - num_val_samples
    train_dset, val_dset = random_split(training_dset,
                                        [num_train_samples, num_val_samples])
    test_dset = PCamDataset(data_dir, 'test', transform=vtransforms.ToTensor())
    train_mean, train_std = calculate_statistics(train_dset,
                                                 batch_size, num_workers)
    val_mean, val_std = calculate_statistics(val_dset,
                                             batch_size, num_workers)
    test_mean, test_std = calculate_statistics(test_dset,
                                               batch_size, num_workers)

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
    test_dset.transform = test_dtransform

    # assemble train, validation, test dataloaders
    train_dloader = DataLoader(train_dset, batch_size=batch_size,
                               shuffle=True, **dloader_args)
    val_dloader = DataLoader(val_dset, batch_size=batch_size,
                             shuffle=False, **dloader_args)
    # test_dloader = DataLoader(test_dset, batch_size=batch_size,
    #                           shuffle=False, **dloader_args)

    # setup model, optimizer, trainers and evaluators
    model = WideResNetBinaryClassifier(in_channels=3,
                                       num_groups=3, num_blocks_per_group=3,
                                       channel_scale_factor=6, init_num_channels=16,
                                       dropout_proba=0.0, residual_scale_factor=0.2)
    optimizer = Adam(model.parameters(), lr=lr, betas=(beta1, beta2))
    trainer = create_supervised_trainer(model, optimizer,
                                        F.binary_cross_entropy,
                                        device=device,
                                        non_blocking=non_blocking)
    eval_metrics = {
        'accuracy': BinaryAccuracy(),
        'bce': Loss(F.binary_cross_entropy),
        'roc': ROC_AUC(F.binary_cross_entropy)
    }
    evaluator = create_supervised_evaluator(model,
                                            metrics=eval_metrics,
                                            device=device,
                                            non_blocking=non_blocking)

    # attach handlers
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, ['loss'])

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_dloader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        avg_roc = metrics['roc']
        pbar.log_message(
            f'Training - Epoch: {engine.state.epoch} \
              ROC: {avg_roc:.2f} \
              Accuracy: {avg_accuracy:.2f} \
              Loss: {avg_bce:.2f}'
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_dloader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_bce = metrics['bce']
        avg_roc = metrics['roc']
        pbar.log_message(
            f'Validation - Epoch: {engine.state.epoch} \
              ROC: {avg_roc:.2f} \
              Accuracy: {avg_accuracy:.2f} \
              Loss: {avg_bce:.2f}'
        )
        pbar.n = pbar.last_print_n = 0

    # conduct training
    trainer.run(train_dloader, max_epochs=max_epochs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PCam Training Script')
    parser.add_argument('--datadir', type=str, default='/input/')
    parser.add_argument('--val-size', type=float, default=0.10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    run(args.datadir, args.val_size, args.batch_size,
        args.max_epochs, args.lr, args.beta1, args.beta2,
        args.num_workers, args.seed)
