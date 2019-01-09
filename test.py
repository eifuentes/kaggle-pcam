""" Evaluation of PCam Test Dataset

    Example Floydhub Run Cmd:

    floyd run --data eifuentes/datasets/pcam/1:/pcam \
              --data eifuentes/projects/kaggle-pcam/10/output:/model \
              --gpu2 --env pytorch-1.0 \
              --message 'evaluation of  binary classification model on PCam test dataset' \
              'python test.py --datadir /pcam --modeldir /model'
"""
import json
import os

import pandas as pd
import torch
import torchvision.transforms as vtransforms
from torch.utils.data import DataLoader

from dataset import PCamDataset
from model import WideResNetBinaryClassifier


def eval_on_test(model, dloader, device, non_blocking=False):
    batch_size = dloader.batch_size
    ids = [f.stem for f in dloader.dataset.filepaths]
    model.to(device=device, non_blocking=non_blocking)
    model.eval()
    with torch.no_grad():
        test_pred_probas = list()
        for i, (x, _) in enumerate(dloader, start=0):
            pred_probas = model(x.to(device=device, non_blocking=non_blocking))
            pred_probas = pred_probas.cpu().detach().tolist()
            for j, pred_proba in enumerate(pred_probas):
                idx = (i * batch_size) + j
                test_pred_probas.append({
                    'id': str(ids[idx]),
                    'label': pred_proba
                })
    return pd.DataFrame.from_records(test_pred_probas).set_index('id', verify_integrity=True)


def run(datadir, modeldir, outdir, batch_size=128, num_workers=32):

    # eval run params
    eval_params = locals()
    print(eval_params)

    # setup and device specific config
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        cudnn.benchmark = True
        device = torch.device('cuda')
        non_blocking = True
        dloader_args = {'num_workers': 1, 'pin_memory': True}
    else:
        device = torch.device('cpu')
        non_blocking = False
        dloader_args = {'num_workers': num_workers, 'pin_memory': False}

    # load experiment metadata
    try:
        with open(os.path.join(modeldir, 'experiment.json'), 'r') as f:
            params = json.load(f)
        print('successfully loaded model experiment.json')
        print(params)
    except Exception:
        print('unable to find model experiment.json')
        norm_mean = [0.7023522853851318, 0.546073853969574, 0.6963717937469482]
        norm_std = [0.23241718113422394, 0.27298659086227417, 0.21017064154148102]

    # create data preprocessing pipeline
    test_dtransform = vtransforms.Compose([
        vtransforms.ToTensor(),
        vtransforms.Normalize(norm_mean, norm_std)
    ])

    # create datasets, verify integrity of dataset, and calculate statistics
    test_dset = PCamDataset(datadir, 'test', transform=test_dtransform)

    # assemble train, validation, test dataloaders
    test_dloader = DataLoader(test_dset, batch_size=batch_size,
                              shuffle=False, **dloader_args)

    # setup model, optimizer, tensorboard writer, trainers and evaluators
    model = WideResNetBinaryClassifier(in_channels=3,
                                       num_groups=4, num_blocks_per_group=3,
                                       channel_scale_factor=6, init_num_channels=16,
                                       dropout_proba=0.0, residual_scale_factor=0.2)
    model.load_state_dict(torch.load(os.path.join(modeldir, 'model.pt')))

    # evaluate test set
    test_pred = eval_on_test(model, test_dloader, device, non_blocking)
    print(f'successfully evaluated {len(test_pred)} samples')
    test_pred.to_csv(os.path.join(outdir, 'submission.csv'))
    print('saved results')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PCam Evaluation Script')
    parser.add_argument('--datadir', type=str, default='/pcam')
    parser.add_argument('--modeldir', type=str, default='/model')
    parser.add_argument('--outdir', type=str, default='/output')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    run(args.datadir, args.modeldir, args.outdir, args.batch_size, args.num_workers)
