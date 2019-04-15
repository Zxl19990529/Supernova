import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.core import results2json, coco_eval
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors

from mmdet.core import eval_map
import numpy as np


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)

def voc_eval(result_file, dataset, iou_thr=0.5):
    det_results = mmcv.load(result_file)
    gt_bboxes = []
    gt_labels = []
    gt_ignore = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        bboxes = ann['bboxes']
        labels = ann['labels']
        if 'bboxes_ignore' in ann:
            ignore = np.concatenate([
                np.zeros(bboxes.shape[0], dtype=np.bool),
                np.ones(ann['bboxes_ignore'].shape[0], dtype=np.bool)
            ])
            gt_ignore.append(ignore)
            bboxes = np.vstack([bboxes, ann['bboxes_ignore']])
            labels = np.concatenate([labels, ann['labels_ignore']])
        gt_bboxes.append(bboxes)
        gt_labels.append(labels)
    if not gt_ignore:
        gt_ignore = gt_ignore
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    eval_map(
        det_results,
        gt_bboxes,
        gt_labels,
        gt_ignore=gt_ignore,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        print_summary=True)


def parse_args():
    parser = argparse.ArgumentParser(description='VOC Evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument('--proc_per_gpu',default=1,type=int,help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='bbox',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--iou-thr',type=float,default=0.5,help='IoU threshold for evaluation')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        cfg = mmcv.Config.fromfile(args.config)
        test_dataset = mmcv.runner.obj_from_dict(cfg.data.test, datasets)
        voc_eval(args.out, test_dataset, args.iou_thr)

if __name__ == '__main__':
    main()

