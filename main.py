import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import multiprocessing as mp
import time
import functools
from tqdm import tqdm

from mmdet.models import losses
from mmdet.core import bbox_overlaps

preds_num = 5000
iters = 200
radius = 3.0
center = (10.0, 10.0)
scales = (0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0)
aspect_ratios = (1 / 4, 1 / 3, 1 / 2, 1.0, 2.0, 3.0, 4.0)

modes = ('LIoU', 'navie_iou', 'IoU', 'GIoU', 'DIoU', 'CIoU')


def random_preds_center(center, radius, preds_num):
    """generate random points in a circle

    Args:
        center: center of circle
        radius: radius of circle
        preds_num: number of random points
    
    Returns:
        list: [(x, y), ...]
    """
    random_points = []
    for _ in range(preds_num):
        r = np.sqrt(np.random.rand()) * radius
        theta = np.random.rand() * 2 * np.pi
        x = center[0] + r * np.cos(theta)
        y = center[1] + r * np.sin(theta)
        random_points.append((x, y))
    return random_points


def get_box_xywh(center, scale, aspect_ratio):
    """generate box based on the center, scale (area), and aspect ratio
    
    Args:
        center: center of box
        scale: scale of box
        aspect_ratio: aspect ratio of box
    
    Returns:
        tuple: (x, y, w, h)
    """
    width = np.sqrt(aspect_ratio * scale)
    height = scale / width
    box = (center[0], center[1], width, height)
    return box


def xywh2ltrb(bbox):
    """ convert the coordinates of the box
    
    Args:
        box: represented by [x, y, w, h]

    Returns:
        tensor: box represented by [l, t, r, b]
    """
    lt = bbox[..., :2] - bbox[..., 2:] / 2
    rb = bbox[..., :2] + bbox[..., 2:] / 2
    return torch.cat([lt, rb], dim=1)


def log_time(func):
    """wrapper function for print run time
    """
    @functools.wraps(func)
    def wrapper(*args, desc=''):
        start = time.time()
        print(desc, 'start...')
        ret = func(*args)
        end = time.time()
        print(desc, 'end, spent time: {:.2f}s'.format(end - start))
        return ret

    return wrapper


@log_time
def run(i, mode, targets, preds, iters=200):
    """regression main function

    Args:
        i: i-th random point
        mode: IoU loss type
        targets: ground truth boxes
        preds: predict boxes at i-th random point
    
    Returns:
        No return. Write data to __output/run_{mode}_{i}.txt
    """
    def get_eta(t):
        if t <= 0.8 * iters:
            return 0.1
        elif t <= 0.9 * iters:
            return 0.01
        else:
            return 0.001

    output = []
    loss_func = getattr(losses, mode.lower() + '_loss')
    for p in preds:
        for gt in targets:
            pred_tensor = torch.tensor([p], requires_grad=True)
            target_tensor = torch.tensor([gt], requires_grad=False)
            optim = torch.optim.SGD([pred_tensor], lr=0.0)

            with torch.no_grad():
                initial_loss = losses.l1_loss(pred_tensor,
                                              target_tensor,
                                              reduction='sum').item()
            output.append([0, *p, *gt, initial_loss])

            for t in range(1, iters + 1):
                pred_tensor_ltrb = xywh2ltrb(pred_tensor)
                target_tensor_ltrb = xywh2ltrb(target_tensor)

                if 'LIoU' == mode:
                    loss = loss_func((pred_tensor_ltrb, pred_tensor_ltrb),
                                     (target_tensor_ltrb, target_tensor_ltrb),
                                     reduction='sum')
                else:
                    loss = loss_func(pred_tensor_ltrb,
                                     target_tensor_ltrb,
                                     reduction='sum')

                optim.zero_grad()
                loss.backward()
                for g in optim.param_groups:
                    with torch.no_grad():
                        iou = bbox_overlaps(pred_tensor_ltrb,
                                            target_tensor_ltrb,
                                            is_aligned=True).item()
                    g['lr'] = get_eta(t) * (2 - iou)
                optim.step()

                with torch.no_grad():
                    l1_loss = losses.l1_loss(pred_tensor,
                                             target_tensor,
                                             reduction='sum').item()
                output.append([t, *(pred_tensor.tolist()[0]), *gt, l1_loss])

    os.makedirs('output', exist_ok=True)
    with open(f'output/run_{mode}_{i}.txt', 'w', encoding='utf-8') as f:
        for ele in output:
            f.write(','.join(map(str, ele)) + '\n')


@log_time
def gather(mode):
    output = f'output/loss_{mode}.txt'
    with open(output, 'w', encoding='utf-8') as out:
        for i in tqdm(range(preds_num)):
            iter2loss = [0.0 for _ in range(iters + 1)]
            filename = f'output/run_{mode}_{i}.txt'
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    segs = line.split(',')
                    iter, loss = int(segs[0]), float(segs[-1])
                    iter2loss[iter] += loss

            out.write(','.join(map(str, iter2loss)) + '\n')


@log_time
def main():
    centers_file = 'preds_center.txt'
    if os.path.isfile(centers_file):
        preds_center = np.loadtxt(centers_file, delimiter=",").tolist()
    else:
        preds_center = random_preds_center(center, radius, preds_num)
        with open(centers_file, 'w', encoding='utf-8') as f:
            for c in preds_center:
                f.write(f'{c[0]},{c[1]}\n')

    targets = []
    for ar in aspect_ratios:
        targets.append(get_box_xywh(center, 1.0, ar))

    preds = []
    for c in preds_center:
        cur_preds = []
        for s in scales:
            for ar in aspect_ratios:
                cur_preds.append(get_box_xywh(c, s, ar))
        preds.append(cur_preds)

    for m in modes:
        threads_pool = mp.Pool(os.cpu_count())
        for i, p in enumerate(preds):
            threads_pool.apply_async(run, (i, m, targets, p, iters),
                                     {'desc': f'{m} {i}'})
        threads_pool.close()
        threads_pool.join()

        gather(m, desc=f'gather {m}')


if __name__ == '__main__':
    main(desc='main')
