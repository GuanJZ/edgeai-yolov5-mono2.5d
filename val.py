"""Validate a trained YOLOv5 model accuracy on a custom dataset

Usage:
    $ python path/to/val.py --data coco128.yaml --weights yolov5s.pt --img 640
"""

import argparse
import json
import os
import sys
from pathlib import Path
from threading import Thread

import numpy as np
import torch
import yaml
from tqdm import tqdm
import shutil

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add yolov5/ to path

from models.experimental import attempt_load
from utils.datasets import create_dataloader
from utils.general import coco80_to_coco91_class, check_dataset, check_file, check_img_size, check_requirements, \
    box_iou, non_max_suppression, scale_coords, xyxy2xywh, xywh2xyxy, set_logging, increment_path, colorstr
from utils.metrics import ap_per_class, ConfusionMatrix
from utils.plots import plot_images, output_to_target, plot_study_txt
from utils.torch_utils import select_device, time_sync
from utils.loggers import Loggers


def save_one_txt(predn, save_conf, shape, file):
    # Save one txt result
    gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
    for *xyxy, conf, cls in predn.tolist():
        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
        with open(file, 'a') as f:
            f.write(('%g ' * len(line)).rstrip() % line + '\n')


def save_one_json(predn, jdict, path, class_map):
    # Save one JSON result {"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}
    image_id = int(path.stem) if path.stem.isnumeric() else path.stem
    box = xyxy2xywh(predn[:, :4])  # xywh
    box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
    for p, b in zip(predn.tolist(), box.tolist()):
        jdict.append({'image_id': image_id,
                      'category_id': class_map[int(p[5])],
                      'bbox': [round(x, 3) for x in b],
                      'score': round(p[4], 5)})

def calc_theta_ray(width, box_2d, proj_matrix):
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0, 0]))
    center = (box_2d[:, 2] + box_2d[:, 0]) / 2
    dx = center - (width / 2)

    if dx.shape[0] == 1:
        mult = -np.ones(dx.shape) if dx[0] < 0 else -np.ones(dx.shape)
    else:
        mult = np.ones(dx.shape)
        mult[dx < 0] = -1
    dx = np.abs(dx)
    theta = np.arctan((2 * dx * np.tan(fovx / 2)) / width)
    theta = theta * mult

    return theta

def compute_location(detections, labels):
    """
    iou>=0.5的情况下,detetcions 匹配到labels,从而得到 detections的location
    :param detections:
    :param labels:
    :return:
    """
    iou = box_iou(detections[:, :4], labels[:, 1:5])
    x = torch.where(iou >= 1e-6)
    matches = None
    if x[0].shape[0]:
        matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
        if x[0].shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            # matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            # matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
    return matches

def process_batch(predictions, labels, iouv):
    # Evaluate 1 batch of predictions
    correct = torch.zeros(predictions.shape[0], len(iouv), dtype=torch.bool, device=iouv.device)
    detected = []  # label indices
    tcls, pcls = labels[:, 0], predictions[:, 5]
    nl = labels.shape[0]  # number of labels
    for cls in torch.unique(tcls):
        ti = (cls == tcls).nonzero().view(-1)  # label indices
        pi = (cls == pcls).nonzero().view(-1)  # prediction indices
        if pi.shape[0]:  # find detections
            ious, i = box_iou(predictions[pi, 0:4], labels[ti, 1:5]).max(1)  # best ious, indices
            detected_set = set()
            for j in (ious > iouv[0]).nonzero():
                d = ti[i[j]]  # detected label
                if d.item() not in detected_set:
                    detected_set.add(d.item())
                    detected.append(d)  # append detections
                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                    if len(detected) == nl:  # all labels already located in image
                        break
    return correct


@torch.no_grad()
def run(data,
        weights=None,  # model.pt path(s)
        batch_size=32,  # batch size
        imgsz=640,  # inference size (pixels)
        conf_thres=0.001,  # confidence threshold
        iou_thres=0.6,  # NMS IoU threshold
        task='val',  # train, val, test, speed or study
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        single_cls=False,  # treat as single-class dataset
        augment=False,  # augmented inference
        verbose=False,  # verbose output
        save_txt=False,  # save results to *.txt
        save_hybrid=False,  # save label+prediction hybrid results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_json=False,  # save a COCO-JSON results file
        project='runs/val',  # save to project/name
        name='exp',  # save to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        half=True,  # use FP16 half-precision inference
        model=None,
        dataloader=None,
        save_dir=Path(''),
        plots=True,
        loggers=Loggers(),
        compute_loss=None,
        do_3d=False,
        ):
    # Initialize/load model and set device
    training = model is not None
    if training:  # called by train.py
        device = next(model.parameters()).device  # get model device

    else:  # called directly
        device = select_device(device, batch_size=batch_size)

        # Directories
        save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        gs = max(int(model.stride.max()), 32)  # grid size (max stride)
        imgsz = check_img_size(imgsz, s=gs)  # check image size

        # Multi-GPU disabled, incompatible with .half() https://github.com/ultralytics/yolov5/issues/99
        # if device.type != 'cpu' and torch.cuda.device_count() > 1:
        #     model = nn.DataParallel(model)

        # Data
        with open(data, encoding='ascii', errors='ignore') as f:
            data = yaml.safe_load(f)
        check_dataset(data)  # check

    if do_3d:
        # 3d predicts
        preds_3d, labels_3d, img_paths = [], [], []
        save_pred_3d = True
        if save_pred_3d:
            pred_3d_save_dir = os.path.join(save_dir, "pred_results")
            if os.path.exists(pred_3d_save_dir):
                shutil.rmtree(pred_3d_save_dir)
            os.makedirs(pred_3d_save_dir)

    # Half
    half &= device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Configure
    model.eval()
    is_coco = type(data['val']) is str and data['val'].endswith('coco/val2017.txt')  # COCO dataset
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if not training:
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        task = task if task in ('train', 'val', 'test') else 'val'  # path to train/val/test images
        dataloader = create_dataloader(data[task], imgsz, batch_size, gs, single_cls, pad=0.5, rect=True,
                                       prefix=colorstr(f'{task}: '))[0]

    seen = 0
    confusion_matrix = ConfusionMatrix(nc=nc)
    names = {k: v for k, v in enumerate(model.names if hasattr(model, 'names') else model.module.names)}
    class_map = coco80_to_coco91_class() if is_coco else list(range(1000))
    s = ('%20s' + '%11s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
    p, r, f1, mp, mr, map50, map, t0, t1, t2 = 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(6, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (img, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        t_ = time_sync()
        img = img.to(device, non_blocking=True)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = img.shape  # batch size, channels, height, width
        t = time_sync()
        t0 += t - t_

        # Run model
        out, train_out = model(img, augment=augment)  # inference and training outputs
        t1 += time_sync() - t

        # Compute loss
        if compute_loss:
            loss += compute_loss([x.float() for x in train_out], targets)[1]  # box, obj, cls

        # Run NMS
        targets[:, 2:6] *= torch.Tensor([width, height, width, height]).to(device)  # to pixels
        lb = [targets[targets[:, 0] == i, 1:6] for i in range(nb)] if save_hybrid else []  # for autolabelling
        t = time_sync()
        out = non_max_suppression(out, conf_thres, iou_thres, labels=lb, multi_label=True, agnostic=single_cls)
        t2 += time_sync() - t

        # Statistics per image
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            path, shape = Path(paths[si]), shapes[si][0]
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Predictions
            if single_cls:
                pred[:, 5] = 0
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shape, shapes[si][1])  # native-space pred

            # 1. hwl_log --> hwl
            # predn
            # from [x,y,x,y,conf,cls,h_log,w_log,l_log, cos1,sin1,cos2,sin2,conf1,conf2]
            # to
            # [x,y,x,y,conf,cls,h,w,l, cos1,sin1,cos2,sin2,conf1,conf2]
            # predn[:, 6:9] = torch.exp(predn[:, 6:9]) 后处理已经操作, 这里是多余的

            # 1.2 将角度normalize


            # 2. theta
            img_width = shapes[si][0][1]
            intrinsic_path = paths[si].replace("images", "calibs").replace("jpg", "txt")
            with open(intrinsic_path, 'r')as f:
                parse_file = f.read().strip().splitlines()
                for line in parse_file:
                    if line is not None and line.split()[0] == "P2:":
                        proj_matrix = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
            theta = calc_theta_ray(img_width, predn[:, :4].cpu().numpy(), proj_matrix)

            # 3. alpha
            orint, conf = predn[:, 9:13], predn[:, 13:15]
            _, conf_idxs = torch.max(conf, dim=1)
            alpha = torch.zeros(conf.shape[0])
            for enum, orient_idx in enumerate(conf_idxs):
                cos, sin = orint[enum, orient_idx * 2], orint[enum, orient_idx * 2 + 1]
                # 因为数据预处理将alpha的区间从[-pi, pi]移动到[0, 2*pi], 所以这里还需要再减去pi, [-pi, pi]
                alpha[enum] = torch.atan2(sin, cos) + (orient_idx + 0.5 - 1) * torch.pi

            # 4. ry
            Ry = torch.tensor(theta) + alpha

            # predn
            # from ([xyxy, conf, cls, (H, W_, L), ([cos1, sin1], [cos2, sin2]), (conf1, conf2)])
            # to
            # [xyxy, conf, cls, (H, W, L), Ry]
            predn_processed = torch.cat((predn[:, :9], Ry[:, None].to(predn.device)), dim=1)

            # Evaluate
            if nl:
                tbox = xywh2xyxy(labels[:, 1:5])  # target boxes
                scale_coords(img[si].shape[1:], tbox, shape, shapes[si][1])  # native-space labels
                # labelsn
                # from [cls, x, y, x, y, h, w, l, X, Y, Z, Ry, cos1, sin1, cos2, sin2, conf1, conf2, truncated, occluded, alpha]
                # to
                # [cls, x, y, x, y, h, w, l, X, Y, Z, Ry]
                labelsn = torch.cat((labels[:, 0:1], tbox, labels[:, 5:12]), 1)  # native-space labels
                correct = process_batch(predn_processed, labelsn, iouv)
                # if plots:
                #     confusion_matrix.process_batch(predn, labelsn)

                if do_3d:
                    matches = compute_location(predn_processed, labelsn)
                    # predn3d
                    # from predn [0:x, 1:y, 2:x, 3:y, 4:conf, 5:cls, 6:H, 7:W, 8:L, 9:Ry, 10:bbcp_x, 11:bbcp_y]
                    # to
                    # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry, conf, bbcp_x, bbcp_y]
                    predn_3d = np.zeros((predn_processed.shape[0], 16))
                    predn_arr = predn_processed.cpu().numpy()
                    labelsn_arr = labelsn.cpu().numpy()
                    if matches is not None:
                        for enum, m in enumerate(matches):
                            predn_3d[enum, 0] = predn_arr[int(m[0]), 5]
                            predn_3d[enum, 4:8] = predn_arr[int(m[0]), :4]
                            predn_3d[enum, 8:11] = predn_arr[int(m[0]), 6:9]
                            predn_3d[enum, 11:14] = labelsn_arr[int(m[1]), 8:11]
                            predn_3d[enum, 14] = predn_arr[int(m[0]), 9]
                            predn_3d[enum, 15] = predn_arr[int(m[0]), 4]
                            # predn_3d[enum, 16:18] = predn_arr[int(m[0]), 10:12]
                    preds_3d.append(predn_3d)
                    save_pred_3d = True
                    if save_pred_3d:
                        pred_3d_save_path = os.path.join(pred_3d_save_dir,
                                                         os.path.basename(paths[si]).replace("jpg", "txt"))
                        np.savetxt(pred_3d_save_path, predn_3d, delimiter=" ", fmt='%.08f')

                    # labelsn_3d
                    # from labelsn (cls, xyxy, HWL, XYZ, Ry, bbcp_x, bbcp_y)
                    # to
                    # (ndarray)[cls, 0, 0, 0, x1, y1, x2, y2, H, W, L, X, Y, Z, Ry, bbcp_x, bbcp_y]
                    labelsn_3d = np.zeros((labelsn.shape[0], 15))
                    labelsn_3d[:, 0] = labelsn_arr[:, 0]
                    labelsn_3d[:, 4:8] = labelsn_arr[:, 1:5]
                    labelsn_3d[:, 8:] = labelsn_arr[:, 5:]

                    labels_3d.append(labelsn_3d)
                    img_paths.append(paths[si])

            else:
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))  # (correct, conf, pcls, tcls)

            # Save/log
            if save_txt:
                save_one_txt(predn, save_conf, shape, file=save_dir / 'labels' / (path.stem + '.txt'))
            if save_json:
                save_one_json(predn, jdict, path, class_map)  # append to COCO-JSON dictionary
            loggers.on_val_batch_end(pred, predn, path, names, img[si])

        # # Plot images
        # if plots and batch_i < 3:
        #     f = save_dir / f'val_batch{batch_i}_labels.jpg'  # labels
        #     Thread(target=plot_images, args=(img, targets, paths, f, names), daemon=True).start()
        #     f = save_dir / f'val_batch{batch_i}_pred.jpg'  # predictions
        #     Thread(target=plot_images, args=(img, output_to_target(out), paths, f, names), daemon=True).start()

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=plots, save_dir=save_dir, names=names)
        # AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() - 1
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%11i' * 2 + '%11.3g' * 4  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map50, map))

    # Print results per class
    if (verbose or (nc < 50 and not training)) and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap50[i], ap[i]))

    # Print speeds
    t = tuple(x / seen * 1E3 for x in (t0, t1, t2))  # speeds per image
    if not training:
        shape = (batch_size, 3, imgsz, imgsz)
        print(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {shape}' % t)

    # Plots
    if plots:
        # confusion_matrix.plot(save_dir=save_dir, names=list(names.values()))
        loggers.on_val_end()

    if do_3d:
        from utils.show_2d3d_box import show_2d3d_box
        # conf_thres = AP50_F1_max_idx / 1000.0
        conf_thres = 0.5
        # conf_thres = 0
        final_preds_3d = [pred[pred[:, 15] >= conf_thres] for pred in preds_3d]

        print("writing 3D BBoxes")
        show_2d3d_box(final_preds_3d, labels_3d, img_paths, data["names"], save_dir, True)

    # Save JSON
    if save_json and len(jdict):
        w = Path(weights[0] if isinstance(weights, list) else weights).stem if weights is not None else ''  # weights
        anno_json = str(Path(data.get('path', '../coco')) / 'annotations/instances_val2017.json')  # annotations json
        pred_json = str(save_dir / f"{w}_predictions.json")  # predictions json
        print(f'\nEvaluating pycocotools mAP... saving {pred_json}...')
        with open(pred_json, 'w') as f:
            json.dump(jdict, f)

        try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            check_requirements(['pycocotools'])
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            anno = COCO(anno_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            if is_coco:
                eval.params.imgIds = [int(Path(x).stem) for x in dataloader.dataset.img_files]  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
        except Exception as e:
            print(f'pycocotools unable to run: {e}')

    # Return results
    model.float()  # for training
    if not training:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map50, map, *(loss.cpu() / len(dataloader)).tolist()), maps, t


def parse_opt():
    parser = argparse.ArgumentParser(prog='val.py')
    parser.add_argument('--data', type=str, default='data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    parser.add_argument('--project', default='runs/val', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--do-3d', action='store_true', help='')
    opt = parser.parse_args()
    opt.save_json |= opt.data.endswith('coco.yaml')
    opt.save_txt |= opt.save_hybrid
    opt.data = check_file(opt.data)  # check file
    return opt


def main(opt):
    set_logging()
    print(colorstr('val: ') + ', '.join(f'{k}={v}' for k, v in vars(opt).items()))
    # check_requirements(exclude=('tensorboard', 'thop'))

    if opt.task in ('train', 'val', 'test'):  # run normally
        run(**vars(opt))

    elif opt.task == 'speed':  # speed benchmarks
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=opt.imgsz, conf_thres=.25, iou_thres=.45,
                save_json=False, plots=False)

    elif opt.task == 'study':  # run over a range of settings and save/plot
        # python val.py --task study --data coco.yaml --iou 0.7 --weights yolov5s.pt yolov5m.pt yolov5l.pt yolov5x.pt
        x = list(range(256, 1536 + 128, 128))  # x axis (image sizes)
        for w in opt.weights if isinstance(opt.weights, list) else [opt.weights]:
            f = f'study_{Path(opt.data).stem}_{Path(w).stem}.txt'  # filename to save to
            y = []  # y axis
            for i in x:  # img-size
                print(f'\nRunning {f} point {i}...')
                r, _, t = run(opt.data, weights=w, batch_size=opt.batch_size, imgsz=i, conf_thres=opt.conf_thres,
                              iou_thres=opt.iou_thres, save_json=opt.save_json, plots=False)
                y.append(r + t)  # results and times
            np.savetxt(f, y, fmt='%10.4g')  # save
        os.system('zip -r study.zip study_*.txt')
        plot_study_txt(x=x)  # plot


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
