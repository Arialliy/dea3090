from utils.data import *
from utils.metric import *
from argparse import ArgumentParser, ArgumentTypeError
import torch
import torch.utils.data as Data
from model.MSHNet import *
from model.loss import *
from torch.optim import Adagrad
from tqdm import tqdm
import os.path as osp
import os
import time
import glob
import random
import numpy as np

PROJECT_DIR = osp.dirname(osp.abspath(__file__))
DEFAULT_DATASET_DIR = osp.join(PROJECT_DIR, 'datasets', 'IRSTD-1K')
DEFAULT_WEIGHT_DIR = osp.join(PROJECT_DIR, 'weight')

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', '1', 'y'):
        return True
    if value in ('no', 'false', 'f', '0', 'n'):
        return False
    raise ArgumentTypeError('Boolean value expected.')

def load_torch_file(path):
    try:
        return torch.load(path, weights_only=False)
    except TypeError:
        return torch.load(path)

def get_dea_ramp(epoch, warm_epoch, ramp_epochs):
    if ramp_epochs <= 0:
        return 1.0
    if epoch <= warm_epoch:
        return 0.0
    return min(1.0, float(epoch - warm_epoch) / float(ramp_epochs))

def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def parse_args():

    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of model')

    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--warm-epoch', type=int, default=5)

    parser.add_argument('--base-size', type=int, default=256)
    parser.add_argument('--crop-size', type=int, default=256)
    parser.add_argument('--multi-gpus', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--gpu-ids', type=str, default='')
    parser.add_argument('--pin-memory', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--if-checkpoint', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--reset-optimizer', type=str2bool, nargs='?', const=True, default=False)

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--weight-path', type=str, default=osp.join(DEFAULT_WEIGHT_DIR, 'IRSTD-1k_weight.tar'))
    parser.add_argument('--checkpoint-dir', type=str, default='')
    parser.add_argument('--dea-lambda-single', type=float, default=0.0)
    parser.add_argument('--dea-lambda-dec', type=float, default=0.0)
    parser.add_argument('--dea-lambda-empty', type=float, default=0.0)
    parser.add_argument('--dea-tau', type=float, default=0.5)
    parser.add_argument('--dea-ramp-epochs', type=int, default=0)
    parser.add_argument('--save-dea-debug', action='store_true')
    parser.add_argument('--dea-debug-interval', type=int, default=50)
    parser.add_argument('--dea-debug-max-batches', type=int, default=1)
    parser.add_argument('--dea-detach-evidence', action='store_true')
    parser.add_argument('--seed', type=int, default=20260706)
    parser.add_argument('--deterministic', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pd-fa-min-pd', type=float, default=0.93)
    parser.add_argument('--pd-fa-min-iou', type=float, default=0.655)
    parser.add_argument('--paired-baseline-iou', type=float, default=0.0)
    parser.add_argument('--pd-fa-iou-margin', type=float, default=0.005)

    args = parser.parse_args()
    return args

class Trainer(object):
    def __init__(self, args):
        assert args.mode == 'train' or args.mode == 'test'

        self.args = args
        self.start_epoch = 0   
        self.mode = args.mode

        trainset = IRSTD_Dataset(args, mode='train')
        valset = IRSTD_Dataset(args, mode='val')

        data_generator = torch.Generator()
        data_generator.manual_seed(args.seed)

        loader_kwargs = {
            "num_workers": args.num_workers,
            "pin_memory": args.pin_memory,
            "persistent_workers": args.num_workers > 0,
            "worker_init_fn": seed_worker,
            "generator": data_generator,
        }
        if args.num_workers > 0:
            loader_kwargs["prefetch_factor"] = 2

        self.train_loader = Data.DataLoader(
            trainset,
            args.batch_size,
            shuffle=True,
            drop_last=True,
            **loader_kwargs,
        )
        self.val_loader = Data.DataLoader(
            valset,
            1,
            drop_last=False,
            **loader_kwargs,
        )

        device = torch.device('cuda')
        self.device = device
        torch.backends.cudnn.benchmark = not args.deterministic

        model = MSHNet(3)

        if args.multi_gpus and torch.cuda.device_count() > 1:
            device_ids = self.parse_gpu_ids(args.gpu_ids)
            print('use %d gpus: %s' % (len(device_ids), device_ids))
            model = nn.DataParallel(model, device_ids=device_ids)
        model.to(device)
        self.model = model

        self.optimizer = Adagrad(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        self.down = nn.MaxPool2d(2, 2)
        self.loss_fun = SLSIoULoss()
        self.PD_FA = PD_FA(1, 10, args.base_size)
        self.mIoU = mIoU(1)
        self.ROC  = ROCMetric(1, 10)
        self.best_iou = 0.0
        self.best_pd_fa = float('inf')
        self.best_pd_fa_iou = 0.0
        self.best_pd_fa_pd = 0.0
        self.best_pd_fa_epoch = -1
        self.warm_epoch = args.warm_epoch

        if args.mode=='train':
            if args.if_checkpoint:
                check_folder = args.checkpoint_dir or self.find_latest_checkpoint_folder()
                checkpoint = load_torch_file(osp.join(check_folder, 'checkpoint.pkl'))
                self.load_model_state(checkpoint['net'])
                if args.reset_optimizer:
                    print('reset optimizer state')
                else:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer'])
                    except (ValueError, RuntimeError) as exc:
                        print('skip optimizer state: %s' % exc)
                self.set_optimizer_lr(args.lr)
                self.start_epoch = checkpoint.get('epoch', -1) + 1
                self.best_iou = float(checkpoint.get('best_iou', checkpoint.get('iou', 0.0)))
                self.best_pd_fa = float(checkpoint.get('best_pd_fa', float('inf')))
                self.best_pd_fa_iou = float(checkpoint.get('best_pd_fa_iou', 0.0))
                self.best_pd_fa_pd = float(checkpoint.get('best_pd_fa_pd', 0.0))
                self.best_pd_fa_epoch = int(checkpoint.get('best_pd_fa_epoch', -1))
                self.save_folder = check_folder
            else:
                self.save_folder = osp.join(
                    DEFAULT_WEIGHT_DIR,
                    'MSHNet-%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))),
                )
                os.makedirs(self.save_folder, exist_ok=True)
        if args.mode=='test':
          
            weight = load_torch_file(args.weight_path)
            state_dict = self.extract_state_dict(weight)
            self.load_model_state(state_dict)
            '''
                # iou_67.87_weight
                weight = torch.load(args.weight_path)
                self.model.load_state_dict(weight)
            '''
            self.warm_epoch = -1

    def parse_gpu_ids(self, gpu_ids):
        if gpu_ids:
            device_ids = [int(item) for item in gpu_ids.split(',') if item.strip()]
        else:
            device_ids = list(range(torch.cuda.device_count()))
        if not device_ids:
            raise ValueError('No GPU ids selected.')
        return device_ids

    def extract_state_dict(self, weight_obj):
        if isinstance(weight_obj, dict):
            if 'state_dict' in weight_obj:
                return weight_obj['state_dict']
            if 'net' in weight_obj:
                return weight_obj['net']

            looks_like_state_dict = all(
                torch.is_tensor(value) for value in weight_obj.values()
            )
            if looks_like_state_dict:
                return weight_obj

        raise RuntimeError(
            'Unsupported weight format. Expected raw state_dict, '
            'dict with state_dict, or dict with net.'
        )

    def load_model_state(self, state_dict):
        try:
            self.model.load_state_dict(state_dict)
            return
        except RuntimeError:
            pass

        if isinstance(self.model, nn.DataParallel):
            try:
                self.model.module.load_state_dict(state_dict)
                return
            except RuntimeError:
                pass

        has_module_prefix = all(key.startswith('module.') for key in state_dict.keys())
        if has_module_prefix:
            state_dict = {key[len('module.'):]: value for key, value in state_dict.items()}
            target_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            target_model.load_state_dict(state_dict)
            return

        raise RuntimeError('Failed to load model state_dict.')

    def set_optimizer_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        print('set optimizer lr: %.6f' % lr)

    def find_latest_checkpoint_folder(self):
        checkpoint_paths = sorted(
            glob.glob(osp.join(DEFAULT_WEIGHT_DIR, 'MSHNet-*', 'checkpoint.pkl')),
            key=osp.getmtime,
        )
        if not checkpoint_paths:
            raise FileNotFoundError(
                'No checkpoint found under %s. Pass --checkpoint-dir inside the project weight directory.' % DEFAULT_WEIGHT_DIR
            )
        return osp.dirname(checkpoint_paths[-1])
        
    def use_dea(self, epoch):
        return (
            epoch > self.warm_epoch
            and (
                self.args.dea_lambda_single > 0
                or self.args.dea_lambda_dec > 0
                or self.args.dea_lambda_empty > 0
            )
        )

    def save_dea_debug(self, epoch, iteration, data, labels, pred, dea_out):
        if not self.args.save_dea_debug:
            return
        if iteration >= self.args.dea_debug_max_batches:
            return
        if self.args.dea_debug_interval > 0 and epoch % self.args.dea_debug_interval != 0:
            return

        debug_root = self.save_folder if self.save_folder else PROJECT_DIR
        debug_dir = osp.join(debug_root, 'dea_debug')
        os.makedirs(debug_dir, exist_ok=True)

        sample = {
            "image": data[:1].detach().cpu(),
            "label": labels[:1].detach().cpu(),
            "z_full": pred[:1].detach().cpu(),
            "p_full": torch.sigmoid(pred[:1]).detach().cpu(),
            "scale_logits": dea_out["scale_logits"][:1].detach().cpu(),
            "z_only": dea_out["z_only"][:1].detach().cpu(),
            "p_only": torch.sigmoid(dea_out["z_only"][:1]).detach().cpu(),
            "z_only_max": dea_out["z_only_max"][:1].detach().cpu(),
            "p_only_max": torch.sigmoid(dea_out["z_only_max"][:1]).detach().cpu(),
            "z_empty": dea_out["z_empty"][:1].detach().cpu(),
            "p_empty": torch.sigmoid(dea_out["z_empty"][:1]).detach().cpu(),
            "d_logit": dea_out["decidability_logit"][:1].detach().cpu(),
            "d_prob": torch.sigmoid(dea_out["decidability_logit"][:1]).detach().cpu(),
        }

        torch.save(sample, osp.join(debug_dir, 'epoch_%04d_iter_%04d.pt' % (epoch, iteration)))

    def train(self, epoch):
        self.model.train()
        tbar = tqdm(self.train_loader)
        losses = AverageMeter()
        for i, (data, mask) in enumerate(tbar):
  
            data = data.to(self.device, non_blocking=True)
            labels = mask.to(self.device, non_blocking=True)

            tag = epoch > self.warm_epoch
            use_dea = self.use_dea(epoch)

            if use_dea:
                masks, pred, dea_out = self.model(
                    data,
                    tag,
                    return_dea=True,
                    dea_detach_evidence=self.args.dea_detach_evidence,
                )
            else:
                masks, pred = self.model(data, tag)
                dea_out = None

            loss = 0

            loss = loss + self.loss_fun(pred, labels, self.warm_epoch, epoch)
            labels_for_scale = labels
            for j in range(len(masks)):
                if j>0:
                    labels_for_scale = self.down(labels_for_scale)
                loss = loss + self.loss_fun(masks[j], labels_for_scale, self.warm_epoch, epoch)
                
            loss = loss / (len(masks)+1)
            loss_seg_for_debug = loss.detach()

            if use_dea:
                ramp = get_dea_ramp(epoch, self.warm_epoch, self.args.dea_ramp_epochs)
                cur_lambda_single = self.args.dea_lambda_single * ramp
                cur_lambda_dec = self.args.dea_lambda_dec * ramp
                cur_lambda_empty = self.args.dea_lambda_empty * ramp

                loss_dea, dea_log = dea_lite_loss(
                    dea_out=dea_out,
                    z_full=pred,
                    gt=labels,
                    lambda_single=cur_lambda_single,
                    lambda_dec=cur_lambda_dec,
                    lambda_empty=cur_lambda_empty,
                    tau=self.args.dea_tau,
                )
                loss = loss + loss_dea
                self.save_dea_debug(epoch, i, data, labels, pred, dea_out)

                if self.args.save_dea_debug and self.args.dea_debug_interval > 0 and i % self.args.dea_debug_interval == 0:
                    dea_ratio = (loss_dea.detach() / (loss_seg_for_debug + 1e-6)).item()
                    msg = [
                        'dea_ratio=%.4f' % dea_ratio,
                        'lambda_single=%.6f' % cur_lambda_single,
                        'lambda_empty=%.6f' % cur_lambda_empty,
                        'lambda_dec=%.6f' % cur_lambda_dec,
                    ]
                    for key, value in dea_log.items():
                        try:
                            msg.append('%s=%.6f' % (key, float(value)))
                        except (TypeError, ValueError):
                            pass
                    print('[DEA DEBUG] ' + ' | '.join(msg))
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
       
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, loss %.4f' % (epoch, losses.avg))
    
    def test(self, epoch):
        self.model.eval()
        self.mIoU.reset()
        self.PD_FA.reset()
        tbar = tqdm(self.val_loader)
        tag = False
        with torch.no_grad():
            for i, (data, mask) in enumerate(tbar):
    
                data = data.to(self.device, non_blocking=True)
                mask = mask.to(self.device, non_blocking=True)

                if epoch>self.warm_epoch:
                    tag = True

                loss = 0
                _, pred = self.model(data, tag)
                # loss += self.loss_fun(pred, mask,self.warm_epoch, epoch)

                self.mIoU.update(pred, mask)
                self.PD_FA.update(pred, mask)
                self.ROC.update(pred, mask)
                _, mean_IoU = self.mIoU.get()

                tbar.set_description('Epoch %d, IoU %.4f' % (epoch, mean_IoU))
            FA, PD = self.PD_FA.get(len(self.val_loader))
            _, mean_IoU = self.mIoU.get()
            ture_positive_rate, false_positive_rate, _, _ = self.ROC.get()

            
            if self.mode == 'train':
                current_pd = PD[0]
                current_fa = FA[0] * 1000000
                if self.args.paired_baseline_iou > 0:
                    pd_fa_iou_threshold = max(
                        self.args.pd_fa_min_iou,
                        self.args.paired_baseline_iou - self.args.pd_fa_iou_margin,
                    )
                else:
                    pd_fa_iou_threshold = self.args.pd_fa_min_iou

                is_pd_fa_candidate = (
                    current_pd >= self.args.pd_fa_min_pd
                    and mean_IoU >= pd_fa_iou_threshold
                    and current_fa < self.best_pd_fa
                )
                metric_line = '{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n'.format(
                    time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())),
                    epoch,
                    mean_IoU,
                    current_pd,
                    current_fa,
                )
                print(metric_line.strip())
                with open(osp.join(self.save_folder, 'epoch_metric.log'), 'a') as f:
                    f.write(metric_line)

                if mean_IoU > self.best_iou:
                    self.best_iou = mean_IoU
                
                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight.pkl'),
                    )

                    best_iou_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                    }
                    torch.save(
                        best_iou_states,
                        osp.join(self.save_folder, 'checkpoint_best_iou.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, self.best_iou, current_pd, current_fa))

                if is_pd_fa_candidate:
                    self.best_pd_fa = current_fa
                    self.best_pd_fa_iou = mean_IoU
                    self.best_pd_fa_pd = current_pd
                    self.best_pd_fa_epoch = epoch

                    torch.save(
                        self.model.state_dict(),
                        osp.join(self.save_folder, 'weight_pd_fa_best.pkl'),
                    )

                    pd_fa_states = {
                        "net": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                        "iou": mean_IoU,
                        "pd": current_pd,
                        "fa": current_fa,
                        "best_iou": self.best_iou,
                        "best_pd_fa": self.best_pd_fa,
                        "best_pd_fa_iou": self.best_pd_fa_iou,
                        "best_pd_fa_pd": self.best_pd_fa_pd,
                        "best_pd_fa_epoch": self.best_pd_fa_epoch,
                    }
                    torch.save(
                        pd_fa_states,
                        osp.join(self.save_folder, 'checkpoint_pd_fa_best.pkl'),
                    )

                    with open(osp.join(self.save_folder, 'metric_pd_fa_best.log'), 'a') as f:
                        f.write('{} - {:04d}\t - IoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, mean_IoU, current_pd, current_fa))
                        
                latest_states = {
                    "net": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "iou": mean_IoU,
                    "pd": current_pd,
                    "fa": current_fa,
                    "best_iou": self.best_iou,
                    "best_pd_fa": self.best_pd_fa,
                    "best_pd_fa_iou": self.best_pd_fa_iou,
                    "best_pd_fa_pd": self.best_pd_fa_pd,
                    "best_pd_fa_epoch": self.best_pd_fa_epoch,
                }
                torch.save(latest_states, osp.join(self.save_folder, 'checkpoint.pkl'))
            elif self.mode == 'test':
                print('mIoU: '+str(mean_IoU)+'\n')
                print('Pd: '+str(PD[0])+'\n')
                print('Fa: '+str(FA[0]*1000000)+'\n')


         
if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed, args.deterministic)

    trainer = Trainer(args)
    
    if trainer.mode=='train':
        for epoch in range(trainer.start_epoch, args.epochs):
            trainer.train(epoch)
            trainer.test(epoch)
    else:
        trainer.test(1)
 
