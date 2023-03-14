import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import random

from PIL import Image

from torch.utils.data import DataLoader
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch.optim import SGD, lr_scheduler
from project_utils.cluster_utils import mixed_eval, AverageMeter
from models import vision_transformer as vits

from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights

from data.augmentations import get_transform
from data.get_datasets import get_datasets, get_class_splits

from tqdm import tqdm

from torch.nn import functional as F
import torch.nn as nn

from project_utils.cluster_and_log_utils import log_accs_from_preds
from config import exp_root, dino_pretrain_path

# TODO: Debug
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def transform_invert(img, show=False):
    # Tensor -> PIL.Image
    # 注意：img.shape = [3,32,32] cifar10中的一张图片，经过transform后的tensor格式

    if img.dim() == 3:  # single image # 3,32,32
        img = img.unsqueeze(0)         #在第0维增加一个维度 1,3,32,32
    low = float(img.min())
    high = float(img.max())
    # img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))   # (img - low)/(high-low)
    grid = img.squeeze(0)  #去除维度为1的维度
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    if show:
        img.show()
    return img

def test_on_the_fly(model, projection_head, test_loader,
                epoch, save_name,
                args):

    model.eval()
    projection_head.eval()

    all_feats = []
    targets = np.array([])
    mask = np.array([])

    print('Collating features...')
    # First extract all features
    for batch_idx, (images, label, _) in enumerate(tqdm(test_loader)):
        
        if batch_idx > 1000:
            break

        images = images.cuda()

        feats = model(images)
        _, feats, _ = projection_head(feats)
        
        img = transform_invert(images)
        
        for i in range(12):
            if feats[0,i] > 0:
                img.save("cifar100_vis/{}/{}.jpg".format(i, batch_idx))

        print(batch_idx, label)

class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=1, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.5)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

        
    # ----------------------
    # INIT
    # ----------------------
    seed_torch(0)
    args = parser.parse_args()
    device = torch.device('cuda:0')
    args = get_class_splits(args)

    args.num_labeled_classes = len(args.train_classes)
    args.num_unlabeled_classes = len(args.unlabeled_classes)

    init_experiment(args, runner_name=['metric_learn_gcd'])
    print(f'Using evaluation function {args.eval_funcs[0]} to print results')

    # ----------------------
    # BASE MODEL
    # ----------------------
    if args.base_model == 'vit_dino':

        args.interpolation = 3
        args.crop_pct = 0.875
        pretrain_path = dino_pretrain_path

        model = vits.__dict__['vit_base']()

        state_dict = torch.load(pretrain_path, map_location='cpu')
        model.load_state_dict(state_dict)

        if args.warmup_model_dir is not None:
            print(f'Loading weights from {args.warmup_model_dir}')
            model.load_state_dict(torch.load(args.warmup_model_dir, map_location='cpu'))

        model.to(device)

        # NOTE: Hardcoded image size as we do not finetune the entire ViT model
        args.image_size = 224
        args.feat_dim = 768
        args.num_mlp_layers = 3
        args.code_dim = 12
        args.mlp_out_dim = None
        args.grad_from_block = 11


        # ----------------------
        # HOW MUCH OF BASE MODEL TO FINETUNE
        # ----------------------
        for m in model.parameters():
            m.requires_grad = False

        # Only finetune layers from block 'args.grad_from_block' onwards
        for name, m in model.named_parameters():
            if 'block' in name:
                block_num = int(name.split('.')[1])
#                 print(name, block_num)
                if block_num >= args.grad_from_block:
                    m.requires_grad = True

    else:

        raise NotImplementedError

    # --------------------
    # CONTRASTIVE TRANSFORM
    # --------------------
    train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
    train_transform = ContrastiveLearningViewGenerator(base_transform=train_transform, n_views=args.n_views)

    # --------------------
    # DATASETS
    # --------------------
    train_dataset, test_dataset, unlabelled_train_examples_test, datasets, labelled_dataset = get_datasets(args.dataset_name,
                                                                                         test_transform,
                                                                                         test_transform,
                                                                                         args)



    # --------------------
    # DATALOADERS
    # --------------------
    labelled_train_loader = DataLoader(labelled_dataset, num_workers=args.num_workers, batch_size=1, 
                              shuffle=True, drop_last=True)
    unlabelled_train_loader = DataLoader(unlabelled_train_examples_test, num_workers=args.num_workers,
                                        batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers,
                                      batch_size=1, shuffle=True)

    # ----------------------
    # PROJECTION HEAD
    # ----------------------
    projection_head = vits.__dict__['HASHHead'](in_dim=args.feat_dim,
                               out_dim=args.mlp_out_dim, nlayers=args.num_mlp_layers, code_dim=args.code_dim, class_num=args.num_labeled_classes)
    projection_head.to(device)

    log_name = "(29.01.2023_|_34.161)"

    ckpt = torch.load("metric_learn_gcd/log/" + log_name + "/checkpoints/model_best.pt", map_location='cpu')
    model.load_state_dict(ckpt)
    ckpt = torch.load("metric_learn_gcd/log/" + log_name + "/checkpoints/model_proj_head_best.pt")
    projection_head.load_state_dict(ckpt)
#     print(nn.Tanh()(projection_head.t), nn.Tanh()(projection_head.t).abs().mean())
    print(projection_head.center, torch.abs(projection_head.center).mean())

    with torch.no_grad():
        test_on_the_fly(model, projection_head, labelled_train_loader,
                                                epoch=0, save_name='Train ACC Unlabelled',
                                                args=args)