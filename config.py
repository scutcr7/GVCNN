import argparse
import logging
from logging import Formatter
import torch
import os


parser = argparse.ArgumentParser("GVCNN")
parser.add_argument("--data_dir", type=str, default="", help="dataset path")
parser.add_argument('--gpu_device', type=str, default="0", help="gpu id")
parser.add_argument('--seed', type=int, default=6, help="random seed")
parser.add_argument('--mv_backbone', type=str, default='GOOGLENET', help=('ALEXNET', 'VGG13', 'VGG13BN', 'VGG11BN', 'RESNET50', 'GOOGLENET'
                                                                         ,'INCEPTION_V3'))
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--optimizer', type=str, default='SGD', help='SGD, Adam. [default: SGD]')
parser.add_argument('--num_views', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--test_batch_size', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--valid_freq', type=int, default=1)
parser.add_argument('--save_freq', type=int, default=10)
parser.add_argument('--pretrain_model_dir', type=str, default='./pretrain')
parser.add_argument('--save_dir', type=str, default=None, help='The saving directory of training process.')
parser.add_argument('--group_num', type=int, default=8)
args = parser.parse_args()

# Main logger
main_logger = logging.getLogger()
main_logger.setLevel(logging.INFO)
log_console_format = "[%(levelname)s] - %(asctime)s : %(message)s"
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(Formatter(log_console_format))
main_logger.addHandler(console_handler)
logger = logging.getLogger()

# device
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
torch.cuda.manual_seed_all(args.seed)
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
