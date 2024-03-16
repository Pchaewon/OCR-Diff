import os
import yaml
import argparse
from IPython import embed
from easydict import EasyDict

from interfaces.super_resolution import TextSR
# --- main ---#
def main(config, args):
    Mission = TextSR(config, args) # TextSR
    if args.test:
        Mission.test()
    elif args.demo:
        Mission.demo()
    else:
        Mission.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--arch', default='')
    parser.add_argument('--exp_name',  default='TEST1')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default="8", help='')
    parser.add_argument('--resume', type=str, default='', help='')
    parser.add_argument('--rec', default='crnn', choices=['crnn', 'aster', 'moran', 'cdist'])
    parser.add_argument('--syn', action='store_true', default=False, help='use synthetic LR')
    parser.add_argument('--mixed', action='store_true', default=False, help='mix synthetic with real LR')
    parser.add_argument('--mask', action='store_true', default=False, help='')
    parser.add_argument('--hd_u', type=int, default=32, help='')
    parser.add_argument('--srb', type=int, default=5, help='')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--demo_dir', type=str, default='./demo')
    parser.add_argument('--max_updates', type=int, default=300000)
    args = parser.parse_args()
    config_path = os.path.join('config', 'super_resolution.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config, args)