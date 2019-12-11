import configargparse
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_params():
    p = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        default_config_files=['default.yaml'])

    p.add('-v', help='verbose', action='store_true')

    p.add('--data-type')
    p.add('--frames', action='append', type=int)
    p.add('--n-patches', type=int)
    p.add('--epochs', type=int)
    p.add('--n-frames-epoch', type=int)
    p.add('--momentum', type=float)
    p.add('--eps', type=float)
    p.add('--lr-decay', type=float)
    p.add('--lr', type=float)
    p.add('--ds-split', type=float)
    p.add('--ds-shuffle', type=bool)
    p.add('--weight-decay', type=float)
    p.add('--batch-size', type=int)
    p.add('--batch-norm', type=bool)
    p.add('--n-workers', type=int)
    p.add('--cuda', default=False, action='store_true')
    p.add('--in-shape', type=int)
    p.add('--depth', type=int)
    p.add('--aug-noise', type=float)
    p.add('--aug-scale', type=float)
    p.add('--aug-rotate', type=float)
    p.add('--aug-shear', type=float)
    p.add('--aug-flip-proba', type=float)
    p.add('--aug-some', type=int)

    return p
