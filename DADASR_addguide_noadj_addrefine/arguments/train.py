import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

# general
parser.add_argument('--save-dir', required=True, help='Path to directory where models and logs should be saved')
parser.add_argument('--logstep-train', default=10, type=int, help='Training log interval in steps')
parser.add_argument('--save-model', default='both', choices=['last', 'best', 'no', 'both'])
parser.add_argument('--val-every-n-epochs', type=int, default=1, help='Validation interval in epochs')
parser.add_argument('--resume', type=str, default=None, help='Checkpoint path to resume')
parser.add_argument('--seed', type=int, default=12345, help='Random seed')

# data
parser.add_argument('--data-dir', type=str, default='ProcessedData_scale10', help='Root directory of ProcessedData_scale10')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=14)
parser.add_argument('--crop-size', type=int, default=250, help='Size of the HR input patches, divisible by scaling')
parser.add_argument('--scaling', type=int, default=10, help='Scaling factor')
parser.add_argument('--max-rotation', type=float, default=0., help='Maximum rotation angle (degrees)')
parser.add_argument('--no-flip', action='store_true', default=False, help='Switch off random flipping')
parser.add_argument('--in-memory', action='store_true', default=False, help='Hold data in memory during training')

# training
parser.add_argument('--loss', default='rmse', type=str, choices=['l1', 'rmse'])
parser.add_argument('--num-epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--w-decay', type=float, default=1e-5)
parser.add_argument('--lr-scheduler', type=str, default='step', choices=['no', 'step', 'plateau'])
parser.add_argument('--lr-step', type=int, default=100, help='LR scheduler step size (epochs)')
parser.add_argument('--lr-gamma', type=float, default=0.9, help='LR decay rate')
parser.add_argument('--skip-first', action='store_true', help='Don\'t optimize during first epoch')
parser.add_argument('--gradient-clip', type=float, default=0.01, help='If > 0, clips gradient norm to that value')
parser.add_argument('--no-opt', action='store_true', help='Don\'t optimize')

# model
parser.add_argument('--feature-extractor', type=str, default='UNet', help="Feature extractor for edge potentials. 'none' for the unlearned version.") 
parser.add_argument('--Npre', type=int, default=4000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=512, help='N learned iterations with gradients')
parser.add_argument('--use-refinement-net', dest='use_refinement_net', action='store_true', default=True, help='Use the Real-GDSR-style local residual refinement network before diffusion')
parser.add_argument('--no-refinement-net', dest='use_refinement_net', action='store_false', help='Disable local refinement and run diffusion from bicubic DSM')
parser.add_argument('--refinement-channels', type=int, default=64, help='Hidden channel count in the local refinement network')
parser.add_argument('--refinement-blocks', type=int, default=4, help='Number of residual blocks in the local refinement network')
parser.add_argument('--refinement-only', action='store_true', default=False, help='Train/evaluate only local refinement and skip the diffusion loop')
