import configargparse

parser = configargparse.ArgumentParser()
parser.add_argument('-c', '--config', is_config_file=True, help='Path to the config file', type=str)

parser.add_argument('--checkpoint', type=str, required=True, help='Checkpoint path to evaluate')
parser.add_argument('--data-dir', type=str, default='ProcessedData_scale10', help='Root directory of ProcessedData_scale10')
parser.add_argument('--num-workers', type=int, default=8, metavar='N', help='Number of dataloader worker processes')
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--crop-size', type=int, default=0, help='Size of HR test patches. Use 0 for full image.')
parser.add_argument('--scaling', type=int, default=10, help='Scaling factor')
parser.add_argument('--in-memory', default=False, action='store_true', help='Hold data in memory during evaluation')
parser.add_argument('--feature-extractor', type=str, default='UNet', help='Feature extractor for edge potentials')

parser.add_argument('--Npre', type=int, default=8000, help='N learned iterations, but without gradients')
parser.add_argument('--Ntrain', type=int, default=1024, help='N learned iterations with gradients')
