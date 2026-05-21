import yaml
import argparse
from types import SimpleNamespace

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def config_to_namespace(config):
    """Convert dictionary config to SimpleNamespace for easier access"""
    return SimpleNamespace(**config)

def update_args_with_config(args, config):
    """Update argparse args with config values"""
    # Dataset config
    if 'dataset' in config:
        args.dataset = config['dataset'].get('name', args.dataset)
        args.stereo = config['dataset'].get('stereo', args.stereo)
        args.preprocessed = config['dataset'].get('preprocessed', args.preprocessed)
        # Add augmentation parameter
        if 'augmentation' in config['dataset']:
            args.augmentation = config['dataset']['augmentation']

    # Model config
    if 'model' in config:
        args.backbone = config['model'].get('backbone', args.backbone)
        args.regression = config['model'].get('regression', args.regression)
        args.normalize = config['model'].get('normalize', args.normalize)
        args.pred_head_dim = config['model'].get('pred_head_dim', args.pred_head_dim)
        args.cla_res = config['model'].get('cla_res', args.cla_res)
        args.dino = config['model'].get('dino_size', args.dino)

    # Training config
    if 'training' in config:
        args.batch_size = config['training'].get('batch_size', args.batch_size)
        print(f"Batch size from config: {args.batch_size}")
        args.epochs = config['training'].get('epochs', args.epochs)
        args.seed = config['training'].get('seed', args.seed)

    # Loss config
    if 'loss' in config:
        args.loss = config['loss'].get('type', args.loss)

    # Scheduler config
    if 'scheduler' in config:
        args.scheduler = config['scheduler'].get('type', args.scheduler)

    # Logging config
    if 'logging' in config:
        args.logdir = config['logging'].get('logdir', args.logdir)
        args.summary_freq = config['logging'].get('summary_freq', args.summary_freq)

    # WandB config
    if 'wandb' in config:
        args.name_run = config['wandb'].get('name_run', args.name_run)
        args.notes = config['wandb'].get('notes', args.notes)

    # Checkpoint config
    if 'checkpoint' in config:
        args.loadckpt = config['checkpoint'].get('load_path', args.loadckpt)

    return args

def create_parser():
    """Create argument parser with config file option"""
    parser = argparse.ArgumentParser(description='RoadBEV: Road Surface Reconstruction in Bird\'s Eye View')

    # Add config file argument
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration YAML file')

    # Add all the original arguments
    parser.add_argument('--dataset', help='dataset to use: add it to wandb runs')
    parser.add_argument('--stereo', action='store_true', help='if yes, use RoadBEV-stereo; otherwise, RoadBEV-mono')
    parser.add_argument('--cla_res', type=float, default=0.5, help='class resolution for elevation classification')
    parser.add_argument('--batch_size', type=int, default=8, help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='maximum learning rate')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
    parser.add_argument('--logdir', default='/data/rhf/checkpoints/', help='the directory to save logs and checkpoints')
    parser.add_argument('--loadckpt', default=None, help='load the weights from a specific checkpoint')
    parser.add_argument('--summary_freq', type=int, default=10, help='summary_freq')
    parser.add_argument('--seed', type=int, default=307, metavar='S', help='random seed')
    parser.add_argument('--regression', action='store_true', help='regression or classification')
    parser.add_argument('--backbone', default='efficientnet', help='Use DepthAnything3 backbone or EfficientNet')
    parser.add_argument('--gradient_weight', type=float, default=0.01, help='weight for gradient loss in regression')
    parser.add_argument('--notes', type=str, default='', help='notes for wandb run')
    parser.add_argument('--scheduler', type=str, default='onecycle', help='type of lr scheduler to use: onecycle or reduceonplateau')
    parser.add_argument('--loss', type=str, default='L1', help='type of loss to use if regression: L1, gaussian NLL')
    parser.add_argument('--normalize', action='store_true', help='disable normalization')
    parser.add_argument('--name_run', type=str, default=' ', help='give the name of the wandb run')
    parser.add_argument('--pred_head_dim', type=int, default=128, help='define the bottleneck between the transformer encoder and the CNN prediction head')
    parser.add_argument('--preprocessed', action='store_true', help='if yes, the dataloader will load preprocessed data')
    parser.add_argument('--augmentation', action='store_true', help='enable data augmentation for training')
    parser.add_argument('--load_pt', default=None, help='load weights, optimizer, start_idx to resume run')
    parser.add_argument('--dino', default="small", help='ViT encoder size')

    return parser

def parse_args_with_config():
    """Parse arguments and load config file"""
    parser = create_parser()
    args = parser.parse_args()

    # Load config file
    if args.config:
        config = load_config(args.config)
        args = update_args_with_config(args, config)
        # Store config for reference
        args.config_dict = config

    return args