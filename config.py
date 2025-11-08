# config.py
import os

# --- Base Directories ---
DATA_ROOT = '/kaggle/working/LMU-Net/data'
CKPT_ROOT = '/kaggle/working/LMU-Net/ckpt'

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0

os.makedirs(DATA_ROOT, exist_ok=True)
os.makedirs(CKPT_ROOT, exist_ok=True)

GRAYSCALE_DATASETS = [
    'TSRS_RSNA-Epiphysis',
    'TSRS_RSNA-Articular-Surface',
    'JSRT',
    'COVID19_Radiography',
    'DentalPanoramic',
    'SixDiseasesChestXRay',
    'MontgomeryCounty'
]

COLOR_DATASETS = [
    'CVC-ClinicDB',
    'CVC-ColonDB',
    'ETIS-LaribPolypDB',
    'Kvasir-SEG'
]

DATASET_CONFIG = {
    'TSRS_RSNA-Epiphysis': {
        'path': os.path.join(DATA_ROOT, 'TSRS_RSNA-Epiphysis'),
        'structure': 'TSRS_RSNA', 'num_classes': 2, 'size': (428, 428)
    },
    'JSRT': {
        'path': os.path.join(DATA_ROOT, 'jsrt'),
        'structure': 'FLAT_SPLIT', 'num_classes': 2, 'size': (428, 428)
    },
    'MontgomeryCounty': {
        'path': os.path.join(DATA_ROOT, 'MontgomerySet'),
        'structure': 'MONTGOMERY', 'num_classes': 2, 'size': (428, 428)
    },
    'COVID19_Radiography': {
        'path': os.path.join(DATA_ROOT, 'covid19-radiography-database'),
        'structure': 'FLAT_SPLIT', 'num_classes': 4, 'size': (428, 428)
    },

    'CVC-ClinicDB': {
        'path': os.path.join(DATA_ROOT, 'CVC-ClinicDB'),
        'structure': 'FLAT_SPLIT', 'num_classes': 2, 'size': (384, 384)
    },
    'CVC-ColonDB': {
        'path': os.path.join(DATA_ROOT, 'CVC-ColonDB'),
        'structure': 'FLAT_SPLIT', 'num_classes': 2, 'size': (384, 384)
    },
    'ETIS-LaribPolypDB': {
        'path': os.path.join(DATA_ROOT, 'etis-laribpolypdb'),
        'structure': 'FLAT_SPLIT', 'num_classes': 2, 'size': (384, 384)
    },
    'Kvasir-SEG': {
        'path': os.path.join(DATA_ROOT, 'Kvasir-SEG'),
        'structure': 'FLAT_SPLIT', 'num_classes': 2, 'size': (384, 384)
    },
}

BACKBONE_INPUT_RESOLUTIONS = {
    'vgg16': (224, 224),
    'resnet50': (224, 224),
    'inception_v3': (299, 299),
    'efficientnet_b0': (224, 224),
    'efficientnet_b3': (300, 300),
    'vgg19': (224, 224),
}

BACKBONE_CHANNELS = {
    'vgg16': {'e1': 64, 'e2': 128, 'e3': 256, 'e4': 512, 'bottleneck': 512},
    'vgg19': {'e1': 64, 'e2': 128, 'e3': 256, 'e4': 512, 'bottleneck': 512},
    'resnet50': {'e1': 64, 'e2': 256, 'e3': 512, 'e4': 1024, 'bottleneck': 2048},
    'inception_v3': {'e1': 192, 'e2': 288, 'e3': 768, 'e4': 1280, 'bottleneck': 2048},
    'efficientnet_b0': {'e1': 24, 'e2': 40, 'e3': 80, 'e4': 112, 'bottleneck': 320},
    'efficientnet_b3': {'e1': 32, 'e2': 48, 'e3': 136, 'e4': 232, 'bottleneck': 384},
    'efficientnet_b4': {
    'e1': 24,
    'e2': 32,
    'e3': 56,
    'e4': 112,
    'bottleneck': 1792 },
    'mobilenet_v2': {'e1': 16, 'e2': 24, 'e3': 32, 'e4': 96, 'bottleneck': 1280},
}

DEFAULT_ARGS = {
    'num_workers': 2,
    'backbone': 'vgg16',
    'lasa_kernels': [1, 3, 5, 7],
    'epochs': 500,
    'batch_size': 10,
    'lr': 0.0005,
    'weight_decay': 0.0001,
    'patience': 15,
    'scale-h': 224,
    'scale-w': 224,
    'deep_supervision_weights': [0.2, 0.4, 0.6, 0.8, 1.0],
    'focal_loss_weight': 1.0,
    'dice_loss_weight': 1.0,
    'scheduler_type': 'CosineAnnealingWarmRestarts',
    'scheduler_patience': 5,
    'scheduler_factor': 0.5,
    'scheduler_min_lr': 1e-6,
    'scheduler_T0': 10,
    'scheduler_T_mult': 2,
    'test_only': False,
    'resume': False,
    'fine_tune_epochs': 0,
}
def get_backbone_resolution(backbone_name):
    return BACKBONE_INPUT_RESOLUTIONS.get(backbone_name, (224, 224))