import torch
from torchvision import transforms
from Datasets.CelebA.loader import CELEBA

# Setup Augmentations

def global_transformer():
    return transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))])


def get_dataset(params, configs, device):
    if 'dataset' not in params:
        print('ERROR: No dataset is specified')

    if 'celeba' in params['dataset']:
        train_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='train', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        val_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='val', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        test_dst = CELEBA(root=configs['celeba']['path'], is_transform=True, split='test', img_size=(configs['celeba']['img_rows'], configs['celeba']['img_cols']), augmentations=None)
        train_loader = torch.utils.data.DataLoader(train_dst, batch_size=params['batch_size'], shuffle=True, num_workers=1)
        val_loader = torch.utils.data.DataLoader(val_dst, batch_size=params['batch_size'], num_workers=1)
        test_loader = torch.utils.data.DataLoader(test_dst, batch_size=params['batch_size'], num_workers=1)
        return train_loader, train_dst, val_loader, val_dst, test_loader, test_dst