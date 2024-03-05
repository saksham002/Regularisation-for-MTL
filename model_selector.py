#from models.multi_lenet import MultiLeNetO, MultiLeNetR
#from models.segnet import SegnetEncoder, SegnetInstanceDecoder, SegnetSegmentationDecoder, SegnetDepthDecoder
#from models.pspnet import SegmentationDecoder, get_segmentation_encoder
from Datasets.CelebA.model import ResNet, FaceAttributeDecoder, BasicBlock, TaskWeight
import torchvision.models as model_collection
import torch
import torch.nn as nn


def get_model(params, device):
    data = params['dataset']
    """if 'mnist' in data:
        model = {}
        model['rep'] = MultiLeNetR()
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep']
        if 'L' in params['tasks']:
            model['L'] = MultiLeNetO()
            if params['parallel']:
                model['L'] = nn.DataParallel(model['L'])
            model['L']
        if 'R' in params['tasks']:
            model['R'] = MultiLeNetO()
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['R']
        return model

    if 'cityscapes' in data:
        model = {}
        model['rep'] = get_segmentation_encoder() # SegnetEncoder()
        #vgg16 = model_collection.vgg16(pretrained=True)
        #model['rep'].init_vgg16_params(vgg16)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        model['rep']
        if 'S' in params['tasks']:
            model['S'] = SegmentationDecoder(num_class=19, task_type='C')
            if params['parallel']:
                model['S'] = nn.DataParallel(model['S'])
            model['S']
        if 'I' in params['tasks']:
            model['I'] = SegmentationDecoder(num_class=2, task_type='R')
            if params['parallel']:
                model['R'] = nn.DataParallel(model['R'])
            model['I']
        if 'D' in params['tasks']:
            model['D'] = SegmentationDecoder(num_class=1, task_type='R')
            if params['parallel']:
                model['D'] = nn.DataParallel(model['D'])
            model['D']
        return model"""

    if 'celeba' in data:
        model = {}
        model['rep'] = ResNet(BasicBlock, [2,2,2,2]).to(device)
        if params['parallel']:
            model['rep'] = nn.DataParallel(model['rep'])
        for t in params['tasks']:
            model[t] = FaceAttributeDecoder().to(device)
            model[t + "_weight"] = TaskWeight().to(device)
            if params['parallel']:
                model[t] = nn.DataParallel(model[t])
        return model
