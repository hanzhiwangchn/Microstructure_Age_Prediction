import math, torch
import torch.nn as nn
import monai
import logging

logger = logging.getLogger(__name__)


def build_model(args):
    """main function for model building"""
    if args.model == 'densenet':
        model = build_densenet121_monai()
    elif args.model == 'resnet':
        model = ResNet(input_size=args.input_shape)

    if torch.cuda.device_count() > 1:
        model= torch.nn.DataParallel(model, list(range(torch.cuda.device_count())))
    model.to(args.device)

    # parameter initialization
    def weights_init(m):
        if isinstance(m, nn.Conv3d):
            if args.params_init == 'kaiming_uniform':
                torch.nn.init.kaiming_uniform_(m.weight.data)
            elif args.params_init == 'kaiming_normal':
                torch.nn.init.kaiming_normal_(m.weight.data)
    if args.params_init != 'default':
        model.apply(weights_init)

    # model_config for output files
    model_config = f'{args.model}_loss_{args.loss_type}_skewed_{args.skewed_loss}_' \
                   f'correlation_{args.correlation_type}_dataset_{args.dataset}_' \
                   f'{args.comment}_rnd_state_{args.random_state}'
    return model, model_config


# ------------------- Scratch DenseNet from MONAI ---------------------
def build_densenet121_monai():
    model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=1, out_channels=1)
    return model


# ------------------- Scratch ResNet ---------------------
class ResNetBlock(nn.Module):
    def __init__(self, block_number, input_size):
        super(ResNetBlock, self).__init__()

        layer_in = input_size if input_size is not None else 2 ** (block_number + 1)
        layer_out = 2 ** (block_number + 2)

        self.conv1 = nn.Conv3d(layer_in, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(layer_out)
        self.act1 = nn.ELU()
        self.conv2 = nn.Conv3d(layer_out, layer_out, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(layer_out)
        self.shortcut = nn.Conv3d(layer_in, layer_out, kernel_size=1, stride=1, bias=False)
        self.act2 = nn.ELU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.act2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, input_size):
        super(ResNet, self).__init__()

        self.layer1 = self._make_block(1, input_size[0])
        self.layer2 = self._make_block(2)
        self.layer3 = self._make_block(3)
        self.layer4 = self._make_block(4)
        self.layer5 = self._make_block(5)

        d, h, w = ResNet._maxpool_output_size(input_size[1::], nb_layers=5)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(p=0.5),
            nn.Linear(128 * d * h * w, 64),
            nn.ELU(),
            nn.Linear(64, 1)
        )

    @staticmethod
    def _make_block(block_number, input_size=None):
        return nn.Sequential(
            ResNetBlock(block_number, input_size),
            nn.MaxPool3d(3, stride=2))

    @staticmethod
    def _maxpool_output_size(input_size, kernel_size=(3, 3, 3), stride=(2, 2, 2), nb_layers=1):
        d = math.floor((input_size[0] - kernel_size[0]) / stride[0] + 1)
        h = math.floor((input_size[1] - kernel_size[1]) / stride[1] + 1)
        w = math.floor((input_size[2] - kernel_size[2]) / stride[2] + 1)

        if nb_layers == 1:
            return d, h, w
        return ResNet._maxpool_output_size(input_size=(d, h, w), kernel_size=kernel_size,
                                           stride=stride, nb_layers=nb_layers - 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.fc(out)
        return out
