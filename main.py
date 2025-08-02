import os

import torch
import yaml
from torchvision import datasets
from data.multi_view_data_injector import MultiViewDataInjector
from data.transforms import get_simclr_data_transforms
from models.mlp_head import MLPHead
from models.resnet_base_network import ResNet18, ResNet_withAttention
from models.attention import NonLinear_Attention, Dot_Attention
from trainer import BYOLTrainer, BYOLTrainer_withAttention

print(torch.__version__)
#torch.manual_seed(0)


def main():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training with: {device}")

    data_transform = get_simclr_data_transforms(**config['data_transforms'])#224
    data_dir = config['data']['data_dir']

    if config['data']['dataset_name'] == 'stl10':
        train_dataset = datasets.STL10(data_dir, split='train+unlabeled', download=True,
                                   transform=MultiViewDataInjector([data_transform, data_transform]))
        
    elif config['data']['dataset_name'] == 'imagenet-100':
        train_dataset = datasets.ImageFolder(data_dir,
                                             transform=MultiViewDataInjector([data_transform, data_transform]))

    # online network
    if config['method'] == 'byol':
        online_network = ResNet18(**config['network']).to(device)
    elif config['method'] == 'byol_atten':
        online_network = ResNet_withAttention(**config['network']).to(device)
        dim_atten = config['dim_atten']
        projetion = MLPHead(in_channels=online_network.in_features, **config['network']['projection_head']).to(device)
        if config['atten_type'] == 'Non_Linear':
            attention = NonLinear_Attention(online_network.in_features, dim_atten).to(device)
        elif config['atten_type'] == 'Dot':
            attention = Dot_Attention(online_network.in_features, dim_atten).to(device)

    pretrained_folder = config['network']['fine_tune_from']

    # load pre-trained model if defined
    if pretrained_folder:
        try:
            checkpoints_folder = os.path.join('./runs', pretrained_folder, 'checkpoints')

            # load pre-trained parameters
            load_params = torch.load(os.path.join(os.path.join(checkpoints_folder, 'model.pth')),
                                     map_location=torch.device(torch.device(device)))

            online_network.load_state_dict(load_params['online_network_state_dict'])

        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

    # predictor network
    if config['method'] == 'byol':
        predictor = MLPHead(in_channels=online_network.projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)
    elif config['method'] == 'byol_atten':
        predictor = MLPHead(in_channels=projetion.net[-1].out_features,
                        **config['network']['projection_head']).to(device)

    # target encoder
    if config['method'] == 'byol':
        target_network = ResNet18(**config['network']).to(device)
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
                                **config['optimizer']['params'])
        trainer = BYOLTrainer(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])
        
    elif config['method'] == 'byol_atten':
        target_network = ResNet_withAttention(**config['network']).to(device)
        target_projetion = MLPHead(in_channels=online_network.in_features, **config['network']['projection_head']).to(device)
        optimizer = torch.optim.SGD(list(online_network.parameters()) + list(projetion.parameters()) + list(predictor.parameters())
                                    + list(attention.parameters()),
                                **config['optimizer']['params'])
        
        trainer = BYOLTrainer_withAttention(online_network=online_network,
                          target_network=target_network,
                          optimizer=optimizer,
                          projetion=projetion,
                          target_projetion=target_projetion,
                          attention=attention,
                          predictor=predictor,
                          device=device,
                          **config['trainer'])

    #optimizer = torch.optim.SGD(list(online_network.parameters()) + list(predictor.parameters()),
    #                            **config['optimizer']['params'])

    #trainer = BYOLTrainer(online_network=online_network,
    #                      target_network=target_network,
    #                      optimizer=optimizer,
    #                      predictor=predictor,
    #                      device=device,
    #                      **config['trainer'])

    trainer.train(train_dataset)


def torch_seed():
    config = yaml.load(open("./config/config.yaml", "r"), Loader=yaml.FullLoader)
    seed = config['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

if __name__ == '__main__':
    torch_seed()
    main()
