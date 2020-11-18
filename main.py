import torch
import argparse
import visdom
import torch.optim as optim
from model import DenseNet_GCN
from loss import JSD_Loss
from dataset import Sphere_Dataset
from train import train
from test import test
from torch.utils.data import DataLoader


if __name__ == "__main__":

    # 1. parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='how many the model iterate?')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--data_path', type=str, default='D:\Data\\SUN360')
    parser.add_argument('--save_path', type=str, default="./saves")
    parser.add_argument('--save_file_name', type=str, default="densenet_101_kappa_25")

    opts = parser.parse_args()
    print(opts)

    # 2. device config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 3. visdom
    vis = visdom.Visdom()

    # 4. data loader
    train_set = Sphere_Dataset(root=opts.data_path, split='TRAIN')
    train_loader = DataLoader(dataset=train_set,
                              batch_size=opts.batch_size,
                              shuffle=True)

    val_set = Sphere_Dataset(root=opts.data_path, split='TEST')
    val_loader = DataLoader(dataset=val_set,
                            batch_size=opts.batch_size,
                            shuffle=True)

    # 5. model
    model = DenseNet_GCN().to(device)
    # resume
    # model.load_state_dict(torch.load(os.path.join(opts.save_path, opts.save_file_name + '.{}.pth'.format(27))))

    # 6. loss
    criterion = JSD_Loss()

    # 7. optimizer
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=0.0005)

    for epoch in range(opts.epoch):

        # 8 - 1. train
        train(epoch=epoch,
              device=device,
              data_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              vis=vis,
              save_path=opts.save_path,
              save_file_name=opts.save_file_name)

        test(epoch=epoch,
             device=device,
             data_loader=val_loader,
             model=model,
             criterion=criterion,
             vis=vis,
             save_path=opts.save_path,
             save_file_name=opts.save_file_name)

    print("Optimization Finished!")






