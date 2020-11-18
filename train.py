import torch
import numpy as np
import time
from utils import spherical_to_cartesian
from evaluate import angle_acc
import os


def train(epoch, device, data_loader, model, criterion, optimizer, vis, save_path, save_file_name):
    """

    :param epoch : int
    :param data_loader: data.DataLoader
    :param model: nn.Module
    :param loss: nn.Module
    :param optimizer: optim
    :param visdom: visdom
    :return:
    """
    model.train()
    tic = time.time()
    print('Epoch : {}'.format(epoch))

    epoch_loss = []
    epoch_angle_max = []
    epoch_angle_exp = []

    for idx, (images, phi, theta, xyz, pdf, adj, rotated_points) in enumerate(data_loader):

        # -------------------- cuda ------------------------
        images = images.to(device)
        phi = phi.to(device)
        theta = theta.to(device)
        xyz = xyz.to(device)
        # pdf = pdf.to(device)
        adj = adj.to(device)
        rotated_points = rotated_points.to(device)

        # -------------------- loss -------------------------
        output = model(images, adj)  # B, 91, 1
        output = torch.softmax(output.squeeze(-1), dim=1)
        loss = criterion(output, pdf)

        # -------------------- update -------------------------
        # stochastic gradient descent
        optimizer.zero_grad()
        loss.backward()  # delta theta
        optimizer.step()  # model param's update
        toc = time.time() - tic

        # -------------------- eval -------------------------
        # gt
        gt_xyz = xyz.cpu().detach().numpy()
        gt_xyz = np.squeeze(gt_xyz)  # [B, 3]

        # pred_exp
        output_numpy = output.cpu().detach().numpy()
        points = data_loader.dataset.points
        points_x = points[:, 0]
        points_y = points[:, 1]
        points_z = points[:, 2]
        exp_x = np.dot(output_numpy, points_x)
        exp_y = np.dot(output_numpy, points_y)
        exp_z = np.dot(output_numpy, points_z)
        norm = np.sqrt(exp_x**2+exp_y**2+exp_z**2)
        exp_x /= norm
        exp_y /= norm
        exp_z /= norm
        pred_xyz_exp = np.stack([exp_x, exp_y, exp_z], axis=-1)

        # pred_max
        output_numpy = output.cpu().detach().numpy()
        output_index = np.argmax(output_numpy, axis=1)
        pred_xyz_max = data_loader.dataset.points[output_index]

        # phi theat to xyz
        # pred_xyz = spherical_to_cartesian(output_numpy[:, 0], output_numpy[:, 1])

        # get angles
        angle_exp = angle_acc(gt_xyz, pred_xyz_exp)
        angle_max = angle_acc(gt_xyz, pred_xyz_max)

        epoch_loss.append(loss.item())
        epoch_angle_max.append(angle_max)
        epoch_angle_exp.append(angle_exp)

        # get lr
        for param_group in optimizer.param_groups:
            lr = param_group['lr']

        # -------------------- print -------------------------
        if idx % 1 == 0:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Angle error_max: {angle_error_max:.4f}\t'
                  'Angle error_exp: {angle_error_exp:.4f}\t'
                  'learning rate: {lr:.7f} s \t'
                  'Time: {time:.4f} s \t'
                  .format(epoch, idx, len(data_loader),
                          loss=loss.item(),
                          angle_error_max=angle_max,
                          angle_error_exp=angle_exp,
                          lr=lr,
                          time=toc))

        # loss plot
        vis.line(X=torch.ones((1, 1)).cpu() * idx + epoch * data_loader.__len__(),  # step
                 Y=torch.Tensor([loss.item()]).unsqueeze(0).cpu(),
                 win='train_loss',
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='Loss',
                           title='training loss',
                           legend=['Loss']))

        # accuracy plot
        vis.line(X=torch.ones((1, 2)).cpu() * idx + epoch * data_loader.__len__(),  # step
                 Y=torch.Tensor([angle_max, angle_exp]).unsqueeze(0).cpu(),
                 win='angle_accuracy',
                 update='append',
                 opts=dict(xlabel='step',
                           ylabel='angle',
                           title='angle_accuracy',
                           legend=['angle_accuracy_max', 'angle_accuracy_exp']))

    # -------------------- save -------------------------
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    torch.save(model.state_dict(), os.path.join(save_path, save_file_name + '.{}.pth'.format(epoch)))