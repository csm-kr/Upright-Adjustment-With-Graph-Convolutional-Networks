import time
import numpy as np
import torch
import os
from evaluate import angle_acc


def test(epoch, device, data_loader, model, criterion, vis, save_path, save_file_name):
    """

    :param epoch : int
    :param data_loader: data.DataLoader
    :param model: nn.Module
    :param loss: nn.Module
    :param optimizer: optim
    :param visdom: visdom
    :return:
    """
    # ------------------- load -------------------
    tic = time.time()
    print('Test : {}'.format(epoch))

    model.load_state_dict(torch.load(os.path.join(save_path, save_file_name + '.{}.pth'.format(epoch))))
    model.eval()
    with torch.no_grad():

        test_loss = 0
        test_angle_max = 0
        test_angle_exp = 0

        for idx, (images, _, _, xyz, pdf, adj, rotated_points) in enumerate(data_loader):

            # ------------------- cuda -------------------
            images = images.to(device)
            xyz = xyz.to(device)
            adj = adj.to(device)

            # ------------------- loss -------------------
            output = model(images, adj)  # B, 2
            output = torch.softmax(output.squeeze(-1), dim=1)
            loss = criterion(output, pdf)

            # ------------------- eval -------------------
            # gt
            gt_xyz = xyz.cpu().detach().numpy()
            gt_xyz = np.squeeze(gt_xyz)

            # ---------------------------------- pred
            # pred_exp
            output_numpy = output.cpu().detach().numpy()
            points = data_loader.dataset.points
            points_x = points[:, 0]
            points_y = points[:, 1]
            points_z = points[:, 2]
            exp_x = np.dot(output_numpy, points_x)
            exp_y = np.dot(output_numpy, points_y)
            exp_z = np.dot(output_numpy, points_z)
            norm = np.sqrt(exp_x ** 2 + exp_y ** 2 + exp_z ** 2)
            exp_x /= norm
            exp_y /= norm
            exp_z /= norm
            pred_xyz_exp = np.stack([exp_x, exp_y, exp_z], axis=-1)

            # pred_max
            output_numpy = output.cpu().detach().numpy()
            output_index = np.argmax(output_numpy, axis=1)
            pred_xyz_max = data_loader.dataset.points[output_index]

            # pred_xyz = spherical_to_cartesian(output_numpy[:, 0], output_numpy[:, 1])
            angle_exp = angle_acc(gt_xyz, pred_xyz_exp)
            angle_max = angle_acc(gt_xyz, pred_xyz_max)

            # ------------------- print -------------------
            test_loss += loss.item()
            test_angle_exp += angle_exp
            test_angle_max += angle_max

            # print
            if idx % 1 == 0:
                print('Step: [{0}/{1}]\t'
                      'Loss: {test_loss:.4f}\t'
                      'Angle error_max: {test_angle_error_max:.4f}\t'
                      'Angle error_exp: {test_angle_error_exp:.4f}\t'
                      .format(idx, len(data_loader),
                              test_loss=loss.item(),
                              test_angle_error_max=angle_max,
                              test_angle_error_exp=angle_exp))

        test_loss /= len(data_loader)
        test_angle_max /= len(data_loader)
        test_angle_exp /= len(data_loader)

        # plot
        vis.line(X=torch.ones((1, 3)).cpu() * epoch,  # step
                 Y=torch.Tensor([test_loss, test_angle_max, test_angle_exp]).unsqueeze(0).cpu(),
                 win='test',
                 update='append',
                 opts=dict(xlabel='Epochs',
                           ylabel='Angle / Loss ',
                           title='Test results',
                           legend=['Loss', 'Angle_max', 'Angle_exp']))

    print('test_time : {:.4f}s'.format(time.time() - tic))