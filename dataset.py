import os
import cv2
import glob
import torch
import numpy as np
import torch.utils.data as data
# import utils
from vmf import VonMisesFisher
from utils import normalize, cartesian_to_spherical, spherical_to_cartesian
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

visualization = False


def show_spheres(scale, points, rgb, label):
    """

    :param scale: int
    :param points: tuple (x, y, z)
    :param rgb:
    :return:
    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # axis scale setting
    ax.set_xlim3d(-1 * scale, scale)
    ax.set_ylim3d(-1 * scale, scale)
    ax.set_zlim3d(-1 * scale, scale)
    x, y, z = label

    # label
    ax.grid(False)
    ax.plot([0, scale * x], [0, scale * y], [0, scale * z])

    # how rotate they are
    phi2 = np.arctan2(y, x) * 180 / np.pi
    theta = np.arccos(z) * 180 / np.pi

    if phi2 < 0:
        phi2 = 360 + phi2

    # ax.set_aspect('equal')

    rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))
    rgb = np.concatenate([rgb, rgb, rgb], axis=1)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2], alpha=1, facecolors=rgb, depthshade=False,
               edgecolors=None,
               )  # data coloring
    plt.legend(loc=2)

    # Photos viewed at 90 degrees
    ax.view_init(-1 * theta, phi2)

    # Photos from above
    ax.view_init(-1 * theta + 90, phi2)

    plt.draw()
    plt.show()


class Sphere_Dataset(data.Dataset):
    def __init__(self,
                 root='D:\Data\\360_data',
                 num_points=91,
                 kappa=25,
                 num_hops=2,
                 split='TRAIN'):
        """
        set sp dataset
        :param root: root directory
        :param transform: dataset torchvision transform
        """
        # super(Sphere_Dataset, self).__init__()
        super().__init__()
        self.root = root
        self.split = split
        assert self.split in ['TRAIN', 'TEST', 'VAL']
        self.split_root = os.path.join(self.root, self.split.lower())
        self.img_list = glob.glob(os.path.join(self.split_root, '*.jpg'))
        self.top_k = 6
        self.num_hops = num_hops
        self.kappa = kappa
        self.points = np.load('./uniform_points/91_uniform_points.npy')
        self.num_points = num_points

    def create_adj(self):

        top_k = self.top_k
        adj = np.load('./hop_adj/{}_{}_hop_adj_top_{}.npy'.format(self.num_points, self.num_hops, top_k))
        adj = normalize(adj)
        return adj

    def __getitem__(self, idx):

        # read image
        img = cv2.imread(self.img_list[idx])
        height, width, _ = img.shape

        # make rotate
        phi = np.random.randint(0, 180)
        theta = np.random.randint(0, 360)
        map_matrix_dir = os.path.join(self.root, 'map_xy')
        map_x_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_x.npy'
        map_y_path = map_matrix_dir + '/' + str('%03d' % phi) + '_' + str('%03d' % theta) + '_y.npy'

        if os.path.isfile(map_x_path) and os.path.isfile(map_y_path):  # if map_x and map_y both exist
            map_x = np.load(map_x_path)
            map_y = np.load(map_y_path)

        rotated_img = cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_TRANSPARENT)
        rotated_img_vis = rotated_img  # for visualization
        rotated_img = cv2.resize(rotated_img, (442, 221))

        # mean
        imagenet_mean_R = 103.939
        imagenet_mean_G = 116.779
        imagenet_mean_B = 123.68

        rotated_img[:, :, 0] = rotated_img[:, :, 0] - imagenet_mean_B
        rotated_img[:, :, 1] = rotated_img[:, :, 1] - imagenet_mean_G
        rotated_img[:, :, 2] = rotated_img[:, :, 2] - imagenet_mean_R

        # get ground truth angle
        gt_phi, gt_theta = phi, theta

        # get xyz
        gt_xyz = spherical_to_cartesian(gt_phi, gt_theta)

        # von mises fisher distribution
        mu = gt_xyz
        kappa = self.kappa
        vmf = VonMisesFisher(mu=mu, kappa=kappa)
        pdfs = vmf.pdfs(self.points).transpose()
        sum_of_pdfs = np.sum(pdfs)
        gt_pdfs = pdfs / sum_of_pdfs  # (91, 1)

        # visualization
        if visualization:

            cv2.imshow('origin_img', cv2.resize(img, (442, 221)))
            cv2.imshow('rotate_img', cv2.resize(rotated_img_vis, (442, 221)))
            print('phi : {}, theta : {}'.format(gt_phi, gt_theta))
            # show_spheres(scale=2, points=self.points, rgb=pdfs, label=gt_xyz)
            cv2.waitKey(0)

        # prepare to convert numpy to tensor
        rotated_img = rotated_img.astype(np.float32)
        rotated_img = np.transpose(rotated_img, (2, 0, 1))
        rotated_img = torch.FloatTensor(rotated_img)
        rotated_img = rotated_img.float().div(255)

        # convert angle(scalar)/xy to tensor
        gt_phi = torch.FloatTensor([gt_phi])
        gt_theta = torch.FloatTensor([gt_theta])
        gt_xyz = torch.FloatTensor([gt_xyz])
        adj = self.create_adj()

        rotated_points = self.points.astype(np.float32)
        return rotated_img, gt_phi, gt_theta, gt_xyz, gt_pdfs, adj, rotated_points

    def __len__(self):
        return len(self.img_list)


if __name__ == "__main__":

    sp_dataset = Sphere_Dataset(split='VAL')
    sp_loader = data.DataLoader(dataset=sp_dataset,
                                batch_size=2)

    for (rotated_img, gt_phi, gt_theta, gt_xyz, gt_pdfs, _, _) in sp_loader:
        print(rotated_img.shape)
