import torch
import numpy as np
from numba import jit
import scipy.sparse as sp


def cartesian_to_spherical_torch(x, y, z):
    """
    convert cartesian coordinates to spherical coordinates
    :param x:
    :param y:
    :param z:
    :return: phi theta
    """
    phi = torch.atan2(y, x) * 180 / np.pi
    theta = torch.acos(z) * 180 / np.pi
    return phi, theta


def cartesian_to_spherical(x, y, z):
    """
    convert cartesian coordinates to spherical coordinates
    :param x:
    :param y:
    :param z:
    :return: phi theta
    """
    phi2 = np.arctan2(y, x) * 180 / np.pi
    theta = np.arccos(z) * 180 / np.pi

    spherical = np.stack([phi2, theta], axis=-1)
    return spherical


def spherical_to_cartesian(phi, theta):
    """
    convert sphere coordinates to cartesian coordinates
    :param phi: np.ndarray - [B, ] : azimuth angle
    :param theta: np.ndarray - [B, ] : inclination angle
    :return: np.ndarray - [B, 3] : (x, y, z)
    """
    phi *= np.pi / 180
    theta *= np.pi / 180

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    cartesian = np.stack([x, y, z], axis=-1)
    return cartesian


@jit(nopython=True, cache=True)
def calculate_Rmatrix_from_phi_theta(phi, theta):
    """
    A = [0,0,1] B = [x,y,z] ( = phi,theta) the goal is to find rotation matrix R where R*A == B
    please refer to this website https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    v = a cross b ,s = ||v|| (sine of angle), c = a dot b (cosine of angle)
    :param phi: z axis angle
    :param theta: xy axis angle
    :return: rotation matrix that moves [0,0,1] to ([x,y,z] that is equivalent to (phi,theta))
    """

    epsilon = 1e-7
    A = np.array([0, 0, 1], dtype=np.float64)  # original up-vector
    # B = spherical_to_cartesian(phi,theta)  # target vector

    phi = np.deg2rad(phi)
    theta = np.deg2rad(theta)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    B = np.array([x, y, z], dtype=np.float64)

    desiredResult = B
    # dot(R,A) == B
    # If A == B then return identity(3)
    if A[0] - B[0] < epsilon \
            and A[0] - B[0] > -epsilon \
            and A[1] - B[1] < epsilon \
            and A[1] - B[1] > -epsilon \
            and A[2] - B[2] < epsilon \
            and A[2] - B[2] > -epsilon:
        # print('Identity matrix is returned')
        return np.identity(3)

    # v = np.cross(A, B)
    # In the numba, numpy.cross is not supported
    cross_1 = np.multiply(A[1],B[2])-np.multiply(A[2],B[1])
    cross_2 = np.multiply(A[2],B[0])-np.multiply(A[0],B[2])
    cross_3 = np.multiply(A[0],B[1])-np.multiply(A[1],B[0])
    v = np.array([cross_1,cross_2,cross_3])

    c = np.dot(A, B)
    skewSymmetric = skewSymmetricCrossProduct(v)

    if -epsilon < c + 1 and c + 1 < epsilon:
        R = -np.identity(3)
    else:
        R = np.identity(3) + skewSymmetric + np.dot(skewSymmetric, skewSymmetric) * (
                    1 / (1 + c))  # what if 1+c is 0?
    return R


@jit(nopython=True, cache=True)
def skewSymmetricCrossProduct(v):
    """

    :param v: a vector in R^3
    :return: [ 0 -v3 v2 ; v3 0 -v1; -v2 v1 0]
    """
    v1 = v[0]
    v2 = v[1]
    v3 = v[2]

    skewSymmetricMatrix = np.array([[0, -v3, v2], [v3, 0, -v1], [-v2, v1, 0]], dtype=np.float64)

    return skewSymmetricMatrix


def rotate_map_given_R(R, height, width):
    """
    calculating rotation map for corresponding image dimension and phi,theta value.
    (1,0,0)(rho,phi,theta) on sphere goes to (1,phi,theta)
    :param R: phi,theta in degrees ,
    :param height: height of an image
    :param width: width of an image
    :return: rotation map for x and y coordinate
    """

    def pos_conversion(x, y, z):
        # given postech protocol
        # return my protocol

        return z, -x, y

    def inv_conversion(x, y, z):
        # given my conversion
        # convert it to postech system.

        return -y, z, x

    # if not original_file.is_file():
    # step1
    spherePoints = flat_to_sphere(height, width)
    # R = calculate_Rmatrix_from_phi_theta(phi,theta)
    R_inv = np.linalg.inv(R)
    #step2
    spherePointsRotated = rotate_sphere_given_phi_theta(R_inv, spherePoints)

    #Create two mapping variable
    #step3
    [map_x, map_y] = sphere_to_flat(spherePointsRotated,height,width)

    # dst(y,x) = src(map_x(y,x),map_y(y,x))
    return [map_x, map_y]


@jit(nopython=True, cache=True)
def flat_to_sphere(height, width):
    """
    Create matrix that contains x,y,z coordinates
    :param height: height  of image
    :param width: width of image
    :return: return (height,width,3) numpy ndarray. (y,x) of array has (x,y,z) value which is on sphere. (return sphere points)
    """

    sphere = np.zeros((height, width, 3))
    x_to_theta = np.zeros(width)
    y_to_phi = np.zeros(height)

    theta_slope = 2*np.pi/(width-1)
    phi_slope = np.pi/(height-1)

    # linear map from [y,x] to [phi,theta]
    for x in range(0, width):
        x_to_theta[x] = np.rad2deg(np.multiply(x, theta_slope))

    for y in range(0, height):
        y_to_phi[y] = np.rad2deg(np.multiply(y, phi_slope))

    # For every pixel coordinates, create a matrix that contains the
    # corresponding (x,y,z) coordinates
    for y_f in range(0, height):
        for x_f in range(0, width):
            theta = x_to_theta[x_f]
            phi = y_to_phi[y_f]

            phi = np.deg2rad(phi)
            theta = np.deg2rad(theta)
            x_s = np.sin(phi) * np.cos(theta)
            y_s = np.sin(phi) * np.sin(theta)
            z_s = np.cos(phi)
            sphere[y_f, x_f, 0] = x_s
            sphere[y_f, x_f, 1] = y_s
            sphere[y_f, x_f, 2] = z_s

    return sphere


@jit(nopython=True, cache=True)
def sphere_to_flat(spherePointsRotated, height, width):
    """
    calculate destination x,y coordinate given information x,y(2d flat) <-> x,y,z(sphere)
    :param spherePointsRotated: y,x coordinate on 2d flat image,numpy nd array of dimension (height,width,3).
    :param height:
    :param width:
    :return:
    """

    map_y = np.zeros((height, width), dtype=np.float32)
    map_x = np.zeros((height, width), dtype=np.float32)

    factor_phi = (height-1)/np.pi
    factor_theta = (width-1)/(2*np.pi)

    # Get multiplied(by inverted rotation matrix) x,y,z coordinates
    for image_y in range(0, height):
        for image_x in range(0, width):
            pointOnRotatedSphere_x = spherePointsRotated[image_y, image_x, 0]
            pointOnRotatedSphere_y = spherePointsRotated[image_y, image_x, 1]
            pointOnRotatedSphere_z = spherePointsRotated[image_y, image_x, 2]

            x_2 = np.power(pointOnRotatedSphere_x, 2)
            y_2 = np.power(pointOnRotatedSphere_y, 2)
            z_2 = np.power(pointOnRotatedSphere_z, 2)

            theta = float(np.arctan2(pointOnRotatedSphere_y, pointOnRotatedSphere_x))
            # atan2 returns value of which range is [-pi,pi], range of theta is [0,2pi] so if theta is negative value,actual value is theta+2pi
            if theta < 0:
                theta = theta + np.multiply(2,np.pi)

            rho = x_2 + y_2 + z_2
            rho = np.sqrt(rho)
            phi = np.arccos(pointOnRotatedSphere_z / rho)

            map_y[image_y, image_x] = phi*factor_phi
            map_x[image_y, image_x] = theta*factor_theta

    return [map_x, map_y]


@jit(nopython=True, cache=True)
def rotate_sphere_given_phi_theta(R, spherePoints):
    """
    apply R to every point on sphere
    :param R:  phi,theta in degrees
    :param spherePoints: spherePoints(x,y,z of on sphere dimension (height,width,3) )
    :return: spherePointsRotated of which dimension is (h,w,3) and contains (x',y',z' )
             (x',y',z')=R*(x,y,z) where R maps (0,0,1) to (vx,vy,vz) defined by theta,phi (i.e. R*(0,0,1)=(vx,vy,vz))
    """


    h, w, c = spherePoints.shape
    spherePointsRotated = np.zeros((h, w, c),dtype=np.float64)

    for y in range(0, h):
        for x in range(0, w):
            pointOnSphere = spherePoints[y, x, :]
            pointOnSphereRotated = np.dot(R, pointOnSphere)
            spherePointsRotated[y, x, :] = pointOnSphereRotated
            # spherePointsRotated[y, x, :] = np.dot(R, pointOnSphere)

    return spherePointsRotated


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
