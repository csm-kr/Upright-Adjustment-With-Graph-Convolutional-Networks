# Upright Adjustment with graph convolutional networks 

### Intoduction

:octocat: An Official Code for 'Upright Adjustment with graph convolutional networks'
[paper](https://ieeexplore.ieee.org/document/9190715) in ICIP2020
![concept](https://user-images.githubusercontent.com/18729104/99486691-09c3bf00-29a8-11eb-93f1-ebf9e8852d03.png)
the goal of this tasks is to find North pole and restore rotated images.

### Requirements

- Python 3.7
- Pytorch 1.5.0
- visdom
- numpy 
- cv2
- matplotlib
- numba

### Network
![networks](https://user-images.githubusercontent.com/18729104/99487045-d6cdfb00-29a8-11eb-89c5-ff0dd7b4638c.png)

### Results
qualitative results

![image](https://user-images.githubusercontent.com/18729104/99486993-b00fc480-29a8-11eb-87fc-230641239d23.png)

quantitative results

|Base network   | kappa    | Avg degree | within 10 degree| 
|:-------------:|:--------:|:----------:|:---------------:|
|DenseNet121    | 25       |4.0         |  97%            | 


### Quick start

1\. make environment pytorch==1.5.0 and torchvision==0.6.0

  ```bash
  conda create -n ~ python=3.7
  conda activate ~
  conda install pytorch==1.5.0, torchvision==0.6.0
  ```
2\. requirements 

  ```bash
  git clone https://github.com/csm-kr/Upright-Adjustment-with-Graph-Convolutional-Networks
  cd Upright-Adjustment-with-Graph-Convolutional-Networks
  python -m pip install -r requirements.txt
  ```

3\. download SUN 360 dataset from 
```
https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/ESbnmzr-J0ZBjjY8XVfSoIkBM3_9yKE_hYf-hT7vVs7suw?e=hKpbu3
```

and for ubuntu user, 
```
wget "https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/ESbnmzr-J0ZBjjY8XVfSoIkBM3_9yKE_hYf-hT7vVs7suw?e=hKpbu3&download=1"
```


and the components of the dataset are as follows.
```bash
SUN360  |-- train
             |-- 000000.jpg
             |-- 000001.jpg
             | ..(25000 images)
        |-- test
             |-- 030000.jpg
             |-- 030001.jpg
             | ..(4260 images)
        |-- val
             |-- 025000.jpg
             |-- 025001.jpg
             | ..(5000 images)
        |-- map_xy
             |-- 000_000_x.npy
             |-- 000_000_y.npy
             | ..(129600 numpy file)
```
Especially, The **'map_xy'** folder contains a mapping matrix of 180 degrees vertical viewing angle (FoV) and 360 degrees horizontal viewing angle. (about 47GB)

4\. download model weight from 
````
https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EctJ0cICVnBNrdcYT84aGLwBh7HkpgbaHsQ-GFM93-TmlA?e=MgLUBl
````
if ubuntu user, 

```
wget "https://livecauac-my.sharepoint.com/:u:/g/personal/csm8167_cau_ac_kr/EctJ0cICVnBNrdcYT84aGLwBh7HkpgbaHsQ-GFM93-TmlA?e=MgLUBl&download=1"
```


#### Training

```
# python main.py 
usage: main.py [-h] [--epoch] [--batch_size] [--lr] 
               [--data_path] [--save_path] [--save_file_name] 

  -h, --help            show this help message and exit
  --epoch               whole traning epochs 
  --batch_size          for training batch size, test batch is only 1.
  --lr                  initial learning rate (default 1e-5) 
  --data_path           your SUN360 data path. 
  --save_file_name      when you do experiment, you can change save_file_name to distinguish other pths.
  --save_path           the path to save .pth file
  --start_epoch         when you resume, set the start epochs. 
```


#### Testing

```
# python test.py 
usage: test.py [-h] [--test_epoch] [--save_path] 
               [--save_file_name] [--conf_thres]

  -h, --help            show this help message and exit
  --test_epoch          for testing, which epoch param do we get
  --save_path           for testing, params path (default './saves') 
  --save_file_name      save_file_name to distinguish other params. (default 'yolo_v2_vgg_16')
  --conf_thres          for testing, confience threshold which detector detect above the thres (default 0.01) 
```

```
# python main.py 
usage: main.py [-h] [--epoch] [--batch_size] [--lr] 
               [--data_path] [--save_path] [--save_file_name] 

  -h, --help            show this help message and exit
  --epoch               whole traning epochs 
  --batch_size          for training batch size, test batch is only 1.
  --lr                  initial learning rate (default 1e-5) 
  --data_path           your SUN360 data path. 
  --save_file_name      when you do experiment, you can change save_file_name to distinguish other pths.
  --save_path           the path to save .pth file
  --start_epoch         when you resume, set the start epochs. 
```


## Citation


```
@inproceedings{jung2020upright,
  title={Upright Adjustment With Graph Convolutional Networks},
  author={Jung, Raehyuk and Cho, Sungmin and Kwon, Junseok},
  booktitle={2020 IEEE International Conference on Image Processing (ICIP)},
  pages={1058--1062},
  year={2020},
  organization={IEEE}
}
```
