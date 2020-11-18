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

1\. pytorch 

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

3\. download SUN 360 dataset from https://drive.google.com/file/d/1cOadYYkwXKzAf7YGYokEAEn1ofzx4QDX/view?usp=sharing 

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

4\. download model weight from https://drive.google.com/file/d/1cOadYYkwXKzAf7YGYokEAEn1ofzx4QDX/view?usp=sharing move into ./saves

3\. check the TYPE of image.  -- TYPE (e.g. 'jpg')

4\. Enter tne command : **python demo.py --demo_img_path PATH --demo_img_type TYPE**

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
