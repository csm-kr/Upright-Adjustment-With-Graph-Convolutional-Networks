# Upright Adjustment with graph convolutional networks 

### Intoduction

:octocat: An Official Code for 'Upright Adjustment with graph convolutional networks' (pytorch implementation)
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

