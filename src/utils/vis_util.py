import visdom

import time

import numpy as np
import matplotlib.pyplot as plt
from typing import List


class Visualizer(object):
    """
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    """

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def re_init(self, env='default', **kwargs):
        """
        修改visdom的配置
        """
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        return self

    def plot_many(self, d):
        """
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        """
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        """
        self.plot('loss',1.00)
        """
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        """
        self.img('input_img',t.Tensor(64,64))
        """

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def log(self, info, win='log_text'):
        """
        self.log({'loss':1,'lr':0.0001})
        """

        self.log_text += ('[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info))
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        return getattr(self.vis, name)


class DatasetVisualizer(object):
    """
        Dataset Visualizer Utils
    """

    def __init__(self, data):
        super(DatasetVisualizer, self).__init__()
        self.data = data
        self.simple_len = len(data)
        self.plt = plt


def plot_loss(epochs: int, train_loss, valid_loss):
    """
        绘制loss图像
    """
    plt.title("Train Loss and Valid Loss")
    epochs = [i for i in range(1, epochs + 1)]
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, valid_loss, label="valid_loss")
    plt.legend()
    plt.show()


def plot_sen_len_freq(data: List[str], data_type: str):
    data_list = [len(x) for x in data]
    plt.hist(data_list, bins=40)
    plt.title(f"{data_type} Sentences length Frequency")
    plt.xlabel("sentences length")
    plt.ylabel("Frequency")
    plt.show()
