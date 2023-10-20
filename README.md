# Note
# 2023.10.19 -- 基础知识总结

## 一、文件读取与保存

## 1.加载数据流程

- 文件读取：mat文件或h5py文件。

```python
class MatLoader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatLoader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):

        try:
            self.data = sio.loadmat(self.file_path)   #mat文件
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)    #h5py文件
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

   
```

- 文件保存路径，统一采用os.path.join。

```python
  file_path = os.path.join('data', 'dim_pro8_single_all.mat')
  reader = MatLoader(file_path)
  fields = reader.read_field('field')
```

- torch.utils.data主要包括以下三个类：

​          a. class torch.utils.data.Dataset   其他的数据集类必须是torch.utils.data.Dataset的子类,比如说torchvision.ImageFolder.

​          b. class torch.utils.data.sampler.Sampler(data_source) 

​         参数: data_source (Dataset) – dataset to sample from 

​         作用: 创建一个采样器，class torch.utils.data.sampler.Sampler是所有的Sampler的基类，其中，iter(self)函数来获取一个迭代器,对数据集中元素的索引进行迭代，len(self)方法返回迭代器中包含元素的长度.

​          c. class torch.utils.data.DataLoader

- 读取数据的顺序是：

​         a. 加载数据，提取出feature和label，并转换成tensor

​         b. 创建一个dataset对象

​         c. 创建一个dataloader对象，dataloader类的作用就是实现数据以什么方式输入到什么网络中

​         d. 循环dataloader对象，将data，label拿到模型中去训练

```python
 train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_x, train_y),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
 for batch, (xx, yy) in enumerate(dataloader):
        ......
```

