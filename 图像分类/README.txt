1..数据预处理：代码在utils文件中的utils.py，已经备注地方。包括调整图片大小，水平翻转，一定角度旋转，添加噪声，图片归一化，自己根据情况决定是否使用
   
   数据集放到 datasets 文件夹下面的 train 和 test 文件夹， 形式举例： train 文件夹下面有 dog 和 cat 文件夹；test 文件夹下面有 dog 和 cat 文件夹
2.整个工程用到的代码：train.py, test.py, model文件夹中的vgg1.py, src文件夹中的config.py, utils文件中的utils.py.其他的使用不到

3.图像分类确定好几分类， 要把分类数写到模型里面的 num_classes 参数后面， 例如model文件夹中 vgg1.py 第5行所示。

4代码运行：
训练： python train.py --path ./checkpoints/1/
(注：./checkpoints/1/文件夹下面的config.yml里面是超参数，可改动. 文件夹1相当于第一次实验，文件夹2相当于第二次实验，例如跑第二次实验，需要把config.yml复制到文件夹2下面， 然后  python train.py --path ./checkpoints/2/)

测试： 训练生成的模型在./checkpoints/1/model里面，找到最新的模型。把模型地址写到test.py的修改点处。
运行： python test.py --path ./checkpoints/1/

提醒：由于linux 和 Windows 系统在写路径时的语法有差异，有时候需要简单调整

代码运行环境：
conda create -n 123 python=3.7.3
conda install pytorch=1.1.0
conda install torchvision=0.3.0
conda install pillow=6.0.0
pip install tensorboardX
conda list 查看 是否含有 yaml 包，这个在windows 上不支持，如果有，就卸载 conda remove yaml
然后安装 pip install pyyaml

可视化环境：
conda create -n tf 
conda install tensorflow
pip install tensorboardX

可视化曲线：tensorboard --logdir=./runs

