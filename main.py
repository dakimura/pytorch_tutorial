# -*- coding:utf-8 -*-
"""
pytorchのtutorial "A 60 Minute Blitz" を自分で理解しながらすすめるためのコード
https://pytorch.org/tutorials/beginner/blitz

https://qiita.com/poorko/items/c151ff4a827f114fe954
に解説があってわかりやすかった
"""

import torchvision.transforms as transforms
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import torch  # 基本モジュール
from torch.autograd import Variable  # 自動微分用
import torch.nn as nn  # ネットワーク構築用
import torch.optim as optim  # 最適化関数
import torch.nn.functional as F  # ネットワーク用の様々な関数
import torch.utils.data as torchdata # データセット読み込み関連
import torchvision  # 画像関連
from torchvision import datasets, models, transforms  # 画像用データセット諸々


def imshow(img: torch.Tensor):
    """
    正規化された画像を元に戻してからmatplotlibで表示するユーティリティ関数

    :param img:
    :return:
    """
    img = img / 2 + 0.5  # 正規化してるのを元に戻す

    # Tensorをnumpy arrayに変換
    npimg = img.numpy()

    # imshowは2次元numpy配列から画像を作る関数
    # transposeは転置する関数で、第２引数が与えられた場合は(x,y,z)軸を引数に与えられた順番で交換する
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

    plt.show()


# notebookとは、コードと、実行結果と、そのメモをまとめたもの。
# https://qiita.com/hinastory/items/e179361ae806e8776c70
# Notebookサーバを通じて、編集や実行はブラウザ上で行えるので他人にコードを説明するときに便利。

# CNN = Convolutional Newral Network 畳み込みニューラルネットワーク
# 畳み込みとは、画像処理でよく用いられる手法で、カーネルまたはフィルターと呼ばれる格子状の数値データと、
# カーネルと同じサイズの部分画像（ウィンドウ）の数値データについてその積和を計算して１つの数値にする作業。
# カーネルを数ピクセルずつ動かし（この処理をストライドと呼ぶ）、それぞれのウィンドウで積和を計算することで画像を縮小する（＝畳み込む）
# 畳み込んだ結果得られるものがテンソルで、小さな画像のようなもの。特徴マップ(feature map)とも呼ばれる。
# https://www.atmarkit.co.jp/ait/articles/1804/23/news138.html
# ref. 畳み込みにおけるゼロパディングとは https://jp.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html

# 機械学習でよく出てくるMINISTデータとは、手書きで書かれた数字の画像とそのラベルデータのセット。
# Handwritten digit dataset.
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # in_channels=3 => シアン・マゼンタ・イエローの3チャンネルの画像が１つ与えらえれて、
        # out_channels=6 => 畳み込みの結果、6つのfeature mapを出力する。
        # たいていの場合、このアウトプットチャンネル数は出力に適用するカーネルの数と同じ。
        # kernel_size=5 => 5x5 square convolution kernel = 5*5の正方形の畳み込みカーネルが
        # 使用される。
        #
        # ちなみに、inputチャンネルとoutputチャンネルの数が異なるのはこういう仕組み↓
        # https://discuss.pytorch.org/t/convolution-input-and-output-channels/10205
        # 5*5の正方カーネルを3チャンネルの画像に適用するとき、そのカーネルは5*5*3 = 75のweight(と1つのbias)を持っている。
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5)


        # CNNでは畳み込み(Conv)層に続いて、畳み込み層の出力にプーリングと呼ばれる処理を行う。
        # プーリング処理については https://qiita.com/FukuharaYohei/items/73cce8f5707a353e3c3a がわかりやすかった。
        # Max pooling over a (2, 2) window = 2*2マスの中の最大値を選択して圧縮する処理がmax_pool2d.
        self.pool = nn.MaxPool2d((2, 2))
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        # an affine operation: y = Wx + b
        # アフィン変換。ある図形を引き伸ばしたり回転させたりする操作。
        # http://zellij.hatenablog.com/entry/20120523/p1
        # fc = full connection = 全結合層。すべてのノードがすべての次のノードにつながっている層のこと。
        # 一般的な形として、CNNは畳み込み層→プーリング層→全結合層からなる。
        # Linearは、入力値xに対して y = xA^T + b で表される線形変換を行う。
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120, bias=True)
        self.fc2 = nn.Linear(in_features=120, out_features=84, bias=True)
        self.fc3 = nn.Linear(in_features=84, out_features=10, bias=True)

        # conv1: 1枚の画像(3チャンネル, 32*32ピクセル)を 5*5の正方カーネル6つによって6 * 28 * 28 の特徴量マップに変換
        # pool: 2*2マスの最大値を取るsub sampling処理によって 6 * 14 * 14 の特徴量マップに変換
        # conv2: 6チャンネル*14*14の特徴量マップを、 5*5の正方カーネル6つによって16 * 10 * 10 の特徴量マップに変換
        # pool: 2*2マスの最大値を取るsub sampling処理によって 16 * 5 * 5 の特徴量マップに変換
        # fc1: 線形変換によって 16 * 5 * 5個の特徴量を 120個にする
        # fc2: 線形変換によって 120個の特徴量を84個にする
        # fc3: 線形変換によって 84個の特徴量を10個にする
        # 識別するラベルの数が10個なのでこれで完了.
        # (cat: 0.9, dog: 0.1, horse:0.0, ...) みたいな長さ10のものが出力されて、大体catだなということになる

    def forward(self, x):
        # ReLUとはRectified Linear Unitの略で、ニューラルネットワークでよく使われる活性化関数。別名ランプ関数、恒等関数とも言われる。
        #  http://arduinopid.web.fc2.com/N44.html
        # 活性化関数は、ニューロンが受け取った入力値に対して何かしらの重み付けに基づいて出力値を決める関数。
        # ReLUは受け取った値がプラスの場合はそのまま出力するシンプルな関数。
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # view関数については https://qiita.com/kenta1984/items/d68b72214ce92beebbe2 がわかりやすい。
        # 第一引数に-1を入れることで、列数が第二引数の数になるように敷き詰め直してくれる。
        # view関数の-1の意味は、「行(列)数は分かってるんだけど列(行)数は分からないんだよな…」というときにとりあえず-1を指定しておけば
        # 行もしくは列を与えたinputを見て設定してくれるもの。
        # ここでは入力されたTensorを1次元に平たくしている？
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # def num_flat_features(self, x: torch.Tensor) -> int:
    #     # 例:
    #     # >>> x = torch.randn(3,4,5)
    #     # >>> x.size()[1:]
    #     # torch.Size([4, 5])
    #
    #     # https://stats.stackexchange.com/questions/153531/what-is-batch-size-in-neural-network
    #     # batch sizeというのはネットワークを通じて伝搬されるサンプルの数のこと。
    #     # 例えば1050個の訓練データがあるとして、100個のbatchをセットアップしたいとします。
    #     # 訓練データのうち最初の100データでネットワークを使って訓練します。
    #     # 次の100個も訓練します。すべてのデータを伝搬させるまでこの処理をし続けます。
    #     # 最後のセットは50個しかなくて困ることになります。
    #     # なので、最初から「50」という1050を割り切れるサイズ（＝batch size)を選んでおくというシンプルな解決方法があります。
    #
    #     # https://to-kei.net/neural-network/sgd/
    #     # N 個の訓練データの中から一部、n個を取り出し、パラメータの更新をすることをミニバッチ学習と呼ぶ。
    #     # 取り出した訓練データをミニバッチと呼び、また取り出すデータ数nをミニバッチサイズと呼ぶ。
    #
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


    # backward関数はautogradを使って自動的に計算されるらしい

    # backpropagation(誤差逆伝搬法)は、出力との誤差を損失関数によって求め、その誤差を最小化するために
    # バイアスや重み付けを修正していく方法の１つ。
    # https://www.yukisako.xyz/entry/backpropagation
    # がわかりやすかった。

net = Net()

# 損失を計算する関数
criterion = nn.CrossEntropyLoss()
# optimizer = 最適化手法. lr = learning curve 学習率, momentum = 慣性項
# SGD = Stochastic Gradient Descent : 確率的勾配降下法
# https://qiita.com/tokkuman/items/1944c00415d129ca0ee9 に解説あり
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

transform = transforms.Compose(
    [
        # To Tensor converts PIL(=Python Image Library) image or numpy.ndarray to Tensor.
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

if __name__ == "__main__":
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    trainloader = data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = data.DataLoader(testset, batch_size=4, shuffle=True, num_workers=2)

    classes = ("plane", "car", "bird", "cat", "dear", "dog", "frog", "horse", "ship", "truck")

    # メインの学習処理
    for epoch in range(2):  # 用意された画像をバッチで分けながらそれぞれ2回処理する

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize を行う
            outputs = net(inputs)
            # 損失の計算を行う
            loss = criterion(outputs, labels)
            # 勾配の計算を行う
            loss.backward()
            # 最適化を行う
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    # 現在の学習済みモデルをPATHに保存する
    torch.save(net.state_dict(), PATH)

    # 逆に学習済みモデルを読み込むときは以下のように読み込む
    #net = Net()
    #net.load_state_dict(torch.load(PATH))
    #torch.load(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
