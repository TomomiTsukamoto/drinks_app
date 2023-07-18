# -*- coding: utf-8 -*-
# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch.nn as nn
# 学習時に使用したものと同じ学習済みモデルをインポート
from torchvision.models import resnet18


#前処理
transforms  = transforms.Compose({
    transforms.ToTensor(), 
    transforms.Resize((224, 224))
})

class Net(pl.LightningModule):

    def __init__(self): 
        super().__init__()

        self.feature = resnet18(pretrained=True)
        self.fc = nn.Linear(1000, 2)


    def forward(self, x):
        h = self.feature(x)
        h = self.fc(h)
        return h



