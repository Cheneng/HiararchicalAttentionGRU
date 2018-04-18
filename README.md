# HiararchicalAttentionGRU

> This repo implements the *Hierarchical Attention Networks for Document Classification* in PyTorch


<img src="./pictures/figure1.png" width="500" height="600">

You should rewrite the Dataset class in the data/dataset.py
and put your data in '/data/train' or any other directory.


run by

```
python main.py --lr=0.01 --epoch=20 --batch_size=16 --gpu=0 --seed=0 --label_num=2
```