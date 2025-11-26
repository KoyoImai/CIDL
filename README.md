# Class-Incremental-Decremental-Learning (CIDL)
クラス増加かつクラス減少学習のためのプログラムです．

## プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
```
SSOCL/
├── convs       : データ拡張関連を実装したモジュール群
├── exps        : 学習・評価の設定を記述する.yamlファイルの格納場所．
├── models      : model関連を実装したモジュール群．
├── utils       : Optimizer関連を実装したモジュール群．
├── main.py     : 訓練・評価を実際に行うモジュール群．
└── trainer.py  : その他のモジュールを実装するutilsファイル．
```


## 実行方法
学習・評価の実行方法は以下の通りです．

### CIFAR-100の実行
学習を実行すればデータセットが自動的にダウンロードされます．
- CIFAR-100でbaselineを学習
  ```
  python main.py --config=exps/BASELINE/cifar.json
  ```
- CIFAR-100でMachine Unlearning用のbaselineを学習
  ```
  python main.py --config=exps/BASELINE_MU/cifar.json
  ```
- CIFAR-100で通常のPRLを学習
  ```
  python main.py --config=exps/PRL/cifar.json
  ```
- CIFAR-100で通常のPRL2を学習
  ```
  python main.py --config=exps/PRL2/cifar.json
  ```
- CIFAR-100でMachine Unlearning用に調整したPRLを学習
  ```
  python main.py --config=exps/PRL_MU/cifar.json
  ```

### Tiny-ImageNetの実行
[tiny-imagenet-200](https://mega.nz/file/9ikj1bbB#Zax1V7Q1xPlkxu8C9bOq8Ocq6WAu_-jtyqvta2hkTN0)をダウンロードして，`utils/data.py`で指定されるディレクトリに配置してください．
- Tiny-ImageNetでbaselineを学習
  ```
  python main.py --config=exps/BASELINE/tiny.json
  ```
- Tiny-ImageNetでMachine Unlearning用のbaselineを学習
  ```
  python main.py --config=exps/BASELINE_MU/tiny.json
  ```

### ImageNet100
現在実装中です．
PRLと同じクラスが不明です．
<!-- Please note to change the paths of the different datasets in `utils/data` to the paths of your dataset files -->

## Dataset
We provide a version of the ImageNet-Subset dataset (randomly seeded 1993) that has been segmented for academic research use only. Click [here](https://mega.nz/file/9ikj1bbB#Zax1V7Q1xPlkxu8C9bOq8Ocq6WAu_-jtyqvta2hkTN0) to get it. Please contact me immediately if infringement or violation is involved.

