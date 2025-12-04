# Class-Incremental-Decremental-Learning (CIDL)
クラス増加かつクラス減少学習のためのプログラムです．

## プログラムの全体像
学習・評価に使用するプログラムの全体像は以下の通りです．
```
SSOCL/
├── convs                    : データ拡張関連を実装したモジュール群
├── exps                     : 学習・評価の設定を記述する.yamlファイルの格納場所．
├── models                   : model関連を実装したモジュール群．
├── utils                    : Optimizer関連を実装したモジュール群．
├── main.py                  : 訓練・評価を実際に行うモジュール群．
├── reconstruction.ipynb     : 学習済みモデルが出力特徴から画像を再構成するノートブック．
└── trainer.py               : その他のモジュールを実装するutilsファイル．
```


## 実行方法
学習・評価の実行方法は以下の通りです．

<details>
  <summary><b>CIFAR-100の実行</b></summary>

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

<details>
  <summary><b>Tiny-ImageNetの実行</b></summary>

  [tiny-imagenet-200](https://github.com/rmccorm4/Tiny-Imagenet-200?tab=readme-ov-file)をダウンロードして，`utils/data.py`で指定されるディレクトリに配置してください．
  - Tiny-ImageNetでbaselineを学習
    ```
    python main.py --config=exps/BASELINE/tiny.json
    ```
  - Tiny-ImageNetでMachine Unlearning用のbaselineを学習
    ```
    python main.py --config=exps/BASELINE_MU/tiny.json
    ```
  - Tiny-ImageNetで通常のPRL2を学習
    ```
    python main.py --config=exps/PRL2/tiny.json
    ```
  - Tiny-ImageNetでMachine Unlearning用に調整したPRLを学習
    ```
    python main.py --config=exps/PRL_MU/tiny.json
    ```

<details>
  <summary><b>ImageNet100の実行</b></summary>

  [ImageNet](https://github.com/rmccorm4/Tiny-Imagenet-200?tab=readme-ov-file)をダウンロードして，`utils/data.py`で指定されるディレクトリに配置してください．
  - ImageNetでbaselineを学習
    ```
    python main.py --config=exps/BASELINE/imnet100.json
    ```

### 分析・調査用
リプレイバッファありの学習．
忘却クラスのデータが使用可能と仮定し，どのような損失ならbackbone側の知識を削除できるかを確かめるために使用する．
- CIFAR-100でbaseline-replayを学習
  ```
  python main.py --config=exps/BASELINE_replay/cifar.json
  ```
- CIFAR-100でbaseline-replay2を学習
  ```
  python main.py --config=exps/BASELINE_replay2/cifar.json
  ```
- ImageNet100でbaseline-replay2を学習
  ```
  python main.py --config=exps/BASELINE_replay2/imnet100.json
  ```

### ImageNet100
現在実装中です．
PRLと同じクラスが不明です．
<!-- Please note to change the paths of the different datasets in `utils/data` to the paths of your dataset files -->

## Dataset
We provide a version of the ImageNet-Subset dataset (randomly seeded 1993) that has been segmented for academic research use only. Click [here](https://mega.nz/file/9ikj1bbB#Zax1V7Q1xPlkxu8C9bOq8Ocq6WAu_-jtyqvta2hkTN0) to get it. Please contact me immediately if infringement or violation is involved.

