# ディレクトリ概要
手法ごとの学習や評価を実装したプログラムを配置しています．

## プログラムの全体像
このディレクトリに配置されるプログラムは以下の通りです．
```
models/
├── base.py          : base learnerを実装
├── BASELINE.py      : 忘却的アプローチを行わないベースラインを実装
├── BASELINE_MU.py   : 忘却的アプローチを行うベースラインを実装
├── finetune.py      : finetuningを行うベースラインの実装
├── PRL_MU.py        : PRLにエントロピー最大化損失を追加＆忘却クラスプロトタイプのエントロピー損失除去を実装
├── PRL.py           : 通常のPRLを実装
├── xxx.py           : （今後追加予定）
└── xxx.py           : （今後追加予定）
```


## 各アプローチの概要
### BASELINE
通常のNon Exampler Class Incremental Learning (NECIL) を実行するアプローチです．
損失関数として，以下を採用しています．
- loss_new : 新しいタスクのデータに対する通常の交差エントロピー損失
- loss_fkd : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
- loss_proto : 過去タスクのクラスに対応したプロトタイプを使用した交差エントロピー損失
それぞれの損失に対する重みは，`CIDL/expsBASELINE/xxx.json`に記述する`lambda_xxx`から指定できます．
対応する重みを以下に示します．
- lambda_fkd : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
- lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
