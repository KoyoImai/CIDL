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

<details>
  <summary><b>BASELINE</b></summary>

    通常のNon Exampler Class Incremental Learning (NECIL) を実行するアプローチです．
    損失関数として，以下を採用しています．

    ```
    loss_new   : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd   : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto : 過去タスクのクラスに対応したプロトタイプを使用した交差エントロピー損失
    ```

    それぞれの損失に対する重みは，`CIDL/exps/BASELINE/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    ```

 </details>



<details>
    <summary><b>BASELINE_MU</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加したアプローチです．
    損失関数として，以下を採用しています．
        ```
        loss_new   : 新しいタスクのデータに対する通常の交差エントロピー損失
        loss_fkd   : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
        loss_proto : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
        loss_unl   : 忘却クラスのプロトタイプに対するエントロピー最大化損失
        ```
    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
        ```
        lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
        lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
        lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
        ```

 </details>


<details>
  <summary><b>BASELINE_replay</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加し，リプレイバッファを追加したアプローチです．
    リプレイバッファに過去のデータを保存し，忘却損失にのみ過去データを使用します．
    BASELINE_replay2とは異なり，毎イタレーション必ず忘却クラスを取り出して，忘却損失を計算します．
    損失関数として，以下を採用しています．
    ```

    loss_new      : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd      : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto    : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
    loss_unl      : 忘却クラスのプロトタイプに対するエントロピー最大化損失
    loss_unl_mem  : リプレイバッファに保存した忘却クラスのデータに対するエントロピー最大化損失
    ```

    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．

    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>



<details>
  <summary><b>BASELINE_replay2</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加し，リプレイバッファを追加したアプローチです．
    リプレイバッファに過去のデータを保存し，忘却損失にのみ過去データを使用します．
    BASELINE_replayとは異なり，忘却クラスのミニバッチはランダムで取り出して学習します．
    損失関数として，以下を採用しています．
    ```
    loss_new      : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd      : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto    : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
    loss_unl      : 忘却クラスのプロトタイプに対するエントロピー最大化損失
    loss_unl_mem  : リプレイバッファに保存した忘却クラスのデータに対するエントロピー最大化損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>


<details>
  <summary><b>BASELINE_replay6</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加し，リプレイバッファを追加したアプローチです．
    リプレイバッファに過去のデータを保存し，忘却損失にのみ過去データを使用します．
    リプレイバッファのデータは，データローダーに混ぜず毎イタレーションリプレイバッファから直接取り出す．

    ```
    loss_new      : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd      : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto    : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
    loss_unl      : 忘却クラスのリプレイデータに対するコサイン類似度最大化損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>


 <details>
  <summary><b>BASELINE_DI</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加し，DeepInversion (DI) によるリプレイを追加したアプローチです．
    リプレイバッファに過去のデータを保存し，忘却損失にのみ過去データを使用します．
    リプレイバッファのデータは，データローダーに混ぜず毎イタレーションリプレイバッファから直接取り出す．

    ```
    loss_new      : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd      : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto    : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
    loss_unl      : 忘却クラスのリプレイデータに対するコサイン類似度最大化損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>

 <details>
  <summary><b>BASELINE_DIMMD</b></summary>

    BASELINEアプローチにMachine Unlearning (MU) 的な損失関数を追加し，DeepInversion (DI) によるリプレイを追加したアプローチです．
    BASELINE_DIとは異なり，DeepInversionに追加のアプローチ（MMD損失，feat_div損失，proto損失）を導入しています．
    リプレイバッファに過去のデータを保存し，忘却損失にのみ過去データを使用します．
    リプレイバッファのデータは，データローダーに混ぜず毎イタレーションリプレイバッファから直接取り出す．

    ```
    loss_new      : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_fkd      : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto    : 過去タスクのクラスに対応し，維持クラスに該当するプロトタイプでを使用した交差エントロピー損失（忘却クラスのプロトタイプは不使用）
    loss_unl      : 忘却クラスのリプレイデータに対するコサイン類似度最大化損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/BASELINE_MU/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>


<details>
  <summary><b>PRL</b></summary>

    NECIL手法PRLを実行するアプローチです．
    Machine Unlearning的な評価指標を実装していないので，PRLを評価したい場合はPRL2を実行してください．
    損失関数として，以下を採用しています．
    ```
    loss_new   : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_pes   : ベースタスクのみで使用するコサイン類似度を使用した特徴空間の制約損失
    loss_fkd   : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto : 過去タスクのクラスに対応したプロトタイプを使用した交差エントロピー損失
    loss_pkd   : プロトタイプを使用した蒸留損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/PRL/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_pes   : loss_pes（特徴空間の制約損失）に対する重み（デフォルトは0.1）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_pgru  : loss_pkd（プロトタイプを使用した蒸留損失）に対する重み（デフォルトは2）
    ```

 </details>


<details>
  <summary><b>PRL2</b></summary>

    NECIL手法PRLを実行するアプローチです．
    PRLとは異なり，Machine Unlearning的な評価を実装しているので，こちらを優先して実行するといいです．
    損失関数として，以下を採用しています．
    ```
    loss_new   : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_pes   : ベースタスクのみで使用するコサイン類似度を使用した特徴空間の制約損失
    loss_fkd   : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto : 過去タスクのクラスに対応したプロトタイプを使用した交差エントロピー損失
    loss_pkd   : プロトタイプを使用した蒸留損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/PRL2/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_pes   : loss_pes（特徴空間の制約損失）に対する重み（デフォルトは0.1）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_pgru  : loss_pkd（プロトタイプを使用した蒸留損失）に対する重み（デフォルトは2）
    ```

 </details>




<details>
  <summary><b>PRL_MU</b></summary>

    NECIL手法PRLにMU的損失関数を追加したアプローチです．
    損失関数として，以下を採用しています．
    ```
    loss_new   : 新しいタスクのデータに対する通常の交差エントロピー損失
    loss_pes   : ベースタスクのみで使用するコサイン類似度を使用した特徴空間の制約損失
    loss_fkd   : 新しいタスクのデータの特徴量に対するL2ノルムの蒸留損失
    loss_proto : 過去タスクのクラスに対応したプロトタイプを使用した交差エントロピー損失
    loss_pkd   : プロトタイプを使用した蒸留損失
    loss_unl   : 忘却クラスのプロトタイプに対するエントロピー最大化損失
    ```
    それぞれの損失に対する重みは，`CIDL/exps/PRL/xxx.json`に記述する`lambda_xxx`から指定できます．
    対応する重みを以下に示します．
    ```
    lambda_fkd   : loss_fkd（L2ノルムの蒸留損失）に対する重み（デフォルトは10）
    lambda_pes   : loss_pes（特徴空間の制約損失）に対する重み（デフォルトは0.1）
    lambda_proto : loss_proto（プロトタイプを使用した交差エントロピー損失）に対する重み（デフォルトは10）
    lambda_pgru  : loss_pkd（プロトタイプを使用した蒸留損失）に対する重み（デフォルトは2）
    lambda_unl   : loss_unl（忘却クラスのエントロピー最大化損失）に対する重み
    ```

 </details>





