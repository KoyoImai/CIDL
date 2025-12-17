import logging
import copy
import numpy as np
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
from models.base import BaseLearner
from utils.inc_net import IncrementalNet, AKAIncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.loss import PES_Loss
from torchvision import transforms
from utils.toolkit import AutoencoderSigmoid
from utils.autoaugment import CIFAR10Policy
import time


EPSILON = 1e-8


class BASELINE_replay8(BaseLearner):
    def __init__(self, args):
        """
        _cur_task: 現在のタスクid（初期値-1）
        """

        super().__init__(args)

        self.args = args

        # backbone model の獲得
        self._network = IncrementalNet(args, False)

        # プロトタイプの初期化
        self._protos = {}

        #=== 使用する Unlearning 損失の種類 ===#
        # ex) "maxim_entropy"，"proto_cos"
        self.unleran_type = args["unlearn_type"]

        # 忘却クラスの初期化
        self.forget_classes = []                # 現在タスクで忘却する対象クラスのリスト
        self.forget_list = args["forget_cls"]   # 将来忘却をするクラスのリスト

        # データセットのサイズを設定
        if "cifar" in self.args["dataset"]:
            self.size = 32
        elif "tiny" in self.args["dataset"]:
            self.size = 56
        elif "imagenet" in self.args["dataset"]:
            self.size = 224
    
    def after_task(self):

        # これまでに学習したクラス数の更新
        self._known_classes = self._total_classes

        #=== 知識蒸留用教師モデルの更新 ===#
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        
        #=== リプレイバッファの更新 ===#
        if self.data_manager is not None:

            # 1クラスあたり保存するサンプル数
            m = self._memory_per_class

            logging.info(f"Update replay memory: m={m} per class")
            self.build_rehearsal_memory(self.data_manager, m)

        #=== チェックポイントの保存 ===#
        ckpt_dir = "checkpoint/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}/".format(
            self.args["model_name"],
            self.args["log_name"],
            self.args["dataset"],
            self.args["unlearn_type"],
            self.args["init_cls"],
            self.args["increment"],
            self.args["lambda_fkd"], self.args["lambda_proto"], self.args["lambda_pes"], self.args["lambda_pgru"], self.args["lambda_unl"])

        self.save_checkpoint(ckpt_dir)

    def incremental_train(self, data_manager):
        
        #=== data_manager の登録 ===#
        self.data_manager = data_manager
        self._cur_task += 1

        #=== 変更前のラベル順序を保存 ===#
        self._class_order = data_manager.get_class_order()

        #=== 現在タスクのクラスまでを含めた合計のクラス数を更新 ===#
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        #=== 忘却クラスの更新 ===#
        # 現在タスクで新しく忘却するクラス
        self.cur_forget_classes = [cls for cls in self.forget_list[self._cur_task]]

        # これまでに忘却したクラスの累積
        self.forget_classes += self.cur_forget_classes

        # 前タスクまでに学習して，知識を維持したいクラスのリスト
        self.learned_classes_list = [cls for cls in range(self._total_classes) if cls not in self.forget_classes]

        # 忘却するクラスの表示
        logging.info(
            "forget classes on task{}: total={} (new={})".format(
                self._cur_task, self.forget_classes, self.cur_forget_classes
            )
        )

        #=== model の構造を更新 ===#
        self._network.update_fc(self._total_classes*4)
        self._network_module_ptr = self._network
        
        # model の表示
        logging.info(
            "model: {}".format(self._network_module_ptr))
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))
        
        #=== dataloader の表示 ===#
        # 訓練用データセットの作成
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                 source="train",
                                                 mode="train",
                                                 appendent=None)

        # 訓練用データローダーの作成
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.args["batch_size"],
                                       shuffle=True,
                                       num_workers=self.args["num_workers"],
                                       pin_memory=True)
        
        # テスト用データセットの作成
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                source="test",
                                                mode="test")
        
        # テスト用データローダーの作成
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.args["batch_size"],
                                      shuffle=False,
                                      num_workers=self.args["num_workers"])
        
        # データパラレルの用意
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        #=== 訓練を実行 ===#
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):

        #=== 学習済みパラメータの読み込み === #
        resume = False
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("checkpoint/{}/{}/{}/{}/phase{}.pkl".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
            logging.info('!!!resume!!!')
        
        #=== model をデバイス上に配置 ===#
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        
        #=== タスクの学習 ===#
        if not resume:

            # ベースタスクの設定
            if self._cur_task == 0:
                self._epoch_num = self.args["init_epochs"]
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            
            # 追加タスクの設定
            else:
                trainable_list = nn.ModuleList([])
                trainable_list.append(self._network)
                self._epoch_num = self.args["epochs"]
                logging.info('All params total: {}'.format(count_parameters(trainable_list)))
                optimizer = torch.optim.Adam(trainable_list.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])

            # タスクの学習を実行
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        
        #=== プロトタイプの構築 ===#
        self._build_protos()
    
    def _build_protos(self):
        
        # プロトタイプの初期化
        prototype = {}

        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                
                # class_idx のデータセットを構築
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1),
                                                                           source="train",
                                                                           mode="test",
                                                                           ret_data=True)
                # class_idx のデータローダーを構築
                idx_loader = DataLoader(idx_dataset,
                                        batch_size=self.args["batch_size"], 
                                        shuffle=False,
                                        num_workers=4)
                
                # class_idx の特徴量を取り出す
                vectors, _ = self._extract_vectors(idx_loader)

                # class_idx の平均特徴を計算
                class_mean = np.mean(vectors, axis=0)

                # class_idx の平均特徴をプロトタイプとしてリストに追加
                prototype[class_idx] = class_mean
            
            # プロトタイプの更新
            self._protos.update(prototype)

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):

        #=== プログレスバーの設定 ===#
        prog_bar = tqdm(range(self._epoch_num))

        #=== 1エポックずつ学習 ===#
        for _, epoch in enumerate(prog_bar):

            #=== model を trainモード に変更
            self._network.train()

            #=== 記録用変数の初期化 ===#
            losses = 0.
            losses_new = 0.
            losses_fkd = 0.
            losses_proto = 0.
            losses_unl = 0.
            losses_unl_mem = 0.
            
            correct = 0.
            total = 0.

            #=== 1エポックの学習 ===#
            for i, (_, inputs, targets) in enumerate(train_loader):

                # ----------------------------------------
                # ① 現在タスクのバッチを gpu に載せる
                # ----------------------------------------
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                # ----------------------------------------
                # ② リプレイバッファから忘却用バッチを取り出す
                # ----------------------------------------
                mem_forg_inputs, mem_forg_targets = self._sample_memory_batch(
                    num_samples=self.args.get("forget_batch_size", 10),
                    target_classes=self.cur_forget_classes
                )
                # print("mem_forg_targets: ", mem_forg_targets)

                # if mem_forg_inputs is not None:
                #     inputs = torch.cat([inputs, mem_forg_inputs], dim=0)
                #     targets = torch.cat([targets, mem_forg_targets], dim=0)
                
                # ----------------------------------------
                # ③ リプレイバッファらから維持用バッチを取り出す
                # ----------------------------------------
                mem_retain_inputs, mem_retain_targets = self._sample_memory_batch(
                    num_samples=self.args.get("forget_batch_size", 10),
                    target_classes=self.learned_classes_list
                )
                # print("mem_retain_targets: ", mem_retain_targets)

                # if mem_retain_inputs is not None:
                #     inputs = torch.cat([inputs, mem_retain_inputs], dim=0)
                #     targets = torch.cat([targets, mem_retain_targets], dim=0)
                
                # ----------------------------------------
                # ④ rotation をかけて class augmentation
                # ----------------------------------------
                # 訓練用データに回転拡張を適用
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)

                # class augmentation に合わせてラベルを修正
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)

                # ----------------------------------------
                # ⑤ 損失を計算
                # ----------------------------------------
                logits, loss_new, loss_fkd, loss_proto, loss_di_forg, loss_di_retain = self._compute_loss(inputs, targets, aug_targets,
                                                                                                          mem_forg_inputs, mem_forg_targets,
                                                                                                          mem_retain_inputs, mem_retain_targets)
                loss = loss_new + loss_fkd + loss_proto + loss_unl

                # ----------------------------------------
                # ⑥ パラメータを更新
                # ----------------------------------------
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 記録の更新
                losses += loss.item()
                losses_new += loss_new.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_unl += loss_unl.item()

                # 正解率の計算
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aug_targets.expand_as(preds)).cpu().sum()
                total += len(aug_targets)
            
            # 学習率の調整
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)

            # 5 エポック毎に精度や損失を表示
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_unl/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_unl/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)

    def _compute_loss(self, inputs, targets, aug_targets, mem_forg_inputs, mem_forg_targets, mem_retain_inputs, mem_retain_targets):

        # 損失計算の前準備
        cur_forget = set(getattr(self, "cur_forget_classes", []))
        lambda_unl = self.args.get("lambda_unl", 0.0)

        #=== forward処理を行い features / logits を取り出す ===#
        features = self._network_module_ptr.extract_vector(inputs)      # [4B, D]
        logits = self._network_module_ptr.fc(features)["logits"]        # [4B, C*4]

        # === base task: Cross Entropy Loss だけ計算 ===#
        if self._cur_task == 0:

            # CE損失を計算
            loss_new = torch.nn.functional.cross_entropy(
                logits / self.args["temp"], aug_targets
            )

            # ベースタスクで使用しない損失は 0 で埋める
            loss_fkd = torch.tensor(0., device=self._device)
            loss_proto = torch.tensor(0., device=self._device)
            loss_forg = torch.tensor(0., device=self._device)
            loss_retain = torch.tensor(0., device=self._device)

            return logits, loss_new, loss_fkd, loss_proto, loss_forg, loss_retain

        #=== 忘却 / 保持サンプルのマスクを作成 ===#
        # バッチサイズ
        B = targets.shape[0]

        if len(cur_forget) > 0:

            # バッチの元サンプルが忘却対象かどうかを判定
            tgt_np = targets.detach().cpu().numpy()
            forget_flags = np.isin(tgt_np, np.array(list(cur_forget)))
            forget_flags = torch.from_numpy(forget_flags).to(self._device)  # bool [B]

            # rotation 分だけ 4 倍して [4B] に拡張
            forget_mask = torch.stack(
                [forget_flags for _ in range(4)], dim=1
            ).view(-1)  # bool [4B]
        
        else:
            forget_mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=self._device)

        retain_mask = ~forget_mask

        #=== 蒸留損失の計算 ===#
        # 教師モデルの出力を獲得
        with torch.no_grad():
            features_old = self.old_network_module_ptr.extract_vector(inputs)
        
        # マスクを使用して保持クラスの出力だけ取り出す
        if retain_mask.any():
            f_new = features[retain_mask]
            f_old = features_old[retain_mask]
            loss_fkd = self.args["lambda_fkd"] * torch.dist(f_new, f_old, 2)
        else:
            loss_fkd = torch.tensor(0., device=self._device)
        
        #=== プロトタイプ損失の計算 ===#
        proto_features = []
        proto_targets = []

        # 忘却するクラスのプロトタイプは使用しない
        old_class_list = list(self._protos.keys())
        old_class_list = [c for c in old_class_list if c not in self.forget_classes]

        # 維持クラスの出力を選択するためにインデックスを取り出す
        retain_indices = torch.nonzero(retain_mask).view(-1).cpu().numpy()

        if len(old_class_list) == 0:
            loss_proto = torch.tensor(0., device=self._device)
        else:
            for _ in range(features.shape[0] // 4):

                # 取り出す画像特徴のインデックス i をランダムに選択
                i = np.random.choice(retain_indices)

                # 使用するプロトタイプをランダムに選択
                np.random.shuffle(old_class_list)

                # プロトタイプと画像特徴を mixup
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self._protos[old_class_list[0]]  - lam * features.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self._protos[old_class_list[0]]  + lam * features.detach().cpu().numpy()[i]
                
                # mixup した特徴をリストに格納
                proto_features.append(temp)
                proto_targets.append(old_class_list[0])

            # 特徴とラベルを numpy から tensor に変更
            proto_features = torch.from_numpy(np.asarray(proto_features)).float().to(
                self._device, non_blocking=True
            )
            proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(
                self._device, non_blocking=True
            )

            # mixup 後の特徴量を fc層 に入力しプロトタイプ損失を計算
            proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
            loss_proto = self.args["lambda_proto"] * torch.nn.functional.cross_entropy(
                proto_logits / self.args["temp"], proto_targets * 4
            )
        
        #=== 維持クラスのみ対象にした Cross Entropy Loss を計算 ===#
        if retain_mask.any():
            logits_retain = logits[retain_mask]
            targets_retain = aug_targets[retain_mask]
            loss_clf = torch.nn.functional.cross_entropy(
                logits_retain / self.args["temp"], targets_retain
            )
        else:
            loss_clf = torch.tensor(0., device=self._device)
        
        loss_new = loss_clf

        #=== リプレイバッファに保存した忘却クラスのみを対象とする忘却損失 ===#
        loss_forg = torch.tensor(0., device=self._device)

        # mem_forg_inputs の forward 処理
        forg_features = self._network_module_ptr.extract_vector(mem_forg_inputs)  
        forg_logits = self._network_module_ptr.fc(forg_features)["logits"] 

        # 同じクラス c の特徴量のコサイン類似度を最小化することで忘却を発生する
        if self.args["unlearn_type"] == "minimum_cosine":
            feat = torch.nn.functional.normalize(forg_features, dim=1)  # [N, D]
            uniq = torch.unique(mem_forg_targets)

            per_class_means = []
            for c in uniq:
                idx = (mem_forg_targets == c)
                n_c = int(idx.sum().item())
                if n_c < 2:
                    continue

                f = feat[idx]          # [n_c, D]
                sim = f @ f.t()        # [n_c, n_c] (cos sim)
                
                # 対角(i=j)を除外して平均
                offdiag = sim[~torch.eye(n_c, dtype=torch.bool, device=sim.device)]
                per_class_means.append(offdiag.mean())

            if len(per_class_means) > 0:
                l_sim = torch.stack(per_class_means).mean()
                # 係数：lambda_forg があればそれ、なければ lambda_unl を流用
                
                loss_forg = self.args["lambda_forg"] * l_sim
        
        else:
            assert False

        #=== リプレイバッファに保存した維持クラスのみを対象とする蒸留損失 ===#
        loss_retain = torch.tensor(0., device=self._device)
        
        # 学習中モデルの forward 処理
        retain_features = self._network_module_ptr.extract_vector(mem_retain_inputs) 
        retain_logits = self._network_module_ptr.fc(retain_features)["logits"]

        # L2 ノルムによる蒸留損失
        if self.retain_type == "l2":
            
            # パラメータを凍結した過去モデル
            with torch.no_grad():
                retain_features_old = self.old_network_module_ptr.extract_vector(mem_retain_inputs)
            
            # retain損失の計算
            loss_retain = self.args["lambda_retain"] * torch.dist(retain_features, retain_features_old)
        
        else:
            assert False
        

        return logits, loss_new, loss_fkd, loss_proto, loss_forg, loss_retain




    #-------------------- リプレイバッファ関連の処理 --------------------
    def _sample_memory_batch(self, num_samples, target_classes=None):
        """
        self._data_memory / self._targets_memory から
        target_classes（指定がなければ self.forget_classes）に属するサンプルだけを
        ランダムに num_samples 個取り出して 1 バッチ分の (inputs, targets) を返す
        """

        #=== メモリがからの時は何もしない ===#
        if not hasattr(self, "_data_memory") or self._data_memory.size == 0:
            return None, None
        
        #=== 対照クラスが指定されていないなら何もしない ===#
        if target_classes is None or len(target_classes) == 0:
            return None, None

        #=== numpy の targets から target_classes に属する index を抜き出す ===#
        mask = np.isin(self._targets_memory, np.array(target_classes))
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None, None
        
        #=== self._data_memory / self._targets_memory から取り出す ===#
        # サンプル数の決定
        num = min(num_samples, len(idxs))
        
        # 取り出すサンプルのインデックスをランダムに決定
        sampled = np.random.choice(idxs, size=num, replace=False)

        # データとラベルを取り出す
        mem_data = self._data_memory[sampled]
        mem_targets = self._targets_memory[sampled]

        #=== DataManager 経由で学習用データを整形 ===#
        # リプレイ用データセットの作成
        mem_dataset = self.data_manager.get_dataset(
            [],
            source="train",
            mode="test",
            appendent=(mem_data, mem_targets),
        )

        # リプレイ用データローダーの作成
        mem_loader = DataLoader(
            mem_dataset,
            batch_size=num,
            shuffle=True,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )

        # リプレイ用ミニバッチを取り出す
        _, inputs, targets = next(iter(mem_loader))
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        return inputs, targets

    def build_rehearsal_memory(self, data_manager, per_class):
        
        """
        学習済みのクラスを対象に，n毎ずつリプレイバッファに保存
        """

        # クラス毎に保存するサンプル数
        n = int(per_class)

        if n < 0:
            return

        #=== リプレイバッファの初期化 ===#
        if not hasattr(self, "_data_memory") or getattr(self._data_memory, "size", 0) == 0:
            self._data_memory = np.array([])
            self._targets_memory = np.array([])
        
        #=== タスクやクラス情報を取り出す ===#
        task_size = data_manager.get_task_size(self._cur_task)
        start_cls = self._known_classes - task_size
        end_cls = self._known_classes

        # すでにリプレイバッファにあるクラスは対象から外す
        saved_cls = set(self._targets_memory.tolist()) if getattr(self._targets_memory, "size", 0) > 0 else set()

        #=== リプレイバッファに追加するデータの選択 ===#
        add_data_list, add_targets_list = [], []

        for cls in range(start_cls, end_cls):
            
            # すでに保存済みのクラスならスキップ
            if cls in saved_cls:
                continue
                
            # クラス cls のデータとラベルのみを取り出す
            data, targets, _ = data_manager.get_dataset(
                np.arange(cls, cls + 1),
                source="train",
                mode="test",
                ret_data=True,
            )

            # ランダムに取り出すデータを選択
            idxs = np.arange(len(targets))
            k = min(n, len(idxs))
            pick = np.random.choice(idxs, size=k, replace=False)

            # データとラベルをリストに格納
            add_data_list.append(data[pick])
            add_targets_list.append(targets[pick])

        # 追加保存しない場合は終了
        if len(add_data_list) == 0:
            return
        
        # numpy 配列として concat
        add_data = np.concatenate(add_data_list, axis=0)
        add_targets = np.concatenate(add_targets_list, axis=0)

        # 追記（削除・削減なし）
        if self._data_memory.size == 0:
            self._data_memory = add_data
            self._targets_memory = add_targets
        else:
            self._data_memory = np.concatenate([self._data_memory, add_data], axis=0)
            self._targets_memory = np.concatenate([self._targets_memory, add_targets], axis=0)

        logging.info(
            f"[Replay7] memory appended: +{len(add_targets)} samples "
            f"(classes {start_cls}-{end_cls-1}, n={n}/class). "
            f"total={len(self._targets_memory)}"
        )


    #-------------------- 評価関連の処理 --------------------   
    def eval_task(self):

        # -------------------------
        # CNN 評価
        # -------------------------
        #=== model の forward 処理 ===#
        y_pred, y_true = self._eval_cnn(self.test_loader)
        y_true = np.asarray(y_true)

        #=== 忘却クラス / 保持クラスを分割するためのマスクを作成 ===#
        forget_set = set(getattr(self, "forget_classes", []))
        if len(forget_set) > 0:
            mask_forget = np.isin(y_true, list(forget_set))
        else:
            mask_forget = np.zeros_like(y_true, dtype=bool)
        mask_retain = ~mask_forget

        #=== 保持クラスのみで精度を計算 ===#
        if mask_retain.any():
            y_pred_retain = y_pred[mask_retain]
            y_true_retain = y_true[mask_retain]
        else:
            logging.warning(
                "MU eval (CNN): no retain samples found, using all samples for metrics."
            )
            y_pred_retain = y_pred
            y_true_retain = y_true
        cnn_accy = self._evaluate(y_pred_retain, y_true_retain)

        #=== 忘却クラスのみで精度を計算 ===#
        if mask_forget.any():
            y_pred_forget = y_pred[mask_forget]
            top1_pred_forget = y_pred_forget[:, 0]
            y_true_forget = y_true[mask_forget]
            forget_acc_cnn = np.around(
                (top1_pred_forget == y_true_forget).sum() * 100.0 / len(y_true_forget),
                decimals=2,
            )
        else:
            forget_acc_cnn = None
        
        # dict に精度を保存しておく
        cnn_accy["forget_top1"] = forget_acc_cnn
        cnn_accy["num_retain_samples"] = int(mask_retain.sum())
        cnn_accy["num_forget_samples"] = int(mask_forget.sum())

        # 記録の表示
        logging.info(
            f"MU eval (CNN) - retain samples: {mask_retain.sum()}, "
            f"forget samples: {mask_forget.sum()}"
        )
        logging.info(f"MU eval (CNN) - forget top1: {forget_acc_cnn}")

        #=== 維持クラス / 忘却クラス の CNN における調和平均 ===#
        retain_acc_cnn = cnn_accy["top1"]

        if forget_acc_cnn is not None:
            forget_err_cnn = 100.0 - forget_acc_cnn
            if retain_acc_cnn + forget_err_cnn > 0:
                hmean_cnn = 2.0 * retain_acc_cnn * forget_err_cnn / (retain_acc_cnn + forget_err_cnn)
            else:
                hmean_cnn = 0.0
        else:
            forget_err_cnn = None
            hmean_cnn = None

        # dict に保存
        cnn_accy["forget_err"] = forget_err_cnn
        cnn_accy["hmean"] = hmean_cnn

        # 記録の表示
        logging.info(
            f"MU (CNN) retain_acc={retain_acc_cnn:.2f}, "
            f"forget_err={forget_err_cnn}, hmean={hmean_cnn}"
        )

        # -------------------------
        # NME 評価
        # -------------------------
        nme_accy = None
        y_pred_nme, y_true_nme = None, None

        if hasattr(self, "_protos") and len(self._protos) > 0:
            
            # protos を class means として使う場合
            protos = np.asarray(list(self._protos.values()))
            protos = protos / (np.linalg.norm(protos, axis=1, keepdims=True) + 1e-8)
            y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, protos)
        
        if y_pred_nme is not None:
            y_true_nme = np.asarray(y_true_nme)

            if len(forget_set) > 0:
                mask_forget_nme = np.isin(y_true_nme, list(forget_set))
            else:
                mask_forget_nme = np.zeros_like(y_true_nme, dtype=bool)
            mask_retain_nme = ~mask_forget_nme

            # 保持クラスのみで NME の精度を計算
            if mask_retain_nme.any():
                y_pred_retain_nme = y_pred_nme[mask_retain_nme]
                y_true_retain_nme = y_true_nme[mask_retain_nme]
            else:
                logging.warning(
                    "MU eval (NME): no retain samples found, using all samples for metrics."
                )
                y_pred_retain_nme = y_pred_nme
                y_true_retain_nme = y_true_nme

            nme_accy = self._evaluate(y_pred_retain_nme, y_true_retain_nme)

            # 忘却クラスの精度（NME）
            if mask_forget_nme.any():
                top1_pred_forget_nme = y_pred_nme[mask_forget_nme][:, 0]
                true_forget_nme = y_true_nme[mask_forget_nme]
                forget_acc_nme = np.around(
                    (top1_pred_forget_nme == true_forget_nme).sum()
                    * 100.0
                    / len(true_forget_nme),
                    decimals=2,
                )
            else:
                forget_acc_nme = None

            nme_accy["forget_top1"] = forget_acc_nme
            nme_accy["num_retain_samples"] = int(mask_retain_nme.sum())
            nme_accy["num_forget_samples"] = int(mask_forget_nme.sum())

            logging.info(f"MU eval (NME) - forget top1: {forget_acc_nme}")

            #=== 維持クラス / 忘却クラス の NME における調和平均 ===#
            retain_acc_nme = nme_accy["top1"]  # 保持クラスのみで計算した top1 (%)

            if forget_acc_nme is not None:
                forget_err_nme = 100.0 - forget_acc_nme
                if retain_acc_nme + forget_err_nme > 0:
                    hmean_nme = 2.0 * retain_acc_nme * forget_err_nme / (retain_acc_nme + forget_err_nme)
                else:
                    hmean_nme = 0.0
            else:
                forget_err_nme = None
                hmean_nme = None

            nme_accy["forget_err"] = forget_err_nme
            nme_accy["hmean"] = hmean_nme

            logging.info(
                f"MU (NME) retain_acc={retain_acc_nme:.2f}, "
                f"forget_err={forget_err_nme}, hmean={hmean_nme}"
            )

        return cnn_accy, nme_accy
    
    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"][:, ::4]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct)*100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"][:, ::4]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  



