
import os
import random
import logging
import copy
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.utils as vutils

from models.base import BaseLearner
from utils.inc_net import IncrementalNet, AKAIncrementalNet
from utils.toolkit import count_parameters, target2onehot, tensor2numpy
from utils.loss import PES_Loss
from torchvision import transforms
from utils.toolkit import AutoencoderSigmoid
from utils.autoaugment import CIFAR10Policy
import time

EPSILON = 1e-8

# ========================= DeepInversion helpers =========================
class DeepInversionFeatureHook:
    """
    BatchNorm2d の running_mean / running_var と
    実際の feature の mean / var のズレを L2 で測る hook。
    DeepInversion の r_feature loss 用。
    """
    # [DI-NEW]
    def __init__(self, module: nn.BatchNorm2d):
        self.r_feature = None
        self.hook = module.register_forward_hook(self._hook_fn)

    # [DI-NEW]
    def _hook_fn(self, module, input, output):
        x = input[0]  # (N, C, H, W)
        n, c = x.shape[:2]

        mean = x.mean(dim=[0, 2, 3])
        var = x.permute(1, 0, 2, 3).contiguous().view(c, -1).var(dim=1, unbiased=False)

        mean_bn = module.running_mean
        var_bn = module.running_var

        # L2 norm の和
        self.r_feature = torch.norm(mean - mean_bn, 2) + torch.norm(var - var_bn, 2)

    # [DI-NEW]
    def close(self):
        self.hook.remove()

def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2

def lr_policy(lr_fn):
    def _alr(optimizer, iteration, epoch):
        lr = lr_fn(iteration, epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return _alr

def lr_cosine_policy(base_lr, warmup_length, epochs):
    def _lr_fn(iteration, epoch):
        if epoch < warmup_length:
            lr = base_lr * (epoch + 1) / warmup_length
        else:
            e = epoch - warmup_length
            es = epochs - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        return lr

    return lr_policy(_lr_fn)

def clip(image_tensor, use_fp16=False):
    '''
    adjust the input based on mean and variance
    '''
    if use_fp16:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float16)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float16)
    else:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    for c in range(3):
        m, s = mean[c], std[c]
        image_tensor[:, c] = torch.clamp(image_tensor[:, c], -m / s, (1 - m) / s)
    return image_tensor


# ============================================
# RBF カーネル & MMD 損失の定義
# ============================================
import torch
import torch.nn as nn


class RBF(nn.Module):
    """
    RBF (Gaussian) カーネル行列を計算するクラス。
    
    k(x_i, x_j) = exp(- ||x_i - x_j||^2 / bandwidth)
    
    - bandwidth を指定しない場合は，
      バッチ内の平均距離^2 から簡易的に推定します。
    """

    def __init__(self, bandwidth: float = None):
        """
        Args:
            bandwidth (float or None):
                カーネルの帯域幅。
                None のときはデータから自動推定。
        """
        super().__init__()
        self.bandwidth = bandwidth

    def _get_bandwidth(self, dist2: torch.Tensor) -> torch.Tensor:
        """
        L2 距離^2 の行列から bandwidth を決めるヘルパー関数。
        dist2: 形状 (N, N)
        """
        if self.bandwidth is not None:
            # ユーザ指定がある場合はそれを使う
            return torch.tensor(self.bandwidth, device=dist2.device, dtype=dist2.dtype)

        n = dist2.shape[0]
        if n <= 1:
            # サンプルが1つ以下だと平均距離が定義できないので適当に1.0
            return torch.tensor(1.0, device=dist2.device, dtype=dist2.dtype)

        # 全要素（対角も含む）の平均距離^2 を bandwidth として使う簡易ヒューリスティック
        # （より厳密にやるなら対角を除いたり，median heuristic にしてもよい）
        bw = dist2.sum() / (n * n)
        return bw.detach()

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        入力:
            X: 形状 (N, D) の特徴ベクトル集合
        出力:
            K: 形状 (N, N) の RBF カーネル行列
        """
        # 全ペアの L2 距離^2 を計算
        # torch.cdist(X, X) は (N, N) の距離行列を返す
        dist2 = torch.cdist(X, X) ** 2  # (N, N)

        bw = self._get_bandwidth(dist2)  # スカラー（tensor）
        # k(x_i, x_j) = exp(- dist2 / bw)
        K = torch.exp(- dist2 / (bw + 1e-8))
        return K

class MMDLoss(nn.Module):
    """
    RBF カーネルを用いた MMD^2 損失。
    
    MMD^2(X, Y) = E[k(X, X)] + E[k(Y, Y)] - 2 E[k(X, Y)]
    
    ここでは biased な推定量（全要素の単純平均）を用いています。
    """

    def __init__(self, kernel: nn.Module = None):
        """
        Args:
            kernel: カーネルとして使うモジュール。
                    デフォルトは上で定義した RBF。
        """
        super().__init__()
        self.kernel = kernel if kernel is not None else RBF()

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: 形状 (N_x, D) のテンソル
            Y: 形状 (N_y, D) のテンソル
        Returns:
            mmd2: スカラーの MMD^2
        """
        # X と Y を縦に連結して，まとめてカーネル行列を計算
        Z = torch.vstack([X, Y])          # (N_x + N_y, D)
        K = self.kernel(Z)                # (N_x + N_y, N_x + N_y)

        n_x = X.shape[0]

        # ブロックを切り出す
        K_xx = K[:n_x, :n_x]              # X 対 X
        K_xy = K[:n_x, n_x:]              # X 対 Y
        K_yy = K[n_x:, n_x:]              # Y 対 Y

        # 全要素の平均で expectation を近似
        XX = K_xx.mean()
        XY = K_xy.mean()
        YY = K_yy.mean()

        # MMD^2 = E[k(X,X)] - 2E[k(X,Y)] + E[k(Y,Y)]
        mmd2 = XX - 2.0 * XY + YY
        return mmd2

# MMD のバッチ内計算ヘルパー
def compute_batch_mmd(feat: torch.Tensor,
                      targets: torch.Tensor,
                      real_features: dict,
                      mmd_loss_fn: nn.Module,
                      max_real_per_class: int = 100) -> torch.Tensor:
    """
    各クラス c について：
      MMD^2( DI特徴_c , 実特徴_c ) を計算し，クラス平均を返す。
    feat:    (bs, D)   DI 画像の特徴
    targets: (bs,)     クラスID
    """
    uniq_classes = targets.unique().tolist()
    mmd_list = []

    for c in uniq_classes:
        c_int = int(c)
        # DI 側
        mask_di = (targets == c)
        feat_di_c = feat[mask_di]          # (n_c, D)
        if feat_di_c.shape[0] == 0:
            continue

        # 実側
        if c_int not in real_features:
            continue
        feat_real_c = real_features[c_int]  # (N_real, D) on CPU
        if feat_real_c.shape[0] == 0:
            continue

        # 実側を max_real_per_class 個に制限（計算コスト対策）
        if feat_real_c.shape[0] > max_real_per_class:
            idx = torch.randperm(feat_real_c.shape[0])[:max_real_per_class]
            feat_real_c = feat_real_c[idx]

        feat_real_c = feat_real_c.to(feat.device)

        # MMD^2 を計算
        mmd2_c = mmd_loss_fn(feat_di_c, feat_real_c)
        mmd_list.append(mmd2_c)

    if len(mmd_list) == 0:
        return torch.tensor(0.0, device=feat.device)
    else:
        return torch.stack(mmd_list).mean()






class BASELINE_DIMMD3(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        #=== Backbone model の獲得 ===#
        self._network = IncrementalNet(args, False)
        self._memory_data = []
        self._memory_targets = []

        #=== プロトタイプの初期化 ===#
        self._protos = {}

        #=== 使用する Unlearning / retain 損失の種類 ===#
        # ex) "maxim_entropy"，"proto_cos", "minimum_cosine"
        self.unleran_type = args["unlearn_type"]

        # ex) "l2"
        self.retain_type = args["retain_type"]

        #=== 忘却クラス関連の処理 ===#
        self.forget_list = args["forget_cls"]   # タスク毎の忘却予定リスト
        self.forget_classes = []
        self.cur_forget_classes = []

        # 全ての忘却クラスをまとめたリスト
        self.all_forget_classes = sorted(
            {c for task in self.forget_list for c in task}
        )

        #=== リプレイバッファ＆DeepInversion関係の設定 ===#
        # 1クラスあたり保存するサンプル数 n
        mem_per_cls = args.get("memory_per_class", 0)
        self._memory_per_class = mem_per_cls

        # 学習に使用する DI画像 のバッチサイズ
        self.retain_batch_size = args["retain_batch_size"]
        self.forget_batch_size = args["forget_batch_size"]

        #=== データセット関係の設定 ===#
        if "cifar" in self.args["dataset"]:
            self.size = 32
            self.num_classes = 100
        elif "tiny" in self.args["dataset"]:
            self.size = 56
            self.num_classes = 200
        elif "imagenet" in self.args["dataset"]:
            self.size = 224
            self.num_classes = 100
        
        #=== DeepInversion用の初期化設定 ===#
        self.di_batch_size   = args["di_batch_size"]     
        self.di_iterations   = args["di_iterations"]  # 最適化ステップ数
        self.di_lr           = args["di_lr"]
        self.di_r_feature    = args["di_r_feature"]   # BN loss の係数
        self.di_tv_l2        = args["di_tv_l2"]       # TV loss の係数
        self.di_tv_l1        = args["di_tv_l1"]
        self.di_l2           = args["di_l2"]          # 画像 L2 loss の係数
        self.di_mmd          = args["di_mmd"]         # MMD損失の係数
        self.di_proto        = args["di_proto"]       # MMD損失の係数
        self.di_feat_div     = args["di_feat_div"]    # MMD損失の係数
        self.main_loss_multiplier = args["main_loss_multiplier"]

        # MMD損失のために実画像の特徴を保存する
        self.real_features = {c: [] for c in range(self.num_classes)}
        self.real_features_tensor = None
        self.num_features_per_class = args["num_features_per_class"]

    #-------------------- タスク後の後処理 --------------------
    def after_task(self):

        # これまでに学習したクラス数の更新
        self._pre_known_classes = self._known_classes
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

            # 実画像の特徴量を保存
            self.build_real_features()

            logging.info(f"Update replay memory: m={m} per class")
            self.build_rehearsal_memory(self.data_manager, m)

        #=== チェックポイントの保存 ===#
        # ckpt_dir = "checkpoint/{}/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}/".format(
        #     self.args["model_name"],
        #     self.args["log_name"],
        #     self.args["dataset"],
        #     self.args["unlearn_type"],
        #     self.args["init_cls"],
        #     self.args["increment"],
        #     self.args["lambda_fkd"], self.args["lambda_proto"], self.args["lambda_pes"], self.args["lambda_pgru"], self.args["lambda_unl"])
        ckpt_dir = "checkpoint/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}/".format(
            self.args["model_name"],
            self.args["log_name"],
            self.args["dataset"],
            self.args["init_cls"],
            self.args["increment"],
            self.args["lambda_fkd"], self.args["lambda_proto"], self.args["lambda_pes"], self.args["lambda_pgru"], self.args["lambda_unl"])

        # チェックポイントの保存
        self.save_checkpoint(ckpt_dir)

        # DI で生成した画像の保存
        if isinstance(self._data_memory, np.ndarray) and self._data_memory.size > 0:
            self._save_di_images(ckpt_dir)
        
        # 保存した実画像の特徴量の保存
        target_classes = list(range(0, self._total_classes))

        # 何も特徴が無い場合はスキップ
        has_any = any(
            isinstance(self.real_features[c], torch.Tensor)
            and self.real_features[c].numel() > 0
            for c in target_classes
        )
        if has_any:
            # CPU に移して保存
            real_feat_cpu = {
                c: self.real_features[c].cpu()
                for c in target_classes
            }

            save_obj = {
                "features": real_feat_cpu,                 # dict: class_id -> (N_c, D)
                "classes": target_classes,                 # 保存したクラスID
                "num_features_per_class": self.num_features_per_class,
            }

            real_feat_path = os.path.join(
                ckpt_dir, f"real_features_task{self._cur_task}.pth"
            )
            torch.save(save_obj, real_feat_path)
            logging.info(f"Saved real features for MMD to: {real_feat_path}")
        else:
            logging.info("No real features to save for this task.")

    #-------------------- 訓練関連の処理 --------------------
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
            path = "checkpoint/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}/phase{}.pkl".format(
                self.args["model_name"],
                self.args["log_name"],
                self.args["dataset"],
                self.args["init_cls"],
                self.args["increment"],
                self.args["lambda_fkd"], self.args["lambda_proto"], self.args["lambda_pes"], self.args["lambda_pgru"], self.args["lambda_unl"],
                self._cur_task)
            self._network.load_state_dict(torch.load(path)["model_state_dict"])
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
            losses_forg = 0.
            losses_retain = 0.

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
                    num_samples=self.args.get("forget_batch_size", 32),
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
                    num_samples=self.args.get("retain_batch_size", 32),
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
                loss = loss_new + loss_fkd + loss_proto + loss_di_forg + loss_di_retain

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
                losses_forg += loss_di_forg.item()
                losses_retain += loss_di_retain.item()

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
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Loss_retain {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_forg/len(train_loader), losses_retain/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Loss_retain {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_forg/len(train_loader), losses_retain/len(train_loader), train_acc, test_acc)
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

        #=== DeepInversionで生成した忘却クラスのみを対象とする忘却損失 ===#
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

        #=== DeepInversionで生成した維持クラスのみを対象とする蒸留損失 ===#
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
            setup_replay=False,
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

        #=== 保持サンプル数が0なら終了 ===#
        n = int(per_class)
        if n <= 0:
            logging.info("[DI-DIMMD2] Skip building DI memory (n <= 0).")
            self._data_memory = np.array([], dtype=np.uint8)
            self._targets_memory = np.array([], dtype=np.int64)
            return
        
        #=== 対象とするクラスの決定 ===#
        # すでに忘却済みのクラス
        already_forgotten = set(getattr(self, "forget_classes", []))

        # DIで生成するクラス（現在のタスクで学習したクラスに限定）
        target_classes = [
            c for c in range(self._pre_known_classes, self._total_classes)
        ]

        logging.info(
            f"[DI-DIMMD] Building DI memory for ALL non-forgotten classes: "
            f"n={n} per class, target_classes={target_classes}, "
            f"already_forgotten={sorted(already_forgotten)}"
        )

        # 生成対象のクラスがなければ終了
        if len(target_classes) == 0:
            logging.info("[DI-DIMMD] No target classes (all already forgotten).")
            return

        #=== DeepInversionによる画像の生成 ===#
        all_exemplars = []
        all_labels = []

        # labels_all = [0,0,0, 1,1,1, 5,5,5]
        labels_all = np.repeat(np.array(target_classes, dtype=np.int64), n)

        # # （必要ならシャッフル）
        perm = np.random.permutation(labels_all.shape[0])
        labels_all = labels_all[perm]

        max_batch = getattr(self, "di_batch_size", 64)
        start = 0
        total = labels_all.shape[0]

        # 繰り返して画像を生成する
        while start < total:

            end = min(start + max_batch, total)
            batch_labels = labels_all[start:end]  # (B,)

            if self.args["dataset"] in ["imagenet100"]:
                di_imgs = self._generate_di_images_for_labels(batch_labels)
            elif self.args["dataset"] in ["cifar100"]:
                di_imgs = self._generate_di_images_for_labels_cifar(batch_labels)
            else:
                raise NotImplementedError

            all_exemplars.append(di_imgs)      # (B, H, W, 3)
            all_labels.append(batch_labels)    # (B,)
            start = end
        
        new_data = np.concatenate(all_exemplars, axis=0)   # (N, H, W, 3)
        new_labels = np.concatenate(all_labels, axis=0)    # (N,)

        # 既存メモリが無ければ初期化
        if (not hasattr(self, "_data_memory")) or (self._data_memory.size == 0):
            self._data_memory = new_data.astype(np.uint8)
            self._targets_memory = new_labels.astype(np.int64)
        else:
            self._data_memory = np.concatenate([self._data_memory, new_data.astype(np.uint8)], axis=0)
            self._targets_memory = np.concatenate([self._targets_memory, new_labels.astype(np.int64)], axis=0)
        
        logging.info(
            f"[DI-DIMMD2] Built DI memory: data={self._data_memory.shape}, "
            f"labels classes={sorted(set(self._targets_memory.tolist()))}"
        )

    def _generate_di_images_for_labels(self, class_labels: np.ndarray) -> np.ndarray:
        """
        class_labels: np.ndarray, shape (B,)
            各要素が「元のクラスID」（0~num_classes-1）

        戻り値:
            imgs: np.ndarray, shape (B, H, W, 3), dtype=uint8
        """

        # パラメータの初期化
        device = self._device
        setting_id = 0
        jitter = 30
        B = int(class_labels.shape[0])
        H = self.size
        W = self.size

        iters = self.di_iterations
        lr = self.di_lr
        T = 1.0
        main_loss_multiplier = self.main_loss_multiplier
        r_feature_coeff = self.di_r_feature
        tv_l2_coeff = self.di_tv_l2
        tv_l1_coeff = self.di_tv_l1
        l2_coeff = self.di_l2

        first_bn_multiplier = 10
        do_flip = True

        # ============================================
        # プロトタイプを用意（通常のDeepInversionと異なる箇所）
        # ============================================
        protos_dict = self._protos  # {label: proto_vec}
        proto_labels = sorted(protos_dict.keys())

        # 各 proto を tensor にして並べる
        protos_list = []
        for c in proto_labels:
            v = protos_dict[c]                         # np.array or torch.Tensor
            v = torch.as_tensor(v, dtype=torch.float32)
            protos_list.append(v)

        protos_tensor = torch.stack(protos_list, dim=0).to(device=device)  # (C, D)

        # ラベル → 行index のマップを作っておく
        label2row = {c: i for i, c in enumerate(proto_labels)}
        C, D = protos_tensor.shape
        print("num protos:", C, "feat dim:", D)


        # teacher: after_task で更新した old_network を優先的に使う
        teacher = getattr(self, "old_network_module_ptr", None)
        if teacher is None:
            teacher = self._network_module_ptr
        teacher.eval()

        ## Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # 最適化入力の初期化
        inputs = torch.randn((B, 3, self.size, self.size), requires_grad=True, device=device)
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        targets = torch.from_numpy(class_labels).to(device=device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()
        mmd_loss_fn = MMDLoss()

        if setting_id==0:
            skipfirst = False
        else:
            skipfirst = True
        

        iteration = 0
        best_cost = float("inf")
        best_inputs = None
        for lr_it, lower_res in enumerate([2, 1]):
            if lr_it==0:
                iterations_per_layer = 2000
                # iterations_per_layer = 100
            else:
                iterations_per_layer = 1000 if not skipfirst else 2000
                # iterations_per_layer = 100 if not skipfirst else 100
                if setting_id == 2:
                    iterations_per_layer = 20000
            
            if lr_it==0 and skipfirst:
                continue

            lim_0, lim_1 = jitter // lower_res, jitter // lower_res

            if setting_id == 0:
                #multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif setting_id == 1:
                #2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.5, 0.9], eps = 1e-8)
                do_clip = True
            elif setting_id == 2:
                #20k normal resolution the closes to the paper experiments for ResNet50
                optimizer = optim.Adam([inputs], lr=self.di_lr, betas=[0.9, 0.999], eps = 1e-8)
                do_clip = False
            
            lr_scheduler = lr_cosine_policy(self.di_lr, 100, iterations_per_layer)


            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                # learning rate scheduling
                lr_scheduler(optimizer, iteration_loc, iteration_loc)


                # perform downsampling if needed
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs

                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                # Flipping
                flip = random.random() > 0.5
                if flip and do_flip:
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))

                # forward pass
                optimizer.zero_grad()
                teacher.zero_grad()

                outputs = teacher(inputs_jit)
                logits_all = outputs["logits"]
                logits = logits_all[:, ::4]

                # R_cross classification loss
                loss = criterion(logits, targets)

                # 特徴の多様性最大化損失（未完成）
                feature = outputs["features"]
                feat = feature.view(feature.size(0), -1)   # (bs, D)
                # print('feature.shape: ', feature.shape)
                # print("feat.shape: ", feat.shape)

                bs = feat.size(0)

                # ペアごとの差分ベクトル
                # print("feat.unsqueeze(1).shape: ", feat.unsqueeze(1).shape)
                # print("feat.unsqueeze(0).shape: ", feat.unsqueeze(0).shape)
                diff = feat.unsqueeze(1) - feat.unsqueeze(0)   # (bs, bs, D)
                dist2 = (diff ** 2).sum(dim=2)                 # (bs, bs), L2距離の二乗

                # 同一クラスかどうかのマスク
                same_label = (targets.unsqueeze(0) == targets.unsqueeze(1))  # (bs, bs) bool

                # 自分自身 (i == j) のペアは除外
                eye = torch.eye(bs, dtype=torch.bool, device=targets.device)
                same_label = same_label & (~eye)

                if same_label.any():
                    same_dist2 = dist2[same_label]          # 同じクラス同士の距離だけ
                    # 距離を「最大化」したいので，逆数を計算
                    loss_div = 1.0 / same_dist2.mean()
                else:
                    loss_div = torch.zeros(1, device=feat.device)
                

                # --------------------------
                # main loss 2: プロトタイプと平均特徴の損失（未完成）
                # --------------------------
                loss_proto_list = []

                uniq_classes = targets.unique().tolist()
                for c in uniq_classes:
                    c_int = int(c)
                    # このクラスの DI 特徴を集める
                    mask_c = (targets == c)
                    feat_c = feat[mask_c]                 # (n_c, D)
                    if feat_c.shape[0] == 0:
                        continue

                    # クラス c の DI 特徴の平均 μ_c
                    feat_mean_c = feat_c.mean(dim=0)      # (D,)

                    # 対応するプロトタイプ p_c を取得
                    if c_int not in label2row:
                        continue
                    proto_c = protos_tensor[label2row[c_int]]  # (D,)

                    # μ_c と p_c の L2 距離^2
                    loss_c = ((feat_mean_c - proto_c) ** 2).sum()
                    loss_proto_list.append(loss_c)

                if len(loss_proto_list) == 0:
                    loss_proto = torch.tensor(0.0, device=feat.device)
                else:
                    # クラス平均
                    loss_proto = torch.stack(loss_proto_list).mean()

                
                # --------------------------
                # main loss 3: MMD (DI vs Real features)
                # --------------------------
                loss_mmd = compute_batch_mmd(feat, targets, self.real_features, mmd_loss_fn,
                                            max_real_per_class=self.num_features_per_class)

                # R_prior losses
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                # R_feature loss
                rescale = [first_bn_multiplier] + [1. for _ in range(len(loss_r_feature_layers)-1)]
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                # l2 loss on images
                loss_l2 = torch.norm(inputs_jit.view(B, -1), dim=1).mean()

                # combining losses
                loss_aux = tv_l2_coeff * loss_var_l2 + \
                            tv_l1_coeff * loss_var_l1 + \
                            r_feature_coeff * loss_r_feature + \
                            l2_coeff * loss_l2
                        
                loss = main_loss_multiplier * loss + loss_aux + loss_div * self.di_feat_div + loss_proto * self.di_proto + loss_mmd * self.di_mmd

                if iteration % 500 == 0:
                    logging.info("------------iteration {}----------".format(iteration))
                    logging.info("total loss{}".format(loss.item()))
                    logging.info("loss_r_feature{}".format(loss_r_feature.item()))
                    logging.info("loss_div{}".format(loss_div.item()))
                    logging.info("loss_proto{}".format(loss_proto.item()))
                    logging.info("loss_mmd{}".format(loss_mmd.item()))
                    logging.info("main criterion{}".format(criterion(logits, targets).item()))
                
                loss.backward()
                optimizer.step()

                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=False)


                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()

                # if iteration % 100==0:
                #     vutils.save_image(inputs,
                #                         '{}/best_images/output_{:05d}_gpu.png'.format(prefix, iteration // 100,),
                #                         normalize=True, scale_each=True, nrow=int(10))

        for h in loss_r_feature_layers:
            h.close()      # module.register_forward_hook を解除
        
        with torch.no_grad():
            out = best_inputs.clone()

            # ImageNet 正規化を戻す
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            out = out * std + mean
            out = torch.clamp(out, 0.0, 1.0)
            out = (out * 255.0).byte()
            out = out.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
        
        return out

    def _generate_di_images_for_labels_cifar(self, class_labels: np.ndarray) -> np.ndarray:
        """
        class_labels: np.ndarray, shape (B,)
            各要素が「元のクラスID」（0~num_classes-1）

        戻り値:
            imgs: np.ndarray, shape (B, H, W, 3), dtype=uint8
        """

        # パラメータの初期化
        device = self._device
        B = int(class_labels.shape[0])
        H = self.size
        W = self.size

        iters_mi = self.di_iterations
        lr = self.di_lr
        main_loss_multiplier = self.main_loss_multiplier
        r_feature_coeff = self.di_r_feature
        tv_l2_coeff = self.di_tv_l2
        tv_l1_coeff = self.di_tv_l1
        l2_coeff = self.di_l2

        lim_0, lim_1 = 2, 2

        # ============================================
        # プロトタイプを用意（通常のDeepInversionと異なる箇所）
        # ============================================
        protos_dict = self._protos  # {label: proto_vec}
        proto_labels = sorted(protos_dict.keys())

        # 各 proto を tensor にして並べる
        protos_list = []
        for c in proto_labels:
            v = protos_dict[c]                         # np.array or torch.Tensor
            v = torch.as_tensor(v, dtype=torch.float32)
            protos_list.append(v)

        protos_tensor = torch.stack(protos_list, dim=0).to(device=device)  # (C, D)

        # ラベル → 行index のマップを作っておく
        label2row = {c: i for i, c in enumerate(proto_labels)}
        C, D = protos_tensor.shape
        print("num protos:", C, "feat dim:", D)


        # teacher: after_task で更新した old_network を優先的に使う
        teacher = getattr(self, "old_network_module_ptr", None)
        if teacher is None:
            teacher = self._network_module_ptr
        teacher.eval()

        ## Create hooks for feature statistics catching
        loss_r_feature_layers = []
        for module in teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))

        # 最適化入力の初期化
        data_type = torch.float
        inputs = torch.randn((B, 3, H, W), requires_grad=True, device=device, dtype=data_type)
        targets = torch.from_numpy(class_labels).to(device=device, dtype=torch.long)
        optimizer = optim.Adam([inputs], lr=lr)
        criterion = nn.CrossEntropyLoss()
        mmd_loss_fn = MMDLoss()

        # 学習部分
        best_cost = float("inf")
        best_inputs = None
        for epoch in range(iters_mi):
            
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1,off2), dims=(2,3))

            # foward with jit images
            optimizer.zero_grad()
            teacher.zero_grad()
            outputs = teacher(inputs_jit)
            
            # 交差エントロピー損失
            logits_all = outputs["logits"]
            logits = logits_all[:, ::4] 
            loss = criterion(logits, targets)

            # 特徴の多様性最大化損失
            feature = outputs["features"]
            feat = feature.view(feature.size(0), -1)   # (bs, D)

            bs = feat.size(0)

            # ペアごとの差分ベクトル
            # print("feat.unsqueeze(1).shape: ", feat.unsqueeze(1).shape)
            # print("feat.unsqueeze(0).shape: ", feat.unsqueeze(0).shape)
            diff = feat.unsqueeze(1) - feat.unsqueeze(0)   # (bs, bs, D)
            dist2 = (diff ** 2).sum(dim=2)                 # (bs, bs), L2距離の二乗

            # 同一クラスかどうかのマスク
            same_label = (targets.unsqueeze(0) == targets.unsqueeze(1))  # (bs, bs) bool

            # 自分自身 (i == j) のペアは除外
            eye = torch.eye(bs, dtype=torch.bool, device=device)
            same_label = same_label & (~eye)

            if same_label.any():
                same_dist2 = dist2[same_label]          # 同じクラス同士の距離だけ
                # 距離を「最大化」したいので，逆数を計算
                loss_div = 1.0 / same_dist2.mean()
            else:
                loss_div = torch.zeros(1, device=device)
            

            # --------------------------
            # main loss 2: プロトタイプと平均特徴の損失（未完成）
            # --------------------------
            loss_proto_list = []

            uniq_classes = targets.unique().tolist()
            for c in uniq_classes:
                c_int = int(c)
                # このクラスの DI 特徴を集める
                mask_c = (targets == c)
                feat_c = feat[mask_c]                 # (n_c, D)
                if feat_c.shape[0] == 0:
                    continue

                # クラス c の DI 特徴の平均 μ_c
                feat_mean_c = feat_c.mean(dim=0)      # (D,)

                # 対応するプロトタイプ p_c を取得
                if c_int not in label2row:
                    continue
                proto_c = protos_tensor[label2row[c_int]]  # (D,)

                # μ_c と p_c の L2 距離^2
                loss_c = ((feat_mean_c - proto_c) ** 2).sum()
                loss_proto_list.append(loss_c)

            if len(loss_proto_list) == 0:
                loss_proto = torch.tensor(0.0, device=device)
            else:
                # クラス平均
                loss_proto = torch.stack(loss_proto_list).mean()

            
            # --------------------------
            # main loss 3: MMD (DI vs Real features)
            # --------------------------
            loss_mmd = compute_batch_mmd(feat, targets, self.real_features, mmd_loss_fn,
                                            max_real_per_class=self.num_features_per_class)

            # apply total variation regularization
            diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
            diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
            diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
            diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = main_loss_multiplier * loss + tv_l2_coeff * loss_var

            # R_feature loss
            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            loss = loss + r_feature_coeff * loss_distr # best for noise before BN

            # l2 loss
            loss_l2 = torch.norm(inputs_jit, 2)
            loss = loss + l2_coeff * loss_l2 + loss_div * self.di_feat_div + loss_proto * self.di_proto + loss_mmd * self.di_mmd

            if epoch % 500 == 0:
                logging.info("------------iteration {}----------".format(epoch))
                logging.info("total loss{}".format(loss.item()))
                logging.info("loss_r_feature{}".format(loss_distr.item()))
                logging.info("loss_div{}".format(loss_div.item()))
                logging.info("loss_proto{}".format(loss_proto.item()))
                logging.info("loss_mmd{}".format(loss_mmd.item()))
                logging.info("main criterion{}".format(criterion(logits, targets).item()))
                
            loss.backward()
            optimizer.step()

            if best_cost > loss.item() or epoch == 1:
                best_inputs = inputs.data.clone()
                best_cost = loss.item()
        
        for h in loss_r_feature_layers:
            h.close()      # module.register_forward_hook を解除
        
        with torch.no_grad():
            out = best_inputs.clone()

            # ImageNet 正規化を戻す
            mean = torch.tensor([0.5071, 0.4867, 0.4408], device=device).view(1, 3, 1, 1)
            std  = torch.tensor([0.2675, 0.2565, 0.2761], device=device).view(1, 3, 1, 1)
            out = out * std + mean
            out = torch.clamp(out, 0.0, 1.0)
            out = (out * 255.0).byte()
            out = out.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, 3)
        
        return out

    def build_real_features(self):
        """
        DeepInversionの画像生成（mmd損失）に使用する実画像の特徴を計算し保存する．
        保存先は self.real_features 
        """

        # --------------------------------------------------
        # １．対象とするクラスの決定
        # --------------------------------------------------
        # 保存する特徴量の数（クラスごと）
        n_per_class = self.num_features_per_class

        # 保持対象とするクラス（今回のタスクで学習したクラス）
        target_classes = list(range(self._pre_known_classes, self._total_classes))

        # --------------------------------------------------
        # ２．データローダーの作成
        # --------------------------------------------------
        real_dataset = self.data_manager.get_dataset(
            indices=target_classes,
            source="train",
            mode="test",
        )
        real_loader = DataLoader(
            real_dataset,
            batch_size=128,
            shuffle=True,
            num_workers=4,
        )

        # --------------------------------------------------
        # ３．モデルを評価モードに変更
        # --------------------------------------------------
        self._network.eval()

        # --------------------------------------------------
        # ４．特徴量を抽出
        # --------------------------------------------------
        with torch.no_grad():
            for idx, images, labels in real_loader:
                images = images.to(self._device)
                labels = labels.to(self._device)

                outputs = self._network(images)
                feats = outputs["features"]
                feats = feats.view(feats.size(0), -1)

                # クラスごとに n_per_class 個まで集める
                for f, y in zip(feats.cpu(), labels.cpu().tolist()):
                    if y in self.real_features and len(self.real_features[y]) < n_per_class:
                        self.real_features[y].append(f)

                # すべてのクラスで目標数に達したらループ終了
                if all(len(self.real_features[c]) >= n_per_class for c in target_classes):
                    break
        
        # --------------------------------------------------
        # ５．list を tensor に変換
        # --------------------------------------------------
        some_class = next(iter(target_classes))
        if len(self.real_features[some_class]) > 0:
            feat_dim = self.real_features[some_class][0].numel()
        else:
            feat_dim = 0
        
        for c in target_classes:
            if len(self.real_features[c]) > 0:
                self.real_features[c] = torch.stack(self.real_features[c], dim=0)
            else:
                self.real_features[c] = torch.empty(0, feat_dim)

    def _save_di_images(self, checkpoint_dir: str):
        """
        checkpoint_dir:
            BaseLearner.save_checkpoint と同じディレクトリパス。
            その直下に di_task{cur_task}/ を作り、
            - cls{c}_idx{n}.png
            - di_task{cur_task}.pth
            を保存する。
        """
        if getattr(self, "_data_memory", None) is None:
            return
        if isinstance(self._data_memory, np.ndarray) and self._data_memory.size == 0:
            return

        images = self._data_memory          # (N, H, W, 3), uint8 を想定
        labels = self._targets_memory       # (N,)

        save_dir = os.path.join(checkpoint_dir, f"di_task{self._cur_task}")
        os.makedirs(save_dir, exist_ok=True)

        N = images.shape[0]

        # ---------- まず PNG として 1 枚ずつ保存 ----------
        for i in range(N):
            img_arr = images[i]
            cls = int(labels[i])

            img = Image.fromarray(img_arr.astype(np.uint8))
            filename = f"cls{cls}_idx{i:05d}.png"
            path = os.path.join(save_dir, filename)
            img.save(path)

        # ---------- まとめて .pth にも保存 ----------
        # (N, H, W, 3) uint8 → (N, 3, H, W) uint8 の Tensor にして保存
        imgs_tensor = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()  # uint8
        labels_tensor = torch.from_numpy(labels.astype(np.int64))

        save_obj = {
            "images": imgs_tensor,   # shape: (N, 3, H, W), dtype: uint8
            "labels": labels_tensor, # shape: (N,)
            "task": int(self._cur_task),
        }

        pth_path = os.path.join(save_dir, f"di_task{self._cur_task}.pth")
        torch.save(save_obj, pth_path)





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

