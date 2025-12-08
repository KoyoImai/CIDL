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
# [DI-NEW]
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

# [DI-NEW]
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



class BASELINE_DI(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # backbone model の獲得
        self._network = IncrementalNet(args, False)
        self._memory_data = []
        self._memory_targets = []
        
        # プロトタイプの初期化
        self._protos = {}

        # 忘却クラス関連
        self.forget_list = args["forget_cls"]   # タスクごとの忘却予定クラスリスト（外から与えられる）
        self.forget_classes = []                # これまでの全タスクで忘却済みのクラス（累積）
        self.cur_forget_classes = []            # このタスクで新たに忘却するクラス

        # 将来まで含めた忘却クラスのリスト
        self.all_forget_classes = sorted(
            {c for task in self.forget_list for c in task}
        )

        # 1 クラスあたり保存するサンプル数 n
        mem_per_cls = args.get("memory_per_class", None)
        if mem_per_cls is None:
            if len(self.all_forget_classes) > 0:
                mem_per_cls = self._memory_size // len(self.all_forget_classes)
            else:
                mem_per_cls = 0
        self.forget_memory_per_class = mem_per_cls

        # データセットのサイズを設定
        if "cifar" in self.args["dataset"]:
            self.size = 32
        elif "tiny" in self.args["dataset"]:
            self.size = 56
        elif "imagenet" in self.args["dataset"]:
            self.size = 224
        
        # 損失関数
        self.pes_loss_func = PES_Loss()
        self.old_ae = None

        # ===== ランダム教師モデルの追加 =====
        # 同じ構造のネットワークをランダム初期化したまま固定しておく
        self._teacher = IncrementalNet(args, False)
        self._teacher.to(self._device)
        self._teacher.eval()
        for p in self._teacher.parameters():
            p.requires_grad = False
        # ===============================

        # ===== DeepInversion用の初期化を追加 =====
        self.di_batch_size   = 400     
        self.di_iterations   = 100         # 最適化ステップ数
        self.di_lr           = 0.25
        self.di_r_feature    = 0.01         # BN loss の係数
        self.di_tv_l2        = 0.0001       # TV loss の係数
        self.di_tv_l1        = 0.0
        self.di_l2           = 0.00001      # 画像 L2 loss の係数
        self.main_loss_multiplier = 1.0



    def after_task(self):

        # これまでに学習したクラス数の更新
        self._known_classes = self._total_classes
        
        # 知識蒸留用の過去モデルを更新
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        
        # リプレイバッファの更新
        if self._memory_size > 0 and self.data_manager is not None:
            
            # 1 クラスあたり何サンプル保持するか
            if self._memory_per_class is not None:
                m = self._memory_per_class
            else:
                m = self._memory_size // self._known_classes

            logging.info(f"Update replay memory: m={m} per class")
            self.build_rehearsal_memory(self.data_manager, m)

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


    def incremental_train(self, data_manager):
        
        # data_managerの登録
        self.data_manager = data_manager
        self._cur_task += 1

        # 元ラベルの順序
        self._class_order = data_manager.get_class_order()

        # 2タスク目で AutoEncoder を作成
        if self._cur_task == 1:
            self.old_ae = AutoencoderSigmoid(code_dims=512)
            self.old_ae.to(self._device)

        # 現在タスクまでを含めた全てのクラス数を更新
        self._total_classes = self._known_classes + \
            data_manager.get_task_size(self._cur_task)
        
        # 忘却クラスの更新
        # 今タスクで新たに忘却するクラス
        self.cur_forget_classes = [cls for cls in self.forget_list[self._cur_task]]
        # これまでの累積忘却クラス
        self.forget_classes += self.cur_forget_classes

        logging.info(
            "forget classes on task{}: total={} (new={})".format(
                self._cur_task, self.forget_classes, self.cur_forget_classes
            )
        )

        # model の fc層 の出力次元数を変更
        self._network.update_fc(self._total_classes*4)
        self._network_module_ptr = self._network
        logging.info(
            "model: {}".format(self._network_module_ptr))
        logging.info(
            'Learning on {}-{}'.format(self._known_classes, self._total_classes))

        logging.info('All params: {}'.format(count_parameters(self._network)))
        logging.info('Trainable params: {}'.format(
            count_parameters(self._network, True)))

        # print("self._get_memory(): ", self._get_memory())

        # 現在タスクの訓練用データセットを作成
        # train_dataset = data_manager.get_dataset(np.arange(self._known_classes,
        #                                                    self._total_classes),
        #                                                    source='train',
        #                                                    mode='train',
        #                                                    appendent=self._get_memory())
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes,
                                                           self._total_classes),
                                                           source='train',
                                                           mode='train',
                                                           appendent=None)
        # 訓練用データローダーを作成
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=self.args["batch_size"],
                                       shuffle=True,
                                       num_workers=self.args["num_workers"],
                                       pin_memory=True)
        
        # テスト用データセットを作成
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes),
                                                source='test',
                                                mode='test')
        
        # テスト用データローダーを作成
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=self.args["batch_size"],
                                      shuffle=False,
                                      num_workers=self.args["num_workers"])

        # 複数gpuが使用可能ならDPを適用
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        
        # 学習を実行
        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module


    def _train(self, train_loader, test_loader):
        
        # デバッグ用かな？
        resume = False
        if self._cur_task in [0]:
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
            # assert False
        # assert False
        
        # model を device に配置
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module
        
        if not resume:
            # ベースタスクの場合
            if self._cur_task == 0:
                self._epoch_num = self.args["init_epochs"]
                optimizer = torch.optim.Adam(self._network.parameters(), lr=self.args["init_lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            
            # 追加タスクの場合
            else:
                trainable_list = nn.ModuleList([])
                trainable_list.append(self._network)
                trainable_list.append(self.old_ae)
                self._epoch_num = self.args["epochs"]
                logging.info('All params total: {}'.format(count_parameters(trainable_list)))
                optimizer = torch.optim.Adam(trainable_list.parameters(), lr=self.args["lr"], weight_decay=self.args["weight_decay"])
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args["step_size"], gamma=self.args["gamma"])
            self._train_function(train_loader, test_loader, optimizer, scheduler)
        self._build_protos()
            
        
    def _build_protos(self):
        prototype = {}
        with torch.no_grad():
            for class_idx in range(self._known_classes, self._total_classes):
                data, targets, idx_dataset = self.data_manager.get_dataset(np.arange(class_idx, class_idx+1), source='train',
                                                                    mode='test', ret_data=True)
                idx_loader = DataLoader(idx_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=4)
                vectors, _ = self._extract_vectors(idx_loader)
                class_mean = np.mean(vectors, axis=0)
                prototype[class_idx] = class_mean
            self._protos.update(prototype)


    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        
        # progress barの表示
        prog_bar = tqdm(range(self._epoch_num))
        
        # 決められた epoch 分だけ学習を実行
        for _, epoch in enumerate(prog_bar):
            
            # model を訓練モードに変更
            self._network.train()

            # 記録用変数の初期化
            losses = 0.
            losses_new, losses_fkd, losses_proto, losses_unl, losses_unl_mem = 0., 0., 0., 0., 0.
            correct, total = 0, 0

            # 1エポック分の学習を実行
            for i, (_, inputs, targets) in enumerate(train_loader):

                # -----------------------------
                # ① 現タスクのバッチを GPU に載せる
                # -----------------------------
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)

                # -----------------------------
                # ② リプレイメモリから忘却クラスをサンプリング
                #    （現在タスクの forget_classes のみ）
                # -----------------------------
                mem_inputs, mem_targets = self._sample_forget_memory_batch(
                    num_samples=self.args.get("forget_batch_size", 10),
                    target_classes=self.cur_forget_classes,
                )

                # 取得できたら concat して一つのバッチにする
                if mem_inputs is not None:
                    inputs = torch.cat([inputs, mem_inputs], dim=0)
                    targets = torch.cat([targets, mem_targets], dim=0)
                
                # -----------------------------
                # ③ rotation をかけて class augmentation
                #    （現バッチ + リプレイバッチの両方に適用）
                # -----------------------------
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)

                # class augmentation に合わせてラベルを修正
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                
                # -----------------------------
                # ④ 損失計算（_compute_prl_loss はそのまま再利用）
                #    - CE: retain_mask（忘却以外）
                #    - loss_unl: forget_mask（今タスクの忘却クラス）
                # -----------------------------
                logits, loss_new, loss_fkd, loss_proto, loss_unl, loss_unl_mem = self._compute_prl_loss(inputs, targets, aug_targets)
                loss = loss_new + loss_fkd + loss_proto + loss_unl + loss_unl_mem
                
                # パラメータ更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 記録の更新
                losses += loss.item()
                losses_new += loss_new.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_unl += loss_unl.item()
                losses_unl_mem += loss_unl_mem.item()

                # 正解率の計算
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aug_targets.expand_as(preds)).cpu().sum()
                total += len(aug_targets)
            
            # 学習率の調整
            scheduler.step()
            train_acc = np.around(tensor2numpy(
                correct)*100 / total, decimals=2)
            
            # 5 epoch 毎に精度や損失などを表示
            if epoch % 5 != 0:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Loss_mem {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_unl/len(train_loader), losses_unl_mem/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_unl {:.3f}, Loss_mem {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_unl/len(train_loader), losses_unl_mem/len(train_loader), train_acc, test_acc)
            prog_bar.set_description(info)
            logging.info(info)


    def _contras_loss(self, features, features_old):

        # 整合損失: AE（旧feature）と現在featureのMSE
        features_old = self.old_ae(features_old)
        loss_align = nn.MSELoss()(features, features_old)
        
        # 直交損失: AE(protos) と AE(旧feature) の cosine を下げる
        features_old_norm = F.normalize(features_old, p=2, dim=1)

        # 忘却クラスのプロトタイプの前準備
        valid_protos = [
            proto for cls_id, proto in self._protos.items()
            if cls_id not in self.forget_classes
        ]

        if len(valid_protos) == 0:
            # 直交させる相手がいないので align 項だけにする
            return loss_align

        protos = np.asarray(valid_protos)  # shape: [num_valid_classes, D]
        protos = torch.from_numpy(protos).float().to(self._device, non_blocking=True)
        protos = self.old_ae(protos)       # AutoEncoderで射影
        protos = F.normalize(protos, p=2, dim=1)

        # # プロトタイプの準備（not Machine Unlearning用）
        # protos = self._protos.values()             # 各クラスのプロトタイプ
        # protos = torch.from_numpy(np.asarray(list(protos))).float().to(self._device,non_blocking=True)
        # protos = self.old_ae(protos)               # AutoEncoderで射影
        # protos = F.normalize(protos, p=2, dim=1)

        similarity = torch.matmul(protos, features_old_norm.t())
        similarity = similarity.sum() / (similarity.shape[0]*similarity.shape[1])
        
        return loss_align + similarity


    # 損失計算
    def _compute_prl_loss(self, inputs, targets, aug_targets):
        """
        BASELINE_replay 版からの変更点:
          - CE は「忘却対象でない」サンプルだけで計算
          - KL→一様分布の unlearning loss は「忘却対象の」サンプルだけで計算
          - replay メモリのサンプルも train_loader に混ざっているので，
            ここでは単にラベルで判定するだけでよい。
        """
        cur_forget = set(getattr(self, "cur_forget_classes", []))
        lambda_unl = self.args.get("lambda_unl", 0.0)

        # 特徴 & logits
        features = self._network_module_ptr.extract_vector(inputs)      # [4B, D]
        logits = self._network_module_ptr.fc(features)["logits"]        # [4B, C*4]

        # ---------- base task: 普通に CE のみ ----------
        if self._cur_task == 0:
            loss_clf = torch.nn.functional.cross_entropy(
                logits / self.args["temp"], aug_targets
            )
            loss_new = loss_clf
            loss_fkd = torch.tensor(0., device=self._device)
            loss_proto = torch.tensor(0., device=self._device)
            loss_unl = torch.tensor(0., device=self._device)
            loss_unl_mem = torch.tensor(0., device=self._device)
            return logits, loss_new, loss_fkd, loss_proto, loss_unl, loss_unl_mem


        # ---------- 忘却 / 保持サンプルのマスクを作る ----------
        B = targets.shape[0]

        if len(cur_forget) > 0:
            # [B] : このバッチの元サンプルが忘却対象かどうか
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

        # ---------- incremental task ----------
        # まず Feature KD（保持クラス全体に対して）
        with torch.no_grad():
            features_old = self.old_network_module_ptr.extract_vector(inputs)
        
        if retain_mask.any():
            f_new = features[retain_mask]
            f_old = features_old[retain_mask]
            loss_fkd = self.args["lambda_fkd"] * torch.dist(f_new, f_old, 2)
        else:
            loss_fkd = torch.tensor(0., device=self._device)

        # Prototype rehearsal（保持クラスのみ、BASELINE_replay と同じ）
        proto_features = []
        proto_targets = []

        old_class_list = list(self._protos.keys())
        # 忘却クラスは prototype rehearsal には使わない
        old_class_list = [c for c in old_class_list if c not in self.forget_classes]

        retain_indices = torch.nonzero(retain_mask).view(-1).cpu().numpy()

        if len(old_class_list) == 0:
            loss_proto = torch.tensor(0., device=self._device)
        else:
            for _ in range(features.shape[0] // 4):
                i = np.random.choice(retain_indices)
                np.random.shuffle(old_class_list)
                lam = np.random.beta(0.5, 0.5)
                if lam > 0.6:
                    lam = lam * 0.6
                if np.random.random() >= 0.5:
                    temp = (1 + lam) * self._protos[old_class_list[0]]  - lam * features.detach().cpu().numpy()[i]
                else:
                    temp = (1 - lam) * self._protos[old_class_list[0]]  + lam * features.detach().cpu().numpy()[i]
                
                proto_features.append(temp)
                proto_targets.append(old_class_list[0])

            proto_features = torch.from_numpy(np.asarray(proto_features)).float().to(
                self._device, non_blocking=True
            )
            proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(
                self._device, non_blocking=True
            )

            proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
            loss_proto = self.args["lambda_proto"] * torch.nn.functional.cross_entropy(
                proto_logits / self.args["temp"], proto_targets * 4
            )

        

        # ---------- CE: 忘却対象でないサンプルのみ ----------
        if retain_mask.any():
            logits_retain = logits[retain_mask]
            targets_retain = aug_targets[retain_mask]
            loss_clf = torch.nn.functional.cross_entropy(
                logits_retain / self.args["temp"], targets_retain
            )
        else:
            loss_clf = torch.tensor(0., device=self._device)

        loss_new = loss_clf


        # ---------- Unlearning: 忘却クラスの特徴をそのクラスの prototype から引き離す ----------
        loss_unl_inputs = torch.tensor(0., device=self._device)
        if lambda_unl > 0 and forget_mask.any():
            # aug_targets は [4B] の "rotation付きクラスID" なので
            # 元のクラスIDに戻す（C*4 次元のうち 4 で割る）
            base_cls_ids = (aug_targets // 4).to(self._device)   # [4B]

            # 忘却対象サンプルだけ取り出す
            forget_features = features[forget_mask]              # [N_forget, D]
            forget_cls_ids = base_cls_ids[forget_mask]           # [N_forget]

            # 対応する prototype を 1 サンプルずつ並べる
            forget_cls_ids_np = forget_cls_ids.detach().cpu().numpy()
            proto_list = []
            for c in forget_cls_ids_np:
                c_int = int(c)
                assert c_int in self._protos, f"prototype for class {c_int} not found"
                proto_list.append(self._protos[c_int])           # np.array(D,)

            protos_forget = torch.from_numpy(
                np.stack(proto_list, axis=0)                     # [N_forget, D]
            ).float().to(self._device, non_blocking=True)

            # L2 normalize して cosine similarity を計算
            f_norm = F.normalize(forget_features, p=2, dim=1)    # [N_forget, D]
            p_norm = F.normalize(protos_forget,  p=2, dim=1)     # [N_forget, D]

            cos_sim = (f_norm * p_norm).sum(dim=1)               # [N_forget]

            # 平均 cosine を小さくする（= prototype から引き離す）
            loss_unl_inputs = lambda_unl * cos_sim.mean()

        loss_unl = loss_unl_inputs
        loss_unl_mem = torch.tensor(0., device=self._device)

        return logits, loss_new, loss_fkd, loss_proto, loss_unl, loss_unl_mem

    
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
    

    def eval_task(self):
        # -------------------------
        # CNN 評価（保持クラスだけ）
        # -------------------------
        y_pred, y_true = self._eval_cnn(self.test_loader)
        y_true = np.asarray(y_true)

        # 忘却クラス / 保持クラスの分割
        forget_set = set(getattr(self, "forget_classes", []))
        if len(forget_set) > 0:
            mask_forget = np.isin(y_true, list(forget_set))
        else:
            mask_forget = np.zeros_like(y_true, dtype=bool)
        mask_retain = ~mask_forget

        # 保持クラスのみで CNN の精度を計算
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

        # 忘却クラスの精度（CNN）
        if mask_forget.any():
            top1_pred_forget = y_pred[mask_forget][:, 0]
            true_forget = y_true[mask_forget]
            forget_acc_cnn = np.around(
                (top1_pred_forget == true_forget).sum() * 100.0 / len(true_forget),
                decimals=2,
            )
        else:
            forget_acc_cnn = None
        
        # 追加情報として dict に入れておく（trainer は今のままで OK）
        cnn_accy["forget_top1"] = forget_acc_cnn
        cnn_accy["num_retain_samples"] = int(mask_retain.sum())
        cnn_accy["num_forget_samples"] = int(mask_forget.sum())

        logging.info(
            f"MU eval (CNN) - retain samples: {mask_retain.sum()}, "
            f"forget samples: {mask_forget.sum()}"
        )
        logging.info(f"MU eval (CNN) - forget top1: {forget_acc_cnn}")

        ### CNN の調和平均
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
        
        # dict に保存しておく（必要なら trainer から拾える）
        cnn_accy["forget_err"] = forget_err_cnn
        cnn_accy["hmean"] = hmean_cnn

        logging.info(
            f"MU (CNN) retain_acc={retain_acc_cnn:.2f}, "
            f"forget_err={forget_err_cnn}, hmean={hmean_cnn}"
        )

        # -------------------------
        # NME 評価（保持クラスだけ）
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

            ### NME の調和平均
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
   

    def _sample_forget_memory_batch(self, num_samples, target_classes=None):
        """
        self._data_memory / self._targets_memory から
        target_classes（指定がなければ self.forget_classes）に属するサンプルだけを
        ランダムに num_samples 個取り出して 1 バッチ分の (inputs, targets) を返す。

        メモリ or 対象クラスが空のときは (None, None) を返す。
        """
        # メモリがまだ空なら何もしない
        if not hasattr(self, "_data_memory") or self._data_memory.size == 0:
            return None, None

        # 対象クラス集合の決定（指定がなければ累積 forget_classes）
        if target_classes is None:
            target_classes = self.forget_classes

        # 対象クラスが指定されていない or 空なら何もしない
        if target_classes is None or len(target_classes) == 0:
            return None, None

        # numpy の targets から target_classes に属する index を抜き出す
        mask = np.isin(self._targets_memory, np.array(target_classes))
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None, None

        num = min(num_samples, len(idxs))
        sampled = np.random.choice(idxs, size=num, replace=False)

        forget_data = self._data_memory[sampled]
        forget_targets = self._targets_memory[sampled]

        # DataManager 経由で Dataset を作って，普段と同じ transform をかける
        forget_dataset = self.data_manager.get_dataset(
            [],                           # 元データからは何も取らない
            source="train",
            mode="test",                  # ここは test / train どちらでもよいが，とりあえず test で固定
            appendent=(forget_data, forget_targets),
            setup_replay=False,
        )
        forget_loader = DataLoader(
            forget_dataset,
            batch_size=num,
            shuffle=False,
            num_workers=self.args["num_workers"],
            pin_memory=True,
        )

        # 1 バッチだけ取り出す
        _, inputs, targets = next(iter(forget_loader))
        inputs = inputs.to(self._device, non_blocking=True)
        targets = targets.to(self._device, non_blocking=True)
        return inputs, targets

    # ============================================================
    # 与えられたクラスラベル列に対して DeepInversion 画像を生成（ImageNet100）
    # ============================================================  
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
        inputs = torch.randn((B, 3, self.size, self.size), requires_grad=True, device='cuda')
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        targets = torch.from_numpy(class_labels).to(device=device, dtype=torch.long)
        criterion = nn.CrossEntropyLoss()


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
                        
                loss = main_loss_multiplier * loss + loss_aux

                if iteration % 10 == 0:
                    logging.info(f"------------iteration {iteration}----------")
                    logging.info(f"total loss: {loss.item():.4f}")
                    logging.info(f"loss_r_feature: {loss_r_feature.item():.4f}")
                    logging.info(f"main criterion: {criterion(logits, targets).item():.4f}")
                
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

    # リプレイバッファの構築
    def build_rehearsal_memory(self, data_manager, per_class):
        """
        DeepInversion で生成した画像をリプレイバッファに保存する版。
        クラスの選び方（将来 forget クラスのみ & まだ forget 済みでない）は
        BASELINE_replay6 と同じ。
        """
        # per_class 引数は無視し，固定 n を使う
        n = getattr(self, "forget_memory_per_class", per_class)
        if n <= 0:
            logging.info("[Replay2] Skip building replay memory (n <= 0).")
            return
        
        # すでに「忘却対象になった」クラス集合
        already_forgotten = set(getattr(self, "forget_classes", []))

        # これから先のタスクで忘却される予定 & まだ忘却されていない & すでに出現済み
        candidate_classes = [
            c for c in self.all_forget_classes
            if (c not in already_forgotten) and (c < self._total_classes)
        ]

        # すでにリプレイバッファに入っているクラス
        if hasattr(self, "_targets_memory") and self._targets_memory is not None and len(self._targets_memory) > 0:
            existing_classes = set(self._targets_memory.tolist())
        else:
            existing_classes = set()

        # 「バッファにまだ一度も入っていない」クラスだけを今回の追加対象にする
        target_classes = [c for c in candidate_classes if c not in existing_classes]

        logging.info(
            f"[DI] Building replay memory with DeepInversion: "
            f"n={n} per class for target_classes={target_classes}, "
            f"already_forgotten={sorted(already_forgotten)}"
        )

        if len(target_classes) == 0:
            logging.info(
                f"[DI] No NEW target classes for replay memory. "
                f"Keep existing buffer as-is. candidate={candidate_classes}, existing={sorted(existing_classes)}"
            )
            return

        all_exemplars = []
        all_labels = []

        # ================== ここから DI 画像一括生成部分 ==================  #
        # DeepInversionによってに画像を生成
        # クラスごとに分けて生成せず，必要な画像数を一括で生成
        # （gpuなどの問題で一括生成が難しい場合は複数回に分けて生成する）

        
        # 例: target_classes = [0,1,5], n = 3 のとき
        # labels_all = [0,0,0, 1,1,1, 5,5,5]
        labels_all = np.repeat(np.array(target_classes, dtype=np.int64), n)
        total = labels_all.shape[0]

        # labels_allを並び替え
        perm = np.random.permutation(total)
        labels_all = labels_all[perm]

        # 1 回の DeepInversion 最適化で扱う枚数
        # （__init__ で self.di_batch_size を args から読んでおく想定）
        max_batch = getattr(self, "di_batch_size", 64)

        start = 0
        while start < total:
            end = min(start + max_batch, total)
            batch_labels = labels_all[start:end]            # np.ndarray (B,)

            # DeepInversion によって batch_labels に対応する画像を生成
            # elif "imagenet" in self.args["dataset"]:
            if self.args["dataset"] in ["imagenet100"]:
                di_imgs = self._generate_di_images_for_labels(batch_labels)
            elif self.args["dataset"] in ["cifar100"]:
                assert False
            # di_imgs: (B, H, W, 3) の uint8 を想定

            all_exemplars.append(di_imgs)
            all_labels.append(batch_labels)

            start = end

        if len(all_exemplars) == 0:
            logging.info("[DI] No DI images were generated for new target classes.")
            return

        new_data = np.concatenate(all_exemplars, axis=0)
        new_labels = np.concatenate(all_labels, axis=0)

        # 既存のメモリがある場合は後ろに append、無ければ新規作成
        if hasattr(self, "_data_memory") and self._data_memory is not None and len(self._data_memory) > 0:
            self._data_memory = np.concatenate([self._data_memory, new_data], axis=0)
            self._targets_memory = np.concatenate([self._targets_memory, new_labels], axis=0)
        else:
            self._data_memory = new_data
            self._targets_memory = new_labels

        logging.info(
            f"[DI] Added {len(new_labels)} DI exemplars "
            f"for new target classes {target_classes}. "
            f"Total buffer size = {len(self._targets_memory)}"
        )

    # ============================================================
    # DeepInversion で生成したリプレイ画像を保存 (.png + .pth)
    # ============================================================
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



