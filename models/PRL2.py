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


class PRL2(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # backbone model の獲得
        self._network = IncrementalNet(args, False)
        
        # プロトタイプの初期化
        self._protos = {}

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
        
        # 損失関数
        self.pes_loss_func = PES_Loss()
        self.old_ae = None
    

    def after_task(self):
        self._known_classes = self._total_classes
        self._old_network = self._network.copy().freeze()
        if hasattr(self._old_network,"module"):
            self.old_network_module_ptr = self._old_network.module
        else:
            self.old_network_module_ptr = self._old_network
        
        self.save_checkpoint("checkpoint/{}/{}/{}/{}/{}/{}_{}_{}_{}_{}/".format(
            self.args["model_name"],
            self.args["log_name"],
            self.args["dataset"],
            self.args["init_cls"],
            self.args["increment"],
            self.args["lambda_fkd"], self.args["lambda_proto"], self.args["lambda_pes"], self.args["lambda_pgru"], self.args["lambda_unl"],)
        )

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
        self.forget_classes += [cls for cls in self.forget_list[self._cur_task]]
        logging.info(
            "forget classes on task{}: {}".format(self._cur_task, self.forget_classes))

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

        # 現在タスクの訓練用データセットを作成
        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source='train',
                                                 mode='train', appendent=self._get_memory())
        # 訓練用データローダーを作成
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=self.args["num_workers"], pin_memory=True)
        
        # テスト用データセットを作成
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test')
        
        # テスト用データローダーを作成
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=self.args["num_workers"])

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
        if self._cur_task in []:
            self._network.load_state_dict(torch.load("checkpoint/{}/{}/{}/{}/phase{}.pkl".format(self.args["model_name"],self.args["dataset"],self.args["init_cls"],self.args["increment"],self._cur_task))["model_state_dict"])
            resume = True
            logging.info('!!!resume!!!')
        
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
            losses_new, losses_fkd, losses_proto, losses_pes, losses_pkd, losses_unl = 0., 0., 0., 0., 0., 0.
            correct, total = 0, 0

            # 1エポック分の学習を実行
            for i, (_, inputs, targets) in enumerate(train_loader):

                # logging.info("targets: ", targets)

                # 入力とラベルを device 上に配置
                inputs, targets = inputs.to(
                    self._device, non_blocking=True), targets.to(self._device, non_blocking=True)
                
                # class augmentaion の回転処理（おそらく？）
                inputs = torch.stack([torch.rot90(inputs, k, (2, 3)) for k in range(4)], 1)
                inputs = inputs.view(-1, 3, self.size, self.size)

                # class augmentation に合わせてラベルを修正
                aug_targets = torch.stack([targets * 4 + k for k in range(4)], 1).view(-1)
                
                # model にデータを入力 & 損失を計算
                logits, loss_new, loss_fkd, loss_proto, loss_pes, loss_pkd = self._compute_prl_loss(inputs, targets, aug_targets)
                loss = loss_new + loss_fkd + loss_proto + loss_pes + loss_pkd
                
                # パラメータ更新
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 記録の更新
                losses += loss.item()
                losses_new += loss_new.item()
                losses_fkd += loss_fkd.item()
                losses_proto += loss_proto.item()
                losses_pes += loss_pes.item()
                losses_pkd += loss_pkd.item()


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
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_iic {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_pkd {:.3f}, Train_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_pes/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_pkd/len(train_loader), train_acc)
            else:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Loss_new {:.3f}, Loss_iic {:.3f}, Loss_fkd {:.3f}, Loss_proto {:.3f}, Loss_pkd {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}'.format(
                    self._cur_task, epoch+1, self._epoch_num, losses/len(train_loader), losses_new/len(train_loader), losses_pes/len(train_loader), losses_fkd/len(train_loader), losses_proto/len(train_loader), losses_pkd/len(train_loader), train_acc, test_acc)
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
        pes_targets = torch.stack([targets for k in range(4)], 1).view(-1)
        
        # model に inputs を入力し特徴量を出力
        features = self._network_module_ptr.extract_vector(inputs)
        
        # 特徴量を fc層 に入力して logits を獲得
        logits = self._network_module_ptr.fc(features)["logits"]
        
        # =============================
        # 交差エントロピー損失を計算
        # =============================
        loss_clf = F.cross_entropy(logits/self.args["temp"], aug_targets)
        loss_new = loss_clf

        # =============================
        # ベースタスクの場合，交差エントロピー損失とPES損失だけ計算
        # =============================
        if self._cur_task == 0:
            loss_iic = self.args["lambda_pes"] * self.pes_loss_func(features, pes_targets)
            return logits, loss_new, torch.tensor(0.), torch.tensor(0.), loss_iic, torch.tensor(0.)
        
        # 過去モデルの特徴量を取り出す
        features_old = self.old_network_module_ptr.extract_vector(inputs)
        
        # =============================
        # L2損失による蒸留損失
        # =============================
        loss_fkd = self.args["lambda_fkd"] * torch.dist(features, features_old, 2)
        
        # =============================
        # PGRU損失（直交損失と整合損失）
        # =============================
        loss_pkd = self.args["lambda_pgru"] * self._contras_loss(features, features_old)

        # =============================
        # 旧（維持）クラスの prototype とミニバッチのサンプルの feature を混ぜて 交差エントロピー損失を計算
        # =============================
        # 擬似 feature ベクトルを追加するリスト
        proto_features = []
        
        # プロトタイプに対応したラベルを格納するリスト
        proto_targets = []

        # 旧タスクに存在するラベルのリスト
        # print("self.forget_classes: ", self.forget_classes)
        old_class_list = list(self._protos.keys())

        # バッチサイズ分だけサンプルを作成する
        for _ in range(features.shape[0]//4): # batch_size = feature.shape[0] // 4

            # ランダムでサンプルを1つ選択
            i = np.random.randint(0, features.shape[0])

            # 混ぜるプロトタイプをランダムに選択するためシャッフル
            np.random.shuffle(old_class_list)
            lam = np.random.beta(0.5, 0.5)
            if lam > 0.6:
                lam = lam * 0.6

            # サンプルの特徴とプロトタイプを mixup
            if np.random.random() >= 0.5:
                temp = (1 + lam) * self._protos[old_class_list[0]] - lam * features.detach().cpu().numpy()[i]
            else:
                temp = (1 - lam) * self._protos[old_class_list[0]] + lam * features.detach().cpu().numpy()[i]
            
            # 擬似サンプル（擬似特徴）をリストに格納
            proto_features.append(temp)

            # 擬似サンプルに対応するラベルを格納（ラベルはプロトタイプのラベル）
            proto_targets.append(old_class_list[0])

        proto_features = torch.from_numpy(np.asarray(proto_features)).float().to(self._device,non_blocking=True)
        proto_targets = torch.from_numpy(np.asarray(proto_targets)).to(self._device,non_blocking=True)
        
        proto_logits = self._network_module_ptr.fc(proto_features)["logits"]
        loss_proto = self.args["lambda_proto"] * F.cross_entropy(proto_logits/self.args["temp"], proto_targets*4)
        

        return logits, loss_new, loss_fkd, loss_proto, torch.tensor(0.), loss_pkd
        
    
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

        if hasattr(self, "_class_means"):
            # class_means がある場合（通常の NME）
            y_pred_nme, y_true_nme = self._eval_nme(self.test_loader, self._class_means)
        elif hasattr(self, "_protos") and len(self._protos) > 0:
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
    
    # def eval_task(self):
    #     y_pred, y_true = self._eval_cnn(self.test_loader)
    #     cnn_accy = self._evaluate(y_pred, y_true)

    #     # MU 用の評価
    #     forget_set = set(self.forget_classes)
    #     y_true = np.asarray(y_true)
    #     y_pred_top1 = y_pred[:, 0]

    #     mask_forget = np.isin(y_true, list(forget_set))
    #     mask_retain = ~mask_forget

    #     if mask_forget.any():
    #         acc_forget = (y_pred_top1[mask_forget] == y_true[mask_forget]).mean() * 100
    #     else:
    #         acc_forget = None

    #     if mask_retain.any():
    #         acc_retain = (y_pred_top1[mask_retain] == y_true[mask_retain]).mean() * 100
    #     else:
    #         acc_retain = None

    #     logging.info(f"MU eval - forget classes: {self.forget_classes}")
    #     logging.info(f"MU eval - forget acc: {acc_forget}, retain acc: {acc_retain}")

    #     if hasattr(self, '_class_means'):
    #         y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
    #         nme_accy = self._evaluate(y_pred, y_true)
    #     elif hasattr(self, '_protos'):
    #         protos = list(self._protos.values())
    #         y_pred, y_true = self._eval_nme(self.test_loader, protos/np.linalg.norm(protos,axis=1)[:,None])
    #         nme_accy = self._evaluate(y_pred, y_true)
    #     else:
    #         nme_accy = None

    #     return cnn_accy, nme_accy
    

    # def eval_task(self):
    #     y_pred, y_true = self._eval_cnn(self.test_loader)
    #     cnn_accy = self._evaluate(y_pred, y_true)

    #     if hasattr(self, '_class_means'):
    #         y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
    #         nme_accy = self._evaluate(y_pred, y_true)
    #     elif hasattr(self, '_protos'):
    #         protos = list(self._protos.values())
    #         y_pred, y_true = self._eval_nme(self.test_loader, protos/np.linalg.norm(protos,axis=1)[:,None])
    #         nme_accy = self._evaluate(y_pred, y_true)
    #     else:
    #         nme_accy = None

    #     return cnn_accy, nme_accy