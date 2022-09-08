import math
import os
import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.optim import lr_scheduler
from torch.utils.data import random_split, DataLoader
import torch
from tqdm import tqdm

import transforms as T
from inference import inference_user, inference_none
from load_dataset import HMDB51, AIHUB, VideoAIHUB, AIHUB_INFERENCE
from movinets import MoViNet
from movinets.config import _C
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# torch.manual_seed(97)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr) * self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (
                        1 + math.cos(math.pi * (self.T_cur - self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch

        self.eta_max = self.base_eta_max * (self.gamma ** self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    # model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, _, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        out = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(out, target)
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())
        del data
        torch.cuda.empty_cache()

def evaluate(model, data_load, loss_val):
    model.eval()
    # model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            csamp += pred.eq(target).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')
    del data
    torch.cuda.empty_cache()

def train_iter_stream(model, optimz, data_load, loss_val, n_clips=2, n_clip_frames=5):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    # clean the buffer of activations
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    for i, (data, _, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        l_batch = 0
        # backward pass for each clip
        for j in range(n_clips):
            output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
            _, pred = torch.max(output, dim=1)
            # nSamples = [1, 10, 15, 20, 20, 20, 20, 20, 20, 30]
            # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            # normedWeights = torch.FloatTensor(normedWeights).cuda()
            class_weight = torch.tensor([1/131, 1/69, 1/86, 1/378, 1/27, 1/21, 1/207, 1/454, 1/95, 1/46, 1/791])
            loss = F.nll_loss(output, target) / n_clips
            loss.backward()
        l_batch += loss.item() * n_clips
        optimz.step()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)
        del data
        torch.cuda.empty_cache()

def evaluate_stream(model, data_load, loss_val, n_clips=2, n_clip_frames=5):
    model.eval()
    samples = len(data_load.dataset)
    csamp1 = 0
    csamp2 = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                loss = F.nll_loss(output, target)
            tloss += loss.item() # top1기준 loss

            _, pred = torch.max(output, dim=1) # top 1 accuracy 사용시
            csamp1 += pred.eq(target).sum() # top 1 accuracy 사용시 pred와 target이 같은 correct sample의 수

            _, top3_pred = torch.topk(output, 3)  # top 3 accuracy 사용시
            for top3, targ in zip(top3_pred.tolist(), target.tolist()): # top 3 accuracy 사용시 pred에 target이 포함되는 correct sample의 수
                if targ in top3:
                    csamp2 += 1

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    top1_acc = 100.0 * csamp1 / samples
    top3_acc = 100.0 * csamp2 / samples
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  top1-Accuracy:' + '{:5}'.format(csamp1) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(top1_acc) + '%)' +
          '  top3-Accuracy:' + '{:5}'.format(csamp2) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(top3_acc) + '%)\n'
          )
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return top1_acc, 0, 0

def evaluate_none(model, data_load, loss_val, n_clips=2, n_clip_frames=5):
    model.eval()
    # model.cuda()
    samples = len(data_load.dataset)
    labels = 0
    nones = 0
    csamp_label = 0
    csamp_none = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in tqdm(data_load):
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                # print(torch.max(F.softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)))
                loss = F.nll_loss(output, target)
                # a = [[0.7 for i in range(10)] for k in range(len(data))]
                # b = torch.log(torch.FloatTensor(a)).cuda()
                # output = output.ge(b)
            # print(torch.max(F.softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1), dim=1))
            _, pred = torch.max(output, dim=1)
            pred = torch.IntTensor([3 if value == False else pred[i] for i, value in enumerate(_)]).cuda()
            pred = pred.clamp(max=3)
            a = [2 for i in range(len(target))]
            b = torch.FloatTensor(a).cuda()
            tloss += loss.item()
            compare = target.le(b)
            labels += compare.sum()
            nones += (~compare).sum()
            compare_label = pred.eq(target)
            compare_none = pred.gt(b)
            csamp_label += (compare & compare_label).sum()
            csamp_none += (~compare & compare_none).sum()
            # print(compare, compare_label, compare_none)
            # print(target, pred, loss.item())

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    csamp = csamp_label+csamp_none
    acc = 100.0 * csamp / samples
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(acc) + '%)\n')
    acc_label = 100.0 * csamp_label / labels
    acc_none = 100.0 * csamp_none / nones
    print('Label Accuracy: ' + '{:5}'.format(csamp_label) + '/' +
          '{:5}'.format(labels) + ' (' +
          '{:4.2f}'.format(acc_label) + '%)\n' +
          '  None Accuracy:' + '{:5}'.format(csamp_none) + '/' +
          '{:5}'.format(nones) + ' (' +
          '{:4.2f}'.format(acc_none) + '%)\n')
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return acc, acc_label, acc_none

def train():
    num_frames = 10  # 16
    clip_steps = 1
    Bs_Train = 8
    Bs_Test = 8
    transform = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((250, 250)), # 원래 200
        # T.RandomHorizontalFlip(),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        # T.RandomVerticalFlip(),
        # T.RandomPerspective(p=0.8),
        T.RandomCrop((224, 224))
    ])
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((250, 250)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((224, 224))
    ])

    # hmdb51_train = AIHUB('/home/petpeotalk/AIHUB/210915-userdataset-11/train', transform=transform)
    # hmdb51_train = AIHUB('/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/userdata/210915-userdataset-11/train', transform=transform) # 결과 재현용 데이터
    hmdb51_train = AIHUB('/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/labeled_data/20220718_action/original_dataset/train_7_original', transform=transform)
    #
    # hmdb51_test = AIHUB('/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/userdata/210915-userdataset-11/valid', transform=transform_test) # 결과 재현용 데이터
    hmdb51_test = AIHUB('/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/labeled_data/20220718_action/original_dataset/test_3_original', transform=transform_test)

    train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    N_EPOCHS = 100

    # MoviNetA0, ~ A5
    model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True, num_classes=11, conv_type="2plus1d")
    # model = torch.load('weights/modelA5_statedict_v3') # 이 부분을 통해서 model.pt에 이어서 전이학습하는듯
    start_time = time.time()
    #
    trloss_val, tsloss_val = [], []
    optimz = optim.Adam(model.parameters(), lr=0.00001)
    # scheduler = CosineAnnealingWarmUpRestarts(optimz, T_0=30, T_mult=2, eta_max=0.0001,  T_up=2, gamma=0.5)
    # scheduler = CosineAnnealingLR(optimz, T_max=30, eta_min=0.000001)
    scheduler = MultiStepLR(optimz, milestones=[20, 45], gamma=0.1) # 30, 50
    model.classifier[3] = torch.nn.Conv3d(2048, 11, (1, 1, 1))
    # print(model.classifier[3])

    model = model.cuda()

    #######################  load_model  ###########################
    # checkpoint = torch.load("/home/petpeotalk/MoViNet-pytorch/best_model_A2_user_0.0005.pt")
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(model.classifier[3])
    # optimz.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    # scheduler.load_state_dict(checkpoint["scheduler"])
    ################################################################
    writer = SummaryWriter()
    best_acc = 0

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {pytorch_total_params}")

    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_iter_stream(model, optimz, train_loader, trloss_val)
        acc, acc_label, acc_none = evaluate_stream(model, test_loader, tsloss_val)
        scheduler.step()
        writer.add_scalars("Loss", {'train': np.mean(np.array(trloss_val)),
                          'valid': np.mean(np.array(tsloss_val))}, epoch)
        writer.add_scalars("Accuracy", {'Total': acc, 'Label': acc_label, 'None': acc_none}, epoch) # tensorboard에 test accuracy
        if acc > best_acc: # test set에 대한 정확도 갱신 시, weight 저장 가장 좋은 성능 낸 경우가 weight으로 저장 됌.
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimz.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'best_model.pt')
            best_acc = acc
            torch.save(model, '/home/petpeotalk/MoViNet-pytorch/custom_weight/Crop_vs_Original/model_original_data_A2_00001_100_224_crop_multistep.pt')
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


def test_model(model, test_loader):
    # MoviNetA0, ~ A5
    start_time = time.time()

    tsloss_val = []
    evaluate_stream(model, test_loader, tsloss_val)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


if __name__ == '__main__':
    model_name = "modelA5"

    #########################################################################################################
    # model = MoViNet(_C.MODEL.MoViNetA2, causal=True, pretrained=True, num_classes=11, conv_type="2plus1d")
    # checkpoint = torch.load("best_model_A2_user_0.0005.pt")
    # model.classifier[3] = torch.nn.Conv3d(2048, 11, (1, 1, 1))  # 11이 class 개수
    # model = model.cuda()
    # model.load_state_dict(checkpoint['model_state_dict'])

    ################test###################################################################################
    # Bs_Test = 16
    # transform_test = transforms.Compose([
    #     T.ToFloatTensorInZeroOne(),
    #     T.Resize((224, 224))
    #         # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    #     # T.CenterCrop((224, 224))
    # ])
    # #
    # model = torch.load('/home/petpeotalk/MoViNet-pytorch/model_original_data_A2_0001_100_224_crop_customcosineAnnealingWarm.pt')
    # #
    # hmdb51_test = AIHUB('/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/labeled_data/20220718_action/original_dataset/test_3_original', transform=transform_test)
    # test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)
    #######################################################################################################

    train()

    # test_model(model, test_loader)
    # evaluate_none(model, test_loader, [])


###################################inference#############################################
    # model = torch.load('/home/petpeotalk/MoViNet-pytorch/model_original_data_A2_0001_100_224_crop_cosine.pt')
    # path = '/home/petpeotalk/MoviNet_real/dogibogi-ai-research/projects/action_recognition/MoViNet-pytorch/labeled_data/20220718_action/original_dataset/test_3_original'
    # #
    # #
    # label_list = ['bite', 'eat', 'jump', 'lie', 'scratch', 'shake', 'sit', 'sniff', 'stand', 'turn', 'walk']
    # label_total_predict_count_dict = {key: 0 for key in label_list}
    # label_correct_predict_count_dict = {key: 0 for key in label_list}
    # for i in sorted(os.listdir(path)):
    #     # inference_none(model, os.path.join(path, i), i)
    #     label_predict_count_dict, TP = inference_user(model, os.path.join(path, i), i)
    #     label_correct_predict_count_dict[i] +=TP
    #     for key, value in label_predict_count_dict.items():
    #         label_total_predict_count_dict[key]+=value
    #
    # for i in label_list:
    #     if label_total_predict_count_dict[i]==0:
    #         print(i+'로 예측한 data가 존재하지 않음\n')
    #         continue
    #
    #     print(i+' precision: ' + '{:4.2f}'.format(100*label_correct_predict_count_dict[i]/label_total_predict_count_dict[i]) + '%\n')
