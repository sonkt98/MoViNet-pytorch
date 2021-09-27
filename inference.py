import math

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import torchvision.transforms as transforms
from load_dataset import AIHUB_INFERENCE

from movinets import MoViNet
from movinets.config import _C

import transforms as T

def test_none(model, data_load, n_clips=2, n_clip_frames=5):
    model.eval()

    folder_name = []
    prediction = []
    with torch.no_grad():
        for data, _, name in tqdm(data_load):
            data = data.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                # a = [[0.7 for i in range(10)] for k in range(len(data))]
                # b = torch.log(torch.FloatTensor(a)).cuda()
                # output = output.ge(b)
            _, pred = torch.max(output, dim=1)
            minimum, minimum2 = torch.min(output, dim=1)
            output[:,0]=minimum-0.1
            _2, pred2 = torch.max(output, dim=1)
            # print("\n", pred)
            pred = torch.IntTensor([pred2[i] if pred[i]==0 and value < math.log(0.7) else pred[i] for i, value in enumerate(_)]).cuda()
            pred = torch.IntTensor([3 if value < math.log(0.5) else pred[i] for i, value in enumerate(_)]).cuda()

            # print(math.log(0.9),_, pred)
            pred = pred.clamp(max=3)
            folder_name += name
            prediction += (pred.tolist())
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return prediction, folder_name

def inference_none(model, path, target):
    Bs_Test = 16
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])

    hmdb51_test = AIHUB_INFERENCE(path, transform=transform_test)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    prediction, video_list = test_none(model, test_loader)
    label = ['scratch', 'shake', 'turn', 'none']

    total = 0
    total_label = 0
    total_none = 0
    label_acc = 0
    none_acc = 0

    for i, pred_idx in enumerate(prediction):
        if i % 5 == 0: print()
        print('{:<30s}:{:<15s}'.format(video_list[i], label[pred_idx]), end='|')
        total += 1
        if(target == label[pred_idx]): label_acc += 1
        if not (target in label):
            if (pred_idx==3): none_acc+=1
            total_none+=1
        else:
            total_label+=1
    print("\n\nlabel: ",target,", total number: ", total)
    print('total Accuracy:' + '{:5}'.format(label_acc+none_acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100 * (label_acc+none_acc) / total) + '%)')
    print('label Accuracy:' + '{:5}'.format(label_acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100 * label_acc / total) + '%)')
    print('none Accuracy:' + '{:5}'.format(none_acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100 * none_acc / total) + '%)')
    del model
    del hmdb51_test
    
def test_user(model, data_load, n_clips=2, n_clip_frames=5):
    model.eval()

    folder_name = []
    prediction = []
    with torch.no_grad():
        for data, _, name in tqdm(data_load):
            data = data.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                # a = [[0.7 for i in range(10)] for k in range(len(data))]
                # b = torch.log(torch.FloatTensor(a)).cuda()
                # output = output.ge(b)
            _, pred = torch.max(output, dim=1)
            # minimum, minimum2 = torch.min(output, dim=1)
            # output[:,0]=minimum-0.1
            # _2, pred2 = torch.max(output, dim=1)
            # print("\n", pred)
            # pred = torch.IntTensor([pred2[i] if pred[i]==0 and value < math.log(0.7) else pred[i] for i, value in enumerate(_)]).cuda()
            # pred = torch.IntTensor([3 if value < math.log(0.5) else pred[i] for i, value in enumerate(_)]).cuda()

            # print(math.log(0.9),_, pred)
            # pred = pred.clamp(max=3)
            folder_name += name
            prediction += (pred.tolist())
    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return prediction, folder_name

def inference_user(model, path, target):
    Bs_Test = 16
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])

    hmdb51_test = AIHUB_INFERENCE(path, transform=transform_test)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    prediction, video_list = test_user(model, test_loader)
    label = ['bite', 'eat', 'jump', 'lie', 'scratch', 'shake', 'sit', 'sniff', 'stand', 'turn', 'walk']

    total = 0
    acc = 0

    for i, pred_idx in enumerate(prediction):
        if i % 5 == 0: print()
        print('{:<30s}:{:<15s}'.format(video_list[i], label[pred_idx]), end='|')
        total += 1
        if(target == label[pred_idx]): acc += 1
    print("\n\nlabel: ",target,", total number: ", total)
    # print("total: {:4}/{:4}".format(acc, total))
    print('Accuracy:' + '{:5}'.format(acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100*acc/total) + '%)\n')
    del model
    del hmdb51_test