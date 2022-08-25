import math

import torch
import pandas as pd
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
    prediction_top3 = []
    conf1_list = []
    conf3_list = []

    with torch.no_grad():
        for data, _, name in tqdm(data_load):
            data = data.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                # print(output)
                # a = [[0.7 for i in range(10)] for k in range(len(data))]
                # b = torch.log(torch.FloatTensor(a)).cuda()
                # output = output.ge(b)

            conf1, pred = torch.max(output, dim=1) # top 1 accuracy 사용시
            conf3, top3_pred = torch.topk(output, 3)  # top 3 accuracy 사용시

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
            prediction_top3 += (top3_pred.tolist())
            conf1_list += (conf1.tolist())
            conf3_list += (conf3.tolist())
    # # confidence 값 분석을 위한 데이터 프레임 생성
    conf_df = pd.DataFrame({
        'video': folder_name, 'top1': prediction, 'top1-confidence': conf1_list,
        'top3': prediction_top3, 'top3-confidence': conf3_list
    })

    target = folder_name[0].split('-')[1]
    csv_name = 'MoviNet_inference_confidence_' + target + '.csv'
    conf_df.to_csv(csv_name)

    del data
    # GPU memory delete
    torch.cuda.empty_cache()
    return prediction, prediction_top3, folder_name

def inference_user(model, path, target):
    Bs_Test = 16
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])

    hmdb51_test = AIHUB_INFERENCE(path, transform=transform_test)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    prediction, prediction_top3, video_list = test_user(model, test_loader)
    label = ['bite', 'eat', 'jump', 'lie', 'scratch', 'shake', 'sit', 'sniff', 'stand', 'turn', 'walk']

    total = 0
    acc = 0
    top3_acc = 0
    label_predict_count_dict = {key:0 for key in label}

    for i, pred_idx in enumerate(prediction):
        if i % 4 == 0: print()
        top3_label_list = [label[j] for j in prediction_top3[i]]
        print('{:<35s}:{:<10s}{:<35s}'.format(video_list[i], label[pred_idx], str(top3_label_list)), end='|')
        label_predict_count_dict[label[pred_idx]] += 1
        total += 1
        if(target == label[pred_idx]): acc += 1
        if(label.index(target) in prediction_top3[i]): top3_acc += 1
    print("\n\nlabel: ",target,", total number: ", total)
    # print("total: {:4}/{:4}".format(acc, total))
    print('top1-Accuracy:' + '{:5}'.format(acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100*acc/total) + '%)\n' +
          'top3-Accuracy:' + '{:5}'.format(top3_acc) + '/' +
          '{:5}'.format(total) + ' (' +
          '{:4.2f}'.format(100 * top3_acc / total) + '%)\n'
          )
    del model
    del hmdb51_test

    return label_predict_count_dict, acc