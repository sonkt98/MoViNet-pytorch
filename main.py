import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from load_dataset import HMDB51, AIHUB, VideoAIHUB, AIHUB_TEST
from movinets import MoViNet
from movinets.config import _C
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# torch.manual_seed(97)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

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
    # model.cuda()
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
            # nSamples = [755, 1010, 852]
            # normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
            # normedWeights = torch.FloatTensor(normedWeights).cuda()
            loss = F.nll_loss(output, target) / n_clips
            loss.backward()
        l_batch += loss.item() * n_clips
        optimz.step()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 100 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)
        del data
        torch.cuda.empty_cache()

def evaluate_stream(model, data_load, loss_val, n_clips=2, n_clip_frames=5):
    model.eval()
    # model.cuda()
    samples = len(data_load.dataset)
    labels = 0
    nones = 0
    csamp_label = 0
    csamp_none = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                # print(torch.max(F.softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)))
                loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            a = [2 for i in range(4)]
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
            print(target, pred, loss.item())

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
    return acc

def train():
    num_frames = 10  # 16
    clip_steps = 1
    Bs_Train = 4
    Bs_Test = 4
    transform = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        T.RandomHorizontalFlip(),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        # T.RandomVerticalFlip(),
        # T.RandomPerspective(),
        T.RandomCrop((172, 172))])
    transform_test = transforms.Compose([
        T.ToFloatTensorInZeroOne(),
        T.Resize((200, 200)),
        # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        T.CenterCrop((172, 172))])
    #
    # hmdb51_train = HMDB51('aihub_data/', 'test_train_splits/', num_frames, frame_rate=5,
    #                                            step_between_clips=clip_steps, fold=1, train=True,
    #                                            transform=transform, num_workers=2)
    #
    # hmdb51_test = HMDB51('aihub_data/', 'test_train_splits/', num_frames, frame_rate=5,
    #                                           step_between_clips=clip_steps, fold=1, train=False,
    #                                           transform=transform_test, num_workers=2)

    # hmdb51_train = VideoAIHUB('aihub_data/', num_frames, frame_rate=5,
    #                                            step_between_clips=clip_steps, fold=1,
    #                                            transform=transform, num_workers=2)
    #
    # hmdb51_test = VideoAIHUB('Validation/', num_frames, frame_rate=5,
    #                                           step_between_clips=clip_steps, fold=1,
    #                                           transform=transform_test, num_workers=2)

    hmdb51_train = AIHUB('/home/petpeotalk/AIHUB/hackerton-full-dataset-ver2/train', transform=transform)
    #
    hmdb51_test = AIHUB('/home/petpeotalk/AIHUB/hackerton-full-dataset-ver2/test', transform=transform_test)

    train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)

    N_EPOCHS = 50

    # MoviNetA0, ~ A5
    model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True, num_classes=10, conv_type="2plus1d")
    start_time = time.time()

    trloss_val, tsloss_val = [], []
    optimz = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = MultiStepLR(optimz, milestones=[10, 30], gamma=0.1)
    model.classifier[3] = torch.nn.Conv3d(2048, 10, (1, 1, 1))
    # print(model.classifier[3])
    model = model.cuda()

    # model = torch.load('model.pt')
    #######################  load_model  ###########################
    # checkpoint = torch.load("best_model.pt")
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
        acc = evaluate_stream(model, test_loader, tsloss_val)
        scheduler.step()
        writer.add_scalars("Loss", {'train': np.mean(np.array(trloss_val)),
                          'valid': np.mean(np.array(tsloss_val))}, epoch)
        writer.add_scalars("Accuracy", {'valid': acc}, epoch)
        if acc > best_acc:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimz.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, 'best_model.pt')
            best_acc = acc
            torch.save(model, 'model.pt')
    writer.flush()
    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')


def test_model():
    Bs_Test = 4
    # MoviNetA0, ~ A5
    start_time = time.time()

    # model.classifier[3] = torch.nn.Conv3d(2048, 51, (1, 1, 1))
    for i in range(1):
        tsloss_val = []
        transform_test = transforms.Compose([
            T.ToFloatTensorInZeroOne(),
            T.Resize((200, 200)),
            # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
            T.CenterCrop((172, 172))])
        model = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True, num_classes=10, conv_type="2plus1d")
        checkpoint = torch.load("best_model.pt")
        model.classifier[3] = torch.nn.Conv3d(2048, 10, (1, 1, 1))
        model = model.cuda()
        model.load_state_dict(checkpoint['model_state_dict'])

        # model = torch.load('model.pt')
        print(model.classifier)

        hmdb51_test = AIHUB('/home/petpeotalk/AIHUB/hackerton-full-dataset-ver2/test', transform=transform_test)
        test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)
        evaluate_stream(model, test_loader, tsloss_val)

        del model
        del hmdb51_test

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == '__main__':
    model_name = "modelA0"

    # test_model()
    train()
