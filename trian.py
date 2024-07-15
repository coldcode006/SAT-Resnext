import timeit
from datetime import datetime
import socket
import os
import glob
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

# from dataloaders.dataset import VideoDataset
# from network import C3D_model, R2Plus1D_model, R3D_model,resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 40  # Number of epochs for training
resume_epoch =0 # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 10  # Store a model every snapshot epochs
lr = 1e-4  # Learning rate

dataset = 'ucf101'  # Options: hmdb51 or ucf101

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

__file__ = 'work'
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id + 2))

modelName = 'New_SA_ResneXt101'
saveName = modelName + '-' + dataset


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    model = generate_model_resnext(101)

    model.layer1[0].downsample[0] = nn.Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    model.layer1[0].conv1 = nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)

    new_conv = nn.Conv3d(64, 256, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    new_conv_weight = nn.Parameter(new_conv.weight)

    new_conv1 = nn.Conv3d(64, 128, kernel_size=(1, 1, 1), stride=(1, 1, 1), bias=False)
    new_conv1_weight = nn.Parameter(new_conv1.weight)
    # 替换模型中指定卷积层的参数
    model.layer1[0].downsample[0].weight = new_conv_weight
    model.layer1[0].conv1.weight = new_conv1_weight
    checkpoint = torch.load("/home/featurize/work/resnext-101-64f-kinetics.pth")
    new_state_dict = {}
    ###加载预训练模型
    for key, value in checkpoint['state_dict'].items():
        key = key.split(".")
        delimiter = '.'
        key = key[1:]
        key = delimiter.join(key)
        new_state_dict[key] = value

    state_dict = model.state_dict()
    for k, v in new_state_dict.items():
        if k not in state_dict:
            continue
        state_dict[k] = v

    state_dict['fc.weight'] = state_dict['fc.weight'][0:101, :]
    state_dict['fc.bias'] = state_dict['fc.bias'][0:101]
    # print(state_dict.)

    model.load_state_dict(state_dict)
    train_params = model.parameters()


    criterion = FocalLoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50,
                                          gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    # 是否接着上一次训练
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        # optimizer.load_state_dict(checkpoint['opt_dict'])
    # 计算模型总参数
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)
    criterion.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)
    # 导入数据
    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=8, shuffle=True,
                                  num_workers=0)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=8, num_workers=0)


    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    # test_size = len(test_dataloader.dataset)
    # 开始训练

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects重设训练误差和正确率
            running_loss = 0.0
            running_corrects = 0.0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training

                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels.long())

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            scheduler.step()
            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                writer.add_scalar('train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('train_acc_epoch', epoch_acc, epoch)
                train_acc.append(epoch_acc)
                train_loss.append(epoch_loss)
            else:
                writer.add_scalar('val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                val_acc.append(epoch_acc)
                val_loss.append(epoch_loss)

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))
       if useTest and epoch % test_interval == (test_interval - 1):
        model.eval()
        start_time = timeit.default_timer()

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in tqdm(test_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels.long())

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size

        writer.add_scalar('C:/Users/Administrator/Desktop/pytorch-video-recognition-master/data/test_loss_epoch', epoch_loss, epoch)
        writer.add_scalar('C:/Users/Administrator/Desktop/pytorch-video-recognition-master/data/test_acc_epoch', epoch_acc, epoch)

        print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")


    writer.close()


if __name__ == "__main__":
    train_model()
