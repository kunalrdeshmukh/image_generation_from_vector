# Import system libraries
import argparse
import math
import os
import random
import sys
import time


# import 3rd party libraries
import csv
import matplotlib.pyplot as plt
import skimage
import torch
import torchvision
from PIL import Image
from torch import nn , optim, tensor
from torch.autograd import Variable
from torchvision import datasets, transforms


# Import user-defined libraries
from dataset import DatasetFromFolder
from network import Net


irange = range


parser = argparse.ArgumentParser(description='Image Compression')
parser.add_argument('--batchSize', type=int, default=64, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=2, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='Learning Rate. Default=0.001')
parser.add_argument('--beta', type=float, default=0.99, help='beta1 for adam. default=0.99')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--encoder_net', type=str, default='', help='Path to pre-trained encoder net. Default=3')
parser.add_argument('--decoder_net', type=str, default='', help='path to pre-trained deocder net. Default=3')
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--test_path', default='./Dataset/', help='path of test images')
parser.add_argument('--channels', type=int, default=3, help='number of channels in an image. Default=3')
parser.add_argument('--dataset', type=str, default='folder', help='dataset to be used for training and validation. Default=STL10')
parser.add_argument('--data_path', type=str, default='./Dataset/', help='path to images. Default=CLIC')
parser.add_argument('--image_size', type=int, default=200, help='path to images. Default=90')
parser.add_argument('--loss_function', type=int, default=0, help='Loss function. Default=0')
parser.add_argument('--use_GPU', type=int, default=-1, help='0 for GPU, 1 for CPU . Default=AUTO')
parser.add_argument('--mode', type=str, default='both', help='train / test / both . Default=both')


opt = parser.parse_args()

print (opt)

CUDA = torch.cuda.is_available()
LOG_INTERVAL = 5

if opt.use_GPU == 1:
    CUDA = False


def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    """Save a given Tensor into an image file.
    Args:
        tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
            saves the tensor as a grid of images by calling ``make_grid``.
        **kwargs: Other arguments are documented in ``make_grid``.
    """
    torchvision.utils.save_image(tensor,filename)



def img_transform_train(crop_size): 
    return transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
        ])

def img_transform_test(crop_size):
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
    return transforms.Compose([
        transforms.normalize()
    ])

"""### Loss function"""

def loss_function(final_img,orig_img):
    return nn.MSELoss()(final_img,orig_img)

"""### define Train and Test methods"""

def train(epoch,model,train_loader):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    train_loss = 0

    for batch_idx, data in enumerate(train_loader):
        input_data = [0.0]*len(train_loader)
        input_data[batch_idx] = 1.0
        input_tensor = tensor(input_data)
        input_tensor = Variable(input_tensor)
        optimizer.zero_grad()

        if CUDA:
          data = data.cuda()
          input_tensor = input_tensor.cuda()
          fake_image = model(input_tensor)
        else :
          data = data.cpu()
          input_tensor = input_tensor.cpu()
          fake_image = model(input_tensor)
        loss = loss_function(data,fake_image)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)



def test(model,no_of_images):
    t1 = time.time()
    model.load_state_dict(torch.load('./net.pth'))

    model.eval()

    test_loss = 0
    torch.no_grad()
    for i in range(no_of_images):
        input_data = [0.0]*no_of_images
        input_data[i] = 1.0
        input_tensor = tensor(input_data)
        input_tensor = Variable(input_tensor)

        if CUDA:
            input_tensor = input_tensor.cuda()
            fake_img = model(input_tensor)
        else:
            input_tensor = input_tensor.cpu()
            fake_img = model(input_tensor)
        
        torchvision.utils.save_image(fake_img.cpu(),'filename_'+str(i)+'.png')


def main():
    """### Parameters"""
    CHANNELS = opt.channels
    HEIGHT = opt.image_size
    EPOCHS = opt.nEpochs

    """### Load Dataset"""
    trainset = DatasetFromFolder(opt.data_path+'train/', input_transform=img_transform_train((opt.image_size,opt.image_size)),channels = opt.channels)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batchSize,
                                        shuffle=True, num_workers=opt.threads)
    
    info = {}

    info['channels'] = CHANNELS
    info['output_size'] = HEIGHT
    info['input_size'] = len(train_loader)

    if CUDA:
        model = Net(info).cuda()
    else :
        model = Net(info).cpu()

    print("GPU available : "+str(CUDA))


    if opt.mode.upper() == 'TEST':  
        print(" Mode selected : test")
        print("run model for Images in test folder.")
        test(model,test_data_loader)
        sys.exit()
    
    """### Program Execution"""
    tr_loss = []
    vl_loss = []
    for epoch in range(1, EPOCHS+1):
        t1 = time.time()
        print("Epoch : "+str(epoch))
        t_loss = train(epoch,model,train_loader)

        print("Time required for "+str(epoch)+"th epoch: "+str(time.time() - t1))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), '%s/net_epoch_%d.pth' % (opt.outf, epoch))

    
    print("Plot and save train and validation loss curves")
        # Plot train, test loss curves
    plt.plot(range(EPOCHS),tr_loss , 'r--',label='Training Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.savefig('loss_'+str(time.time())+'.png')

    with open('loss_values.csv', "w") as output:
        writer = csv.writer(output, lineterminator='\n')
        writer.writerows([tr_loss,vl_loss])

    print("save the models")

    torch.save(model.state_dict(), './net.pth')

    if opt.mode.upper() == 'BOTH':  
        print("run model for Images in test folder.")
        test(model,len(train_loader))


if __name__ == '__main__':
    main()