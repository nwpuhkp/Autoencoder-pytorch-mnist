import itertools
import os

import numpy as np
from torch import Tensor
from torch.autograd import Variable

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image
from model import *

# 选择运算设备cpu/gpu
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

# 生成一个输出文件夹
def make_dir(image_dir):

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

# 保存网络输出结果
def save_decod_img(img, epoch,name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name.format(epoch))

# 训练程序
def training(model, train_loader, criterion, optimizer, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad() # 梯度清零
            outputs = model(img) # 图片输入网络模型
            loss = criterion(outputs, img) # 损失计算
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))

        # if epoch % 5 == 0:
        #     # 保存图片
        #     save_decod_img(outputs.cpu().data, epoch,'./MNIST_Out_Images/AE1_image{}.png')
        #     save_decod_img(img.cpu().data, epoch,'./ORA_Images/AE1_ora_image{}.png')

    return train_loss

def vae_training(model, train_loader, criterion, optimizer, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            # print("data:",data)
            img, _ = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad() # 梯度清零
            recon_batch, mu, log_var = model(img)
            loss = vae_loss_function(recon_batch, img, mu, log_var)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))

        if epoch % 5 == 0 :
            # 保存图片
            save_decod_img(recon_batch.cpu().data, epoch,'./MNIST_Out_Images/VAE_image{}.png')
            save_decod_img(img.cpu().data, epoch,'./ORA_Images/VAE_ora_image{}.png')

    return train_loss

def dae_training(model, train_loader, criterion, optimizer, Epochs):
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            # print("data:",data)
            img, _ = data
            # img = img.to(device)
            # img = img.view(img.size(0), -1)
            optimizer.zero_grad() # 梯度清零
            noisy = np.clip(img + np.random.normal(0, 0.6, img.shape).astype('float32'), 0, 1)
            recon = model(noisy)
            loss = criterion(recon, img)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))

        if epoch % 5 == 0 :
            # 保存图片
            save_decod_img(recon.cpu().data, epoch,'./MNIST_Out_Images/DAE_image{}.png')
            save_decod_img(img.cpu().data, epoch,'./ORA_Images/DAE_ora_image{}.png')

    return train_loss

def aae_training(encoder, decoder, train_loader, adversarial_loss, reconstruction_loss,optimizer_D, optimizer_G, Epochs):
    global x, decoded_x
    train_loss = []
    for epoch in range(Epochs):
        running_loss = 0.0
        for data in train_loader:
            # print("data:",data)
            x, _ = data
            valid = Variable(Tensor(x.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(x.shape[0], 1).fill_(0.0), requires_grad=False)
            # img = img.to(device)
            # img = img.view(img.size(0), -1)

            # 1) reconstruction + generator loss
            optimizer_G.zero_grad()
            fake_z = encoder(x)
            decoded_x = decoder(fake_z)
            validity_fake_z = discriminator(fake_z)
            G_loss = 0.001 * adversarial_loss(validity_fake_z, valid) + 0.999 * reconstruction_loss(decoded_x, x)
            G_loss.backward()
            optimizer_G.step()

            # 2) discriminator loss
            optimizer_D.zero_grad()
            real_z = Variable(Tensor(np.random.normal(0, 1, (x.shape[0], latent_dim))))
            real_loss = adversarial_loss(discriminator(real_z), valid)
            fake_loss = adversarial_loss(discriminator(fake_z.detach()), fake)
            D_loss = 0.5 * (real_loss + fake_loss)
            D_loss.backward()
            optimizer_D.step()

            running_loss += G_loss.item()

        loss = running_loss / len(train_loader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.6f}'.format(
            epoch+1, Epochs, loss))

        if epoch % 5 == 0 :
            # 保存图片
            save_decod_img(decoded_x.cpu().data, epoch,'./MNIST_Out_Images/AAE_image{}.png')
            save_decod_img(x.cpu().data, epoch,'./ORA_Images/AAE_ora_image{}.png')

    return train_loss

def vae_loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

# 测试图片重建效果
def test_image_reconstruct(model, test_loader):
    for batch in test_loader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        outputs = model(img)
        outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'A2_32_MNIST_reconstruction.png')
        break

def VAE_test_image_reconstruct(model, test_loader):
    for batch in test_loader:
        img, _ = batch
        img = img.to(device)
        img = img.view(img.size(0), -1)
        recon_batch, mu, log_var = model(img)
        outputs = recon_batch.view(recon_batch.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'VAE_MNIST_reconstruction.png')
        break

def DAE_test_image_reconstruct(model, test_loader):
    for batch in test_loader:
        img, _ = batch
        # img = img.to(device)
        # img = img.view(img.size(0), -1)
        noisy = np.clip(img + np.random.normal(0, 0.6, img.shape).astype('float32'), 0, 1)
        recon = model(noisy)
        outputs = recon.view(recon.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'DAE_MNIST_reconstruction.png')
        break

def AAE_test_image_reconstruct(encoder, decoder, test_loader):
    for batch in test_loader:
        img, _ = batch
        # img = img.to(device)
        # img = img.view(img.size(0), -1)
        fake_z = encoder(img)
        decoded_x = decoder(fake_z)
        outputs = decoded_x.view(decoded_x.size(0), 1, 28, 28).cpu().data
        save_image(outputs, 'AAE_MNIST_reconstruction.png')
        break

if __name__ == '__main__':
    # 超参数设置 训练次数 学习率 每次输入图片数
    Epochs = 21
    Lr_Rate = 1e-4
    Batch_Size = 50
    Choose = 2

    # 训练集设置，pytorch进行深度学习对数据集处理包括生成dataset类和dataloader类
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.MNIST(root='./res/data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./res/data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)
    print(train_set)
    print(train_set.classes)

    # 模型结构定义
    if Choose == 1:
        model = Autoencoder1()
    elif Choose == 2:
        model = Autoencoder2()
    elif Choose == 3:
        model = DAE()
    elif Choose == 4:
        model = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=2)
    else:
        model = Autoencoder1()#nothing
        # 1) generator
        encoder = AAE_Encoder()
        decoder = AAE_Decoder()
        # 2) discriminator
        discriminator = Discriminator()
        print(encoder)
        print(decoder)
        print(discriminator)
        # loss
        adversarial_loss = nn.BCELoss()
        reconstruction_loss = nn.MSELoss()
        # optimizer
        optimizer_G = torch.optim.Adam(itertools.chain(encoder.parameters(), decoder.parameters()), lr=Lr_Rate,
                                       betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=Lr_Rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss()
    # print(model)
    # 优化器定义
    optimizer = optim.Adam(model.parameters(), lr=Lr_Rate)

    # 运算设备选择
    device = get_device()
    model.to(device)
    make_dir('MNIST_Out_Images')
    make_dir('ORA_Images')
    # 开始训练
    #AE
    train_loss = training(model, train_loader, criterion, optimizer, Epochs)
    test_image_reconstruct(model, test_loader)
    #VAE
    # train_loss = vae_training(model, train_loader, criterion, optimizer, Epochs)
    # VAE_test_image_reconstruct(model, test_loader)
    #DAE
    # train_loss = dae_training(model, train_loader, criterion, optimizer, Epochs)
    # DAE_test_image_reconstruct(model, test_loader)
    #AAE
    # train_loss = aae_training(encoder,decoder,train_loader,adversarial_loss,reconstruction_loss,optimizer_D,optimizer_G,Epochs)
    # AAE_test_image_reconstruct(encoder,decoder,test_loader)

    torch.save(model.state_dict(), 'model.pt')

    plt.figure()
    plt.plot(train_loss)
    plt.title('Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('deep_aae_mnist_loss.png')