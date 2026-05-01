from VISoR_Analysis.texture_classifier.resnet3d import *
from VISoR_Analysis.texture_classifier.data import *
import torch.nn.functional as F

class AutoEncoder(nn.Module):

    def __init__(self, num_features=64):
        self.inplanes = 16
        super(AutoEncoder, self).__init__()

        # Encoder
        self.conv1 = nn.Conv3d(1, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=(1, 1, 1))
        self.up_layer1 = self._make_layer(CubicBlock, 32, 2)
        self.up_layer2 = self._make_layer(CubicBlock, 64, 2, resample=1)
        self.up_layer3 = self._make_layer(CubicBlock, 128, 2, resample=1)
        self.conv2_mu = nn.Conv3d(self.inplanes, self.inplanes, (3, 3, 3), stride=2, padding=(1, 1, 1))
        #self.conv2_var = nn.Conv3d(self.inplanes, self.inplanes, (3, 3, 3), stride=2, padding=(1, 1, 1))
        self.conv3_mu = nn.Conv3d(self.inplanes, num_features, (1, 1, 1))
        #self.conv3_var = nn.Conv3d(self.inplanes, num_features, (1, 1, 1))

        # Decoder
        self.conv4 = nn.ConvTranspose3d(num_features, self.inplanes, (3, 3, 3), stride=2)
        self.down_layer1 = self._make_layer(CubicBlock, 64, 2, resample=-1)
        self.down_layer2 = self._make_layer(CubicBlock, 32, 2, resample=-1)
        self.down_layer3 = self._make_layer(CubicBlock, 16, 2, resample=-1)
        self.conv5 = nn.Conv3d(16, 1, kernel_size=1,
                               bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, resample=0):
        resample_block = None
        layers = []
        if resample != 0 or self.inplanes != planes * block.expansion:
            if resample >= 0:
                stride = resample + 1
                resample_block = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )
                layers.append(block(self.inplanes, planes, stride, resample_block))
            else:
                stride = -resample + 1
                resample_block = nn.Sequential(
                    nn.ConvTranspose3d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False, output_padding=1),
                    nn.BatchNorm3d(planes * block.expansion),
                )
                layers.append(block(self.inplanes, planes, -stride, resample_block))
        else:
            layers.append(block(self.inplanes, planes, 1, resample_block))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def encode(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.up_layer1(x)
        x = self.up_layer2(x)
        x = self.up_layer3(x)
        mu = self.conv2_mu(x)
        #var = self.conv2_var(x)
        mu = self.relu(mu)
        #var = self.relu(var)
        return self.conv3_mu(mu)#, self.conv3_var(var)

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x):
        x = self.conv4(x)
        x = x[:,:,0:x.size()[2]-1,0:x.size()[3]-1,0:x.size()[4]-1]
        x = self.down_layer1(x)
        x = self.down_layer2(x)
        x = self.down_layer3(x)
        x = self.conv5(x)
        return x

    def forward(self, x):
        #mu, logvar = self.encode(x)
        x = self.encode(x)
        #x = self.reparametrize(mu, logvar)
        x = self.decode(x)
        return x#, mu, logvar


def train(net):
    net.train()
    num_steps = 100000
    snap_dir = 'F:/chaoyu/test/autoencoder/model'
    log = open(os.path.join(snap_dir, 'train_log.txt'), 'w')

    image_path = 'D:/Hao/Data/converted/cfos-C2_2652/Reconstruction/BrainImage'
    image_files = [os.path.join(image_path, 'Z{:05d}_C0.tif'.format(i)) for i in range(0, 3000)]
    train_data = VolumeData(image_files)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True, num_workers=2)

    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)

    '''
    def loss_function(recon_x, x, mu, logvar):
        MSE = F.mse_loss(recon_x, x)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * -0.5
        return torch.log(MSE + KLD)
    '''
    loss_function = nn.MSELoss()

    ct = 0
    enum = enumerate(train_loader)
    while 1:
        if ct >= num_steps:
            break
        try:
            i_iter, batch = enum.__next__()
        except StopIteration:
            enum = enumerate(train_loader)
        except RuntimeError as e:
            print(e)
            continue
        ct += 1
        images = batch
        images = Variable(images).cuda()
        images = torch.log(torch.clamp(images, 100))
        optimizer.zero_grad()
        #pred, mu, logvar = net(images)
        pred = net(images)
        #loss = loss_function(pred, images, mu, logvar)
        loss = loss_function(pred, images)
        loss.backward()
        optimizer.step()

        loss_num = loss.data.cpu().numpy()
        print('iter = ', ct, 'of', num_steps,'completed, loss = ', loss_num)
        log.write(str(loss_num) + '\n')
        log.flush()


        if ct % 1000 == 0 and ct!=0:
            print('taking snapshot ...')
            torch.save(net.state_dict(), os.path.join(snap_dir, 'autoencoder_'+str(ct)+'.pth'))

    print('save model ...')
    torch.save(net.state_dict(), os.path.join(snap_dir, 'autoencoder.pth'))


def test(net: AutoEncoder):
    net.eval()
    image_path = 'E:/DATA/cfos-FS5-2875/Reconstruction/BrainImage'
    image_files = [os.path.join(image_path, 'Z{:05d}_C0.tif'.format(i)) for i in range(1280, 1408)]
    test_data = VolumeData(image_files)
    batch_size = 1
    downsample = 16
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    output = np.zeros([test_data.size[2] * test_data.block_size[2],
                       test_data.size[1] * test_data.block_size[1],
                       test_data.size[0] * test_data.block_size[0]], np.float32)
    features_output = np.zeros([64, output.shape[0] // downsample,  output.shape[1] // downsample,  output.shape[2] // downsample], np.float32)
    tl = enumerate(test_loader)
    ct = 0
    fp = 0
    while 1:
        try:
            i_iter, batch = tl.__next__()
        except StopIteration:
            break
        except RuntimeError as e:
            print(e)
            print(test_data.image_list[i_iter])
            continue
        except ValueError as e:
            print(test_data.image_list[i_iter])
            print(e)
            continue
        pos = (i_iter % test_data.size[0],
               (i_iter % (test_data.size[0] * test_data.size[1])) // test_data.size[0],
               i_iter // (test_data.size[0] * test_data.size[1]))
        pos_s = [pos[i] * test_data.block_size[i] // downsample for i in range(3)]

        images = batch
        images = Variable(images).cuda()
        images = torch.log(torch.clamp(images, 100))
        features = net.encode(images)
        #mu, logvar = net.encode(images)
        #features = net.reparametrize(mu, logvar)
        pred = net.decode(features).detach().cpu().numpy()
        features = features.detach().cpu().numpy()
        np.copyto(output[pos[2] * test_data.block_size[2]: (pos[2] + 1) * test_data.block_size[2],
                  pos[1] * test_data.block_size[1]: (pos[1] + 1) * test_data.block_size[1],
                  pos[0] * test_data.block_size[0]: (pos[0] + 1) * test_data.block_size[0]], pred)
        np.copyto(features_output[:,pos_s[2]:pos_s[2] + features.shape[2],
                  pos_s[1]:pos_s[1] + features.shape[3],
                  pos_s[0]:pos_s[0] + features.shape[4]], features)
        ct += 1
        print(pos)
    tifffile.imwrite('F:/chaoyu/test/autoencoder/output/pred.tif', output)
    tifffile.imwrite('F:/chaoyu/test/autoencoder/output/feature.tif', features_output)


if __name__ == '__main__':
    net = AutoEncoder().cuda()
    try:
        net.load_state_dict(
            torch.load('F:/chaoyu/test/autoencoder/model/autoencoder_100000.pth'))
    except Exception as e:
        print(e)
    #train(net)
    test(net)
