from VISoR_Analysis.texture_classifier.resnet3d import *
import tifffile

def train(net):
    net.train()
    num_steps = 10000
    snap_dir = 'F:/chaoyu/test/thy1/model'
    log = open(os.path.join(snap_dir, 'train_log.txt'), 'w')


    image_path = 'E:/brains/THY1_YFP_891/Reconstruction/Brain'
    image_files = [os.path.join(image_path, 'Z{:05d}_C1.tif'.format(i)) for i in range(0, 3150)]
    train_data = patch_data(image_files, 'F:/chaoyu/test/thy1/template.tif', 'F:/chaoyu/test/thy1/438/deformationField.mhd')
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)

    optimizer = optim.SGD(net.parameters(), lr=5e-4, momentum=0.9, weight_decay=5e-4)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

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
        images, labels = batch
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
        optimizer.zero_grad()
        pred = net(images)
        loss = criterion(pred, labels)
        #if pred[0][0] > pred[0][1] and labels[0] == 1:
        #    loss *= 0.01
        loss.backward()
        optimizer.step()

        loss_num = loss.data.cpu().numpy()
        print('iter = ', ct, 'of', num_steps,'completed, loss = ', loss_num)
        log.write(str(loss_num) + '\n')
        log.flush()


        if ct % 1000 == 0 and ct!=0:
            print('taking snapshot ...')
            torch.save(net.state_dict(), os.path.join(snap_dir, 'texture_classifier_'+str(ct)+'.pth'))

    print('save model ...')
    torch.save(net.state_dict(), os.path.join(snap_dir, 'texture_classifier.pth'))


def test(net):
    net.eval()
    image_path = 'E:/brains/THY1_YFP_891/Reconstruction/Brain'
    image_files = [os.path.join(image_path, 'Z{:05d}_C1.tif'.format(i)) for i in range(1000, 2000)]
    test_data = patch_data_sequential(image_files)
    batch_size = 64
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)
    output = np.zeros(test_data.length, np.float32)
    texture_filter_output = np.zeros((test_data.length, 256), np.float32)
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
        images = batch
        #images, labels = Variable(images).cuda(), Variable(labels.float()).cuda()
        images = Variable(images).cuda()
        pred, top_filter = net.get_top_filter_value(images)
        pred = torch.argmax(pred, 1).data[:,0,0,0].cpu().numpy()
        np.copyto(output[ct * batch_size: ct * batch_size + pred.shape[0]], pred)
        #filter = net.top_value
        top_filter = top_filter.data[:,:,0,0,0].cpu().numpy()
        np.copyto(texture_filter_output[ct * batch_size: ct * batch_size + top_filter.shape[0]], top_filter)
        ct += 1
        print(ct * batch_size, test_data.length)
    output.resize((test_data.size[2], test_data.size[1], test_data.size[0]))
    tifffile.imwrite('F:/chaoyu/test/thy1/texture_classifier_output/pred.tif', output)
    texture_filter_output.resize((test_data.size[2], test_data.size[1], test_data.size[0], 256))
    for i in range(256):
        tifffile.imwrite('F:/chaoyu/test/thy1/texture_classifier_output/{}.tif'.format(i), texture_filter_output[:,:,:,i])



if __name__ == '__main__':
    net = ResNet3D(4).cuda()
    try:
        net.load_state_dict(
            torch.load('F:/chaoyu/test/thy1/model/texture_classifier_100000.pth'))
    except Exception as e:
        print(e)
    train(net)
    #test(net)
