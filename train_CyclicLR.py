import os
# os.environ['CUDA_VISIBLE_DEVICES'] =  '4,5,6,7' #'3,2,1,0'
import sys
sys.path.append("..")
import argparse
from process.data import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
import time

def get_model(model_name, num_class,is_first_bn):
    if model_name == 'baseline':
        from model.model_baseline import Net
    elif model_name == 'model_A':
        from model.FaceBagNet_model_A import Net
    elif model_name == 'model_B':
        from model.FaceBagNet_model_B import Net
    elif model_name == 'model_C':
        from model.FaceBagNet_model_C import Net

    net = Net(num_class=num_class,is_first_bn=is_first_bn)
    return net


def get_augment(image_mode):
    if image_mode == 'color':
        augment = color_augumentor
    elif image_mode == 'depth':
        augment = depth_augumentor
    elif image_mode == 'ir':
        augment = ir_augumentor
    return augment


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        # print("layer")
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def run_train(config):
    # figuring out the path (dependant of model (A, B, C), image mode (color, ir, depth), image size (32, 48...))
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)


    initial_checkpoint = config.pretrained_model


    criterion  = softmax_cross_entropy_criterion

    # make checkpoint, backup dirs ------------------------------
    if not os.path.exists(out_dir +'/checkpoint'):
        os.makedirs(out_dir +'/checkpoint')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')
    if not os.path.exists(out_dir +'/backup'):
        os.makedirs(out_dir +'/backup')

    # verbose output into txt file (configuration etc.)
    log = Logger()
    log.open(os.path.join(out_dir,config.model_name+'.txt'), mode='a')
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')
    log.write('\t<additional comments>\n')
    log.write('\t  ... xxx baseline  ... \n')
    log.write('\n')

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')

    # this is now a function (without parameters being passed yet..)
    augment = get_augment(config.image_mode)

    ######################
    #  Train #############
    ######################

    # inherits Dataset (torch)
    # rotate, scale, augument images
    # fold index ne sluzi nicemu, zasad...
    train_dataset = FDDataset(mode = 'train', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataset_path=config.dataset_path)

    # custom object (not torch inherited)
    # important to have __setattr__, __iter__, __len__
    train_loader  = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size  = config.train_batch_size,
                                drop_last   = True,
                                num_workers = 4)

    ######################
    # Validation ########
    ######################

    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataset_path=config.dataset_path)

    #TODO: parameters? autotune?
    valid_loader  = DataLoader(valid_dataset,
                               shuffle=False,
                               batch_size = max(1, config.valid_batch_size // 36),
                               drop_last  = False,
                               num_workers = config.dataset_workers)

    assert(len(train_dataset)>=config.train_batch_size)
    log.write('train_batch_size = %d\n'%(config.train_batch_size))
    log.write('valid_batch_size = %d\n' % (config.valid_batch_size))
    log.write('train_dataset : \n%s\n'%(train_dataset))
    log.write('valid_dataset : \n%s\n'%(valid_dataset))
    log.write('\n')
    log.write('** net setting **\n')

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)

    print(net)

    net = torch.nn.DataParallel(net)

    net = net.cuda() if torch.cuda.is_available() else net.cpu()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('%s\n'%(type(net)))
    log.write('criterion=%s\n'%criterion)
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    print(get_n_params(net))

    ## start training here! ##############################################
    log.write('** start training here! **\n')
    log.write('                                  |------------ VALID -------------|-------- TRAIN/BATCH ----------|         \n')
    log.write('model_name   lr   iter  epoch     |     loss      acer      acc    |     loss              acc     |  time   \n')
    log.write('----------------------------------------------------------------------------------------------------\n')

    iter = 0
    i = 0

    train_loss = np.zeros(6, np.float32)
    valid_loss = np.zeros(6, np.float32)
    batch_loss = np.zeros(6, np.float32)

    start = timer()
    # -----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.epochs,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0
    for cycle_index in range(config.cycle_num):
        print('cycle index: ' + str(cycle_index))
        min_acer = 1.0

        for epoch in range(0, config.epochs):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']
            print('lr : {:.4f}'.format(lr))

            sum_train_loss = np.zeros(6,np.float32)
            sum = 0
            optimizer.zero_grad()

            for input, truth in train_loader:
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                input = input.cuda() if torch.cuda.is_available() else input.cpu()
                truth = truth.cuda() if torch.cuda.is_available() else truth.cpu()

                logit,_,_ = net.forward(input)

                # input_img = input.detach().numpy()
                #
                # for img in input_img:
                #     r, g, b = img
                #     cv2.imshow('R kanal', img[0])
                #     k = chr(cv2.waitKey())
                #
                # cv2.destroyAllWindows()

                truth = truth.view(logit.shape[0])

                loss  = criterion(logit, truth)
                precision,_ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array(( loss.item(), precision.item(),))

                sum += 1
                if iter%iter_smooth == 0:
                    train_loss = sum_train_loss/sum
                    sum = 0
                i=i+1

            # if epoch >= config.epochs // 2:
            if epoch >= 0:
                net.eval()

                t1 = time.time()
                valid_loss,_ = do_valid_test(net, valid_loader, criterion)
                t2 = time.time()  - t1
                print("validation time: {:.2f} seconds".format(t2))

                net.train()

                if valid_loss[1] < min_acer and epoch > 0:
                    min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save cycle ' + str(cycle_index) + ' min acer model: ' + str(min_acer) + '\n')

                if valid_loss[1] < global_min_acer and epoch > 0:
                    global_min_acer = valid_loss[1]
                    ckpt_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), ckpt_name)
                    log.write('save global min acer model: ' + str(min_acer) + '\n')

            asterisk = ' '

            log.write(config.model_name+' Cycle %d: %0.4f %5.1f %d | %0.6f  %0.6f  %0.3f %s  | %0.6f  %0.6f |%s \n' % (
                cycle_index, lr, iter, epoch,
                valid_loss[0], valid_loss[1], valid_loss[2], asterisk,
                batch_loss[0], batch_loss[1],
                time_to_str((timer() - start), 'min')))

        ckpt_name = out_dir + '/checkpoint/Cycle_' + str(cycle_index) + '_final_model.pth'
        torch.save(net.state_dict(), ckpt_name)
        log.write('save cycle ' + str(cycle_index) + ' final model \n')


# test (inference)
def run_test(config, dir):
    out_dir = './models' # what
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    # net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()


    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))


    valid_dataset = FDDataset(mode = 'val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataset_path=config.dataset_path)
    valid_loader  = DataLoader( valid_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    test_dataset = FDDataset(mode = 'test', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index,augment=augment, dataset_path=config.dataset_path)

    test_loader  = DataLoader( test_dataset,
                                shuffle=False,
                                batch_size  = config.batch_size,
                                drop_last   = False,
                                num_workers=8)

    criterion = softmax_cross_entropy_criterion
    # net.eval()
    #
    # valid_loss,out = do_valid_test(net, valid_loader, criterion)
    # print('%0.6f  %0.6f  %0.3f  (%0.3f) \n' % (valid_loss[0], valid_loss[1], valid_loss[2], valid_loss[3]))

    print('infer!!!!!!!!!')
    out = infer_test(net, test_loader)
    print(out)
    print('done')
    # submission(out,save_dir+'_noTTA.txt', mode='test')


# run train or dest function, dependant on input arguments
def main(config):
    if config.mode == 'train':
        run_train(config)

    if config.mode == 'infer_test':
        # take the best model on validation set
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')

    return


# parse input arguments and pass them to main function
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_fold_index', type=int, default = -1)

    parser.add_argument('--model', type=str, default='model_A')
    parser.add_argument('--image_mode', type=str, default='color', choices=['color', 'depth', 'ir'])
    parser.add_argument('--image_size', type=int, default=48) # empirically adjusted

    parser.add_argument('--train_batch_size', type=int, default=128)
    parser.add_argument('--valid_batch_size', type=int, default=128)
    parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50) # aka number of epochs

    parser.add_argument('--mode', type=str, default='train', choices=['train','infer_test'])
    parser.add_argument('--pretrained_model', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default="/home/loki/Datasets/spoofing/NUAA/Detectedface")

    parser.add_argument('--dataset-workers', type=int, default=4)

    # TODO: implement
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--validation_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)

    config = parser.parse_args()

    print(config)
    print(config.dataset_path)

    main(config)