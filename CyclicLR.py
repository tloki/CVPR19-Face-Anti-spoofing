import sys
sys.path.append("..")
import argparse
from process.data import *
from process.augmentation import *
from metric import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
import time


def get_model(model_name, num_class, is_first_bn) -> nn.Module:
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
    # TODO: add random seed
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # figuring out the path (dependant of model (A, B, C), image mode (color, ir, depth), image size (32, 48...))
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)

    EXP_TABS = 40

    # device = "cuda"

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
    log.open(os.path.join(out_dir, config.model_name+'.txt'), mode='a')
    log.write('config:\n')
    log.write('out_dir:\t"{}"\n'.format(os.path.abspath(out_dir)).expandtabs(EXP_TABS))

    for arg in vars(config):
        log.write("{}:\t{}\n".format(arg, getattr(config, arg)).expandtabs(EXP_TABS))

    ## dataset ----------------------------------------
    log.write('\ndataset setting:\n')

    # this is now a function (without parameters being passed yet..)
    augment = get_augment(config.image_mode)

    ######################
    #  Train #############
    ######################

    # inherits Dataset (torch)
    # rotate, scale, augument images
    # fold index ne sluzi nicemu, zasad...
    train_dataset = FDDataset(mode='train', modality=config.image_mode, image_size=config.image_size,
                              fold_index=config.train_fold_index, augment=augment, dataset_path=config.train_list)

    # custom object (not torch inherited)
    # important to have __setattr__, __iter__, __len__
    train_loader = DataLoader(train_dataset,
                                shuffle=True,
                                batch_size=config.train_batch_size,
                                drop_last=True,
                                num_workers=config.dataset_workers)

    ######################
    # Validation ########
    ######################

    valid_dataset = FDDataset(mode='val', modality=config.image_mode,image_size=config.image_size,
                              fold_index=config.train_fold_index, augment=augment, dataset_path=config.validation_list)

    #TODO: parameters? autotune?
    valid_loader  = DataLoader(valid_dataset,
                               shuffle=False,
                               batch_size=max(1, config.valid_batch_size // 36),
                               drop_last=False,
                               num_workers=config.dataset_workers)

    assert(len(train_dataset)>=config.train_batch_size)
    log.write('train_batch_size:\t{}\n'.format(config.train_batch_size).expandtabs(EXP_TABS))
    log.write('valid_batch_size:\t{}\n'.format(config.valid_batch_size).expandtabs(EXP_TABS))
    # log.write('train_dataset : \n{}\n'%(train_dataset))
    # log.write('valid_dataset : \n%s\n'%(valid_dataset))
    # log.write('\n')
    log.write('\nneural net:\n')

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)

    log.write("number of params:\t{}\n".format(get_n_params(net)).expandtabs(EXP_TABS))
    # print(net)

    net = torch.nn.DataParallel(net)

    net = net.cuda() if torch.cuda.is_available() else net.cpu()
    # net = net.

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('{}\n'.format(type(net)))
    log.write('criterion:\t{}\n'.format(criterion).expandtabs(EXP_TABS))
    log.write('\n')

    iter_smooth = 20
    start_iter = 0
    log.write('\n')

    # print(get_n_params(net))

    iter = 0
    i = 0

    train_loss = np.zeros(6, np.float32)
    valid_metrics = np.zeros(6, np.float32)
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
    for restart_index in range(config.epochs_valid_start):
        log.write('\n' + ('#'*50) + '\n')
        log.write('restart index: ' + str(restart_index) + "\n")
        min_acer = 1.0

        for epoch in range(config.epochs):
            sgdr.step()
            lr = optimizer.param_groups[0]['lr']

            if epoch == 0:
                log.write('start learning rate : {:.4f}\n'.format(lr))

            # tm1 = time.time()

            sum_train_loss = np.zeros(6,np.float32)
            # sum_ = 0
            optimizer.zero_grad()

            start_batch = timer()

            batch_losses = []
            batch_accs = []
            batch_count = 0
            for inpt, truth in train_loader:
                batch_count += 1
                iter = i + start_iter

                # one iteration update  -------------
                net.train()
                inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()
                truth = truth.cuda() if torch.cuda.is_available() else truth.cpu()

                logit, _, _ = net.forward(inpt)

                truth = truth.view(logit.shape[0])

                loss = criterion(logit, truth)
                precision, _ = metric(logit, truth)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # print statistics  ------------
                batch_loss[:2] = np.array([loss.item(), precision.item(), ])
                batch_losses.append(batch_loss[0])
                batch_accs.append(batch_loss[1])

            # train stats
            batch_losses = sum(batch_losses)/len(batch_losses)
            batch_accs = sum(batch_accs)/len(batch_accs)
            now = timer()
            print()
            log.write(
                ('[train] {} | rst: {} | ep: {} | lrn_rate: {:.4f} | loss: {:.3f} | acc: {:.4f} | epoch_tm: {} ' +
                 '| avg_batch_tm: {} | time: {}\n').format(config.model_name, restart_index, epoch + 1, lr,
                                      batch_losses, batch_accs,
                                      time_to_str(now - start_batch, 'sec'),
                                      time_to_str((now - start_batch) / batch_count, 'sec'),
                                      time_to_str(now - start), 'sec'))

            # validation
            if epoch >= config.epochs_valid_start:
                # set mode to evaluation TODO: check documentation
                net.eval()

                valid_time = timer()
                valid_metrics,_ = do_valid_test(net, valid_loader, criterion, logger=log)
                valid_loss, valid_acer, valid_acc, valid_correct, valid_tpr, valid_fpr = valid_metrics[0:6]

                # print(valid_loss,)
                # print("valid_loss: {} valid_acer: {} valid_acc: {} valid_correct: {}".format(
                #     valid_loss, valid_acer, valid_acc, valid_correct))

                valid_time = timer() - valid_time
                now = timer() - start
                # TODO: back to tran mode
                net.train()


                # eval output
                log.write(
                    ('[valid] loss: {:.3f} | acc: {:.4f} | ' +
                     'acer: {:.3f} | tpr: {:.3f} | fpr: {:.3f} | valid_time: {} | ' +
                     'time: {}\n').format(valid_loss, valid_acc, valid_acer, valid_tpr, valid_fpr,
                                          time_to_str(valid_time, 'sec'), time_to_str(now, 'min')))

                checkpoint = False
                if (valid_acer < min_acer) or (valid_acer < global_min_acer) and epoch > 0:
                    log.write('[checkpoint] ')
                    checkpoint = True

                # this cycle best
                if valid_acer < min_acer and epoch > 0:
                    min_acer = valid_acer
                    checkpoint_name = out_dir + '/checkpoint/restart_' + str(restart_index).zfill(3) + '_min_acer_model.pth'
                    torch.save(net.state_dict(), checkpoint_name)
                    log.write('save restart ' + str(restart_index) + ' min acer model: ' + str(min_acer) + " | ")

                # global best
                if valid_acer < global_min_acer and epoch > 0:
                    global_min_acer = valid_acer
                    checkpoint_name = out_dir + '/checkpoint/global_min_acer_model.pth'
                    torch.save(net.state_dict(), checkpoint_name)
                    log.write('save global min acer model: ' + str(min_acer).zfill(3))

                if checkpoint and not epoch == config.epochs - 1:
                    now = timer()
                    log.write("| time: {}\n".format(time_to_str(now - start)))

        checkpoint_name = out_dir + '/checkpoint/restart_' + str(restart_index).zfill(3) + '_final_model.pth'
        torch.save(net.state_dict(), checkpoint_name)
        log.write('[checkpoint!] save restart ' + str(restart_index) + ' final model \n')


def run_realtime(config, dir):
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    # net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    net = torch.nn.DataParallel(net)

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    from mtcnn.mtcnn import MTCNN
    detector = MTCNN()

    test_dataset = FDDataset(mode='realtime', modality=config.image_mode,image_size=config.image_size,
                             fold_index=config.train_fold_index, augment=augment, dataset_path=None,
                             stream_device=config.stream, detect_function=detector.detect_faces)

    # test_loader = DataLoader(test_dataset,
    #                          shuffle=False,
    #                          batch_size=36,
    #                          drop_last=False,
    #                          num_workers=1)

    ins = []

    net.eval()



    out = infer_test_infinite(net, test_dataset)
    print()
    print("percentage predicted true:", 100*np.mean(out > 0.5))
    print("raw percentages:")

    np.set_printoptions(precision=2, suppress=True)
    preds = np.array(out).tolist()

    summary = []

    for (filename, label), out in zip(ins, preds):
        # print(filename, label, out, sep="\t")
        summary.append([filename, label, out])

    summary = sorted(summary, key=lambda f: f[2], reverse=True)
    for f, l, o in summary:
        print(f, l, o)

    # print(np.array(out) * 100)
    print('done')


# test (inference)
def run_test(config, dir):

    device = "cpu"

    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)
    initial_checkpoint = config.pretrained_model
    augment = get_augment(config.image_mode)

    # net ---------------------------------------
    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)

    # TODO: only if multiple GPUs
    net = torch.nn.DataParallel(net)

    net.to(device)

    if initial_checkpoint is not None:
        save_dir = os.path.join(out_dir + '/checkpoint', dir, initial_checkpoint)
        initial_checkpoint = os.path.join(out_dir + '/checkpoint',initial_checkpoint)
        print('\t' + 'initial_checkpoint = {}\n'.format(initial_checkpoint))
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
        if not os.path.exists(os.path.join(out_dir + '/checkpoint', dir)):
            os.makedirs(os.path.join(out_dir + '/checkpoint', dir))

    test_dataset = FDDataset(mode='test', modality=config.image_mode,image_size=config.image_size,
                             fold_index=config.train_fold_index,augment=augment, dataset_path=config.test_list)

    print("test batch size: {}".format(config.test_batch_size if config.test_batch_size > 1
                                        else str(config.test_batch_size) + " (realtime)"))

    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=config.test_batch_size,
                             drop_last=False,
                             num_workers=4)

    ins = []

    with open(config.test_list) as test_list:
        # import os.path
        for line in test_list.readlines():
            filename, _, _, label  = line.split(" ")
            filename = filename.strip()
            label = int(label)
            filename = os.path.basename(filename)
            # print(filename, label, sep="\t")
            ins.append([filename, label])
            # print(line)

    net.eval()

    start_time = time.time()
    out = infer_test(net, test_loader)
    time_delta = time.time() - start_time
    print("testing took {:.2f} seconds".format(time_delta))

    print("percentage predicted true:", 100*np.mean(out > 0.5))
    print("raw percentages:")

    np.set_printoptions(precision=2, suppress=True)
    preds = np.array(out).tolist()

    summary = []

    for (filename, label), out in zip(ins, preds):
        # print(filename, label, out, sep="\t")
        summary.append([filename, label, out])

    # summary = sorted(summary, key=lambda f: f[2], reverse=True)
    # for f, l, o in summary:
    #     print(f, l, o)

    # print(np.array(out) * 100)
    print('done')


# run train or dest function, dependant on input arguments
def main(config):

    # if config.device == "auto":
    #     device =
    # elif config.device == "cpu":
    #     net = net.cpu()
    # elif config.device == "gpu":
    #     net = net.cuda()
    # else:
    #     raise ValueError

    if config.mode.startswith("train"):
        run_train(config)

    elif config.mode.startswith("test"):
        # take the best model on validation set
        config.pretrained_model = r'global_min_acer_model.pth'
        run_test(config, dir='global_test_36_TTA')

    elif config.mode == 'realtime':
        config.pretrained_model = r'global_min_acer_model.pth'
        run_realtime(config, dir='global_test_36_TTA')

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
    parser.add_argument('--test_batch_size', type=int, default=20)
    # parser.add_argument('--cycle_num', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--epochs_valid_start', type=int, default=50)  # number of random restarts

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'infer_test', 'test', 'test-single', 'realtime'])

    parser.add_argument('--stream', default=None)

    parser.add_argument('--pretrained_model', type=str, default=None)
    # parser.add_argument('--dataset_path', type=str, default="/home/loki/Datasets/spoofing/NUAA/Detectedface52")

    parser.add_argument('--dataset_workers', type=int, default=4)
    parser.add_argument('--device', type=str, choices=["auto", "gpu", "cpu"], default="auto")

    # TODO: implement
    # parser.add_argument('--test-sample', type=str, default=None)
    parser.add_argument('--train_list', type=str, default=None)
    parser.add_argument('--validation_list', type=str, default=None)
    parser.add_argument('--test_list', type=str, default=None)

    config = parser.parse_args()

    # print(config)
    # print(config.dataset_path)

    main(config)

    print("end")
