from metric import *
from preprocessing.data import *
from preprocessing.augmentation import *
from loss.cyclic_lr import CosineAnnealingLR_with_Restart
from time import time as timer
from utils import get_model, get_augment, get_n_params
from torch.utils.data import DataLoader
from torch import optim
import sys
sys.path.append("..")


def run_train(config):
    #############################
    # seeds set #################
    #############################

    # TODO: add deterministic/random seed option
    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(a=42, version=2)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


    #############################
    # path config ###############
    #############################

    # figuring out the path (dependant of model (A, B, C), image mode (color, ir, depth), image size (32, 48...))
    out_dir = './models'
    config.model_name = config.model + '_' + config.image_mode + '_' + str(config.image_size)
    out_dir = os.path.join(out_dir,config.model_name)

    EXP_TABS = 40

    # device = "cuda"

    initial_checkpoint = config.pretrained_model

    criterion = softmax_cross_entropy_criterion

    # make checkpoint, backup dirs ------------------------------
    if not os.path.exists(out_dir + '/checkpoint'):
        os.makedirs(out_dir + '/checkpoint')
    if not os.path.exists(out_dir + '/backup'):
        os.makedirs(out_dir + '/backup')
    if not os.path.exists(out_dir + '/backup'):
        os.makedirs(out_dir + '/backup')

    #############################
    ###### Logger config ########
    #############################

    # verbose output into txt file (configuration etc.)
    log = Logger()
    log.open(os.path.join(out_dir, config.model_name+'.txt'), mode='a')
    log.write('config:\n')
    log.write('out_dir:\t"{}"\n'.format(os.path.abspath(out_dir)).expandtabs(EXP_TABS))

    for arg in vars(config):
        log.write("{}:\t{}\n".format(arg, getattr(config, arg)).expandtabs(EXP_TABS))

    log.write('\ndataset setting:\n')

    ##############################
    #  Train dataset #############
    ##############################

    # this is now a function (without parameters being passed yet..)
    augment = get_augment(config.image_mode)

    # inherits Dataset (torch)
    # rotate, scale, augument images
    # fold index ne sluzi nicemu, zasad...
    train_dataset = FDDataset(mode='train',
                              modality=config.image_mode,
                              image_size=config.image_size,
                              fold_index=config.train_fold_index,
                              augment=augment,
                              dataset_path=config.train_list)

    # custom object (not torch inherited)
    # important to have __setattr__, __iter__, __len__
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=config.train_batch_size,
                              drop_last=True,
                              num_workers=config.dataset_workers,
                              worker_init_fn=np.random.seed)

    #############################
    # Validation dataset ########
    #############################

    valid_dataset = FDDataset(mode='val',
                              modality=config.image_mode,
                              image_size=config.image_size,
                              fold_index=config.train_fold_index,
                              augment=augment,
                              dataset_path=config.validation_list)

    # TODO: parameters? autotune?
    valid_loader = DataLoader(valid_dataset,
                              shuffle=False,
                              batch_size=max(1, config.valid_batch_size // 36),
                              drop_last=False,
                              num_workers=config.dataset_workers,
                              worker_init_fn=np.random.seed)

    assert(len(train_dataset) >= config.train_batch_size)

    ##############################
    # Net and dataset stats ######
    ##############################

    log.write('train_batch_size:\t{}\n'.format(config.train_batch_size).expandtabs(EXP_TABS))
    log.write('valid_batch_size:\t{}\n'.format(config.valid_batch_size).expandtabs(EXP_TABS))
    log.write('\nneural net:\n')

    net = get_model(model_name=config.model, num_class=2, is_first_bn=True)
    log.write("number of params:\t{}\n".format(get_n_params(net)).expandtabs(EXP_TABS))

    # for param in net.parameters():
    #     print(param.data)

    ###############################
    # Net init  ###################
    ###############################

    net = torch.nn.DataParallel(net)
    net = net.cuda() if torch.cuda.is_available() else net.cpu()

    if initial_checkpoint is not None:
        initial_checkpoint = os.path.join(out_dir +'/checkpoint',initial_checkpoint)
        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    log.write('{}\n'.format(type(net)))
    log.write('criterion:\t{}\n'.format(criterion).expandtabs(EXP_TABS))
    log.write('\n')

    ################################
    # Train setup ##################
    ################################

    start_iter = 0
    log.write('\n')
    i = 0
    batch_loss = np.zeros(6, np.float32)
    start = timer()
    # -----------------------------------------------
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0005)

    sgdr = CosineAnnealingLR_with_Restart(optimizer,
                                          T_max=config.epochs,
                                          T_mult=1,
                                          model=net,
                                          out_dir='../input/',
                                          take_snapshot=False,
                                          eta_min=1e-3)

    global_min_acer = 1.0


    ##########################################
    #       TRAINING              ############
    ##########################################
    # print("bok!")
    # tqdm = dummy

    # Random restarts
    for restart_index in range(config.epochs_valid_start):
        log.write('\n' + ('#'*50) + '\n')
        log.write('restart index: ' + str(restart_index) + "\n")
        min_acer = 1.0

        ############################
        # Epochs ###################
        ############################

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


            #################################
            # BATCH #########################
            #################################

            for ijk, (inpt, truth) in tqdm(enumerate(train_loader), desc="epoch progress", total=len(train_loader),
                                           leave=False):
            # for ijk, (inpt, truth) in enumerate(train_loader):
                batch_count += 1

                # one iteration update  -------------
                net.train()
                inpt = inpt.cuda() if torch.cuda.is_available() else inpt.cpu()
                truth = truth.cuda() if torch.cuda.is_available() else truth.cpu()

                logit, _, _ = net.forward(inpt)

                # print("logits:", logit)
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
                # print("batch {} loss: {} acc: {}".format(ijk + 1, batch_loss[0], batch_loss[1]))
                # print("truth value:", truth)
                # input()

                # if ijk in range(5):
                #     image = np.moveaxis(inpt.cpu().numpy()[0], 0, -1)
                #     print(image)
                #     cv2.namedWindow("lol")
                #     cv2.imshow('lol', image)
                #     cv2.waitKey(0)

            # train stats
            avg_batch_losses = sum(batch_losses)/len(batch_losses)
            avg_batch_accs = sum(batch_accs)/len(batch_accs)
            now = timer()
            # print()
            log.write(
                ('[train] {} | rst: {} | ep: {} | lrn_rate: {:.5f} | loss: {:.6f} | acc: {:.6f} | epoch_tm: {} ' +
                 '| avg_batch_tm: {} | time: {}\n').format(config.model_name, restart_index, epoch + 1, lr,
                                                           avg_batch_losses, avg_batch_accs,
                                                           time_to_str(now - start_batch, 'sec'),
                                                           time_to_str((now - start_batch) / batch_count, 'ms'),
                                                           time_to_str(now - start), 'sec'))

            ##############################
            # VALIDATION #################
            ##############################

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
                    ('[valid] loss: {:.6f} | acc: {:.6f} | ' +
                     'acer: {:.6f} | tpr: {:.5f} | fpr: {:.5f} | valid_time: {} | ' +
                     'time: {}\n').format(valid_loss, valid_acc, valid_acer, valid_tpr, valid_fpr,
                                          time_to_str(valid_time, 'sec'), time_to_str(now, 'min')))

                checkpoint = False
                if (valid_acer < min_acer) or (valid_acer < global_min_acer) and epoch > 0:
                    log.write('[checkpoint] ')
                    checkpoint = True

                # this cycle best
                if valid_acer < min_acer and epoch > 0:
                    min_acer = valid_acer
                    checkpoint_name = out_dir + '/checkpoint/restart_' + \
                                      str(restart_index).zfill(3) + '_min_acer_model.pth'
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

            log.write("\n")

        checkpoint_name = out_dir + '/checkpoint/restart_' + str(restart_index).zfill(3) + '_final_model.pth'
        torch.save(net.state_dict(), checkpoint_name)
        log.write('[checkpoint!] save restart ' + str(restart_index) + ' final model \n')
