import logging
import time

from common.eval import *
from common.eval_setting import *
import torch.optim as optim
model.eval()

if P.mode == 'test_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood', 'ood_pre']:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                        train_loader=train_loader, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = dict()
        for ood_score in P.ood_score:
            mean = 0
            for ood in auroc_dict.keys():
                mean += auroc_dict[ood][ood_score]
            mean_dict[ood_score] = mean / len(auroc_dict.keys())
        auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for ood in auroc_dict.keys():
        message = ''
        best_auroc = 0
        for ood_score, auroc in auroc_dict[ood].items():
            message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
            if auroc > best_auroc:
                best_auroc = auroc
        message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
        if P.print_score:
            print(message)
        bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))
    print("我可以我能行")
    #计算圆心：
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    c=init_center_c(net=model,train_loader=train_loader)
    # Set optimizer (Adam optimizer for now)
    optimizer = optim.Adam(model.parameters(), lr=P.svdd_lr, weight_decay=P.dweight_decay,
                       amsgrad=True)

    # Set learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=P.dlr_milestones, gamma=0.1)
    #set train
    print("开始正式训练")
    start_time = time.time()
    model.train()
    for epoch in range(P.svdd_epochs):

        scheduler.step()

        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for data in train_loader:
            inputs, _ = data
            inputs = inputs.to(device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            _, outputs_aux = model(inputs,simclr=True)
            outputs = outputs_aux['simclr']
            dist = torch.sum((outputs - c) ** 2, dim=1)
            loss = torch.mean(dist)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        # log epoch statistics
        epoch_train_time = time.time() - epoch_start_time
        # print(loss_epoch / n_batches)
        logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                    .format(epoch + 1, P.svdd_epochs, epoch_train_time, loss_epoch / n_batches))

    train_time = time.time() - start_time
    logger.info('Training time: %.3f' % train_time)

    logger.info('Finished training.')
    print("可以进行到这")
    scores_in=None
    for data in test_loader:
        inputs, _ = data
        inputs = inputs.to(device)
        _, outputs_aux = model(inputs, simclr=True)
        outputs = outputs_aux['simclr']
        score = torch.sum((outputs - c) ** 2, dim=1)
        if scores_in==None:
            scores_in=score
        else:
            scores_in=torch.cat((scores_in,score),dim=0)
    print(scores_in.size())
    # scores_ood=None
    # for data in test_loader_ood:
    #     inputs, _ = data
    #     inputs = inputs.to(device)
    #     _, outputs_aux = model(inputs, simclr=True)
    #     outputs = outputs_aux['simclr']
    #     score = torch.sum((outputs - c) ** 2, dim=1)
    #     if scores_ood == None:
    #         scores_ood = score
    #     else:
    #         scores_ood = torch.cat((scores_ood, score), dim=0)
    # print(scores_ood.size())




else:
    raise NotImplementedError()


