import logging
import time

import numpy as np

import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE

import numpy
from common.eval import *
from common.eval_setting import *

import torch.optim as optim
from evals.evals import get_auroc
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#设置日志文件

logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
handler = logging.FileHandler("log.txt")
handler.setLevel(logging.INFO)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(handler)
logger.addHandler(console)
logger.info('正类为%s' % P.one_class_idx)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
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
    # logger.info('CSI的精度%s' % P.one_class_idx)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))
    print("我可以我能行")
    #计算圆心：
    # logger = logging.getLogger()
    # logging.basicConfig(level=logging.INFO)
    c=init_center_c(net=model,train_loader=train_loader,dataset=P.dataset)
    # Set optimizer (Adam optimizer for now)   设施优化方式
    optimizer = optim.Adam(model.parameters(), lr=P.svdd_lr, weight_decay=P.dweight_decay,
                       amsgrad=True)

    # Set learning rate scheduler   对学习率进行调整
    dweight_decay=[P.dweight_decay]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=dweight_decay, gamma=0.1)
    score_sum=[]
    #set train
    print("开始正式训练")
    start_time = time.time()
    max_score=0.
    # tsne = TSNE(n_components=2, init='pca', random_state=0)
    model.train()
    for epoch in range(P.svdd_epochs):

            # ls=[]
        with torch.no_grad():
            scores_in = None
            for data in test_loader:
                inputs, _ = data
                inputs = inputs.to(device)
                if P.dataset =='mnist':
                    inputs = torch.cat((inputs, inputs, inputs), 2)
                    inputs = inputs.reshape(inputs.shape[0], 3, 28, 28)
                _, outputs_aux = model(inputs, simclr=True)
                outputs = outputs_aux['simclr']

                score = torch.sum((outputs - c) ** 2, dim=1)
                if scores_in == None:
                    scores_in = score
                else:
                    scores_in = torch.cat((scores_in, score), dim=0)
                    # n = np.array(outputs.cpu())
                    # ls.extend(n)

            scores_ood = None
            for data in test_loader_ood:
                inputs, _ = data
                inputs = inputs.to(device)
                if P.dataset == 'mnist':
                    inputs = torch.cat((inputs, inputs, inputs), 2)
                    inputs = inputs.reshape(inputs.shape[0], 3, 28, 28)
                _, outputs_aux = model(inputs, simclr=True)
                outputs = outputs_aux['simclr']
                score = torch.sum((outputs - c) ** 2, dim=1)
                if scores_ood == None:
                    scores_ood = score
                else:
                    scores_ood = torch.cat((scores_ood, score), dim=0)
                    # n = np.array(outputs.cpu())
                    # ls.extend(n)
            auroc_dict = get_auroc(scores_ood.cpu(), scores_in.cpu())

            if auroc_dict>max_score:
                max_score=auroc_dict
                # score_sum.append(auroc_dict)
                # print(auroc_dict)
        if epoch%1==0:
            logger.info(max_score)

        if epoch==P.dlr_milestones:
            logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_last_lr()[0]))
        loss_epoch = 0.0
        n_batches = 0
        epoch_start_time = time.time()
        for data in train_loader:
            inputs, _= data
            inputs = inputs.to(device)
            if P.dataset == 'mnist':
                inputs = torch.cat((inputs, inputs, inputs), 2)
                inputs= inputs.reshape(inputs.shape[0], 3, 28, 28)
            # labels=labels.to(device)

            images1, images2 = hflip(inputs.repeat(2, 1, 1, 1)).chunk(2)
            #加上旋转预测   可能会有内存爆的问题，用128/4=32
            # images1 = torch.cat([P.shift_trans(images1, k) for k in range(P.K_shift)])
            # images2 = torch.cat([P.shift_trans(images2, k) for k in range(P.K_shift)])
            # shift_labels = torch.cat([torch.ones_like(labels) * k for k in range(P.K_shift)], 0)  # B -> 4B
            # shift_labels = shift_labels.repeat(2)

            images_pair = torch.cat([images1, images2], dim=0)
            images_pair = simclr_aug(images_pair)
            # Zero the network parameter gradients
            optimizer.zero_grad()

            # Update network parameters via backpropagation: forward + backward + optimize
            _, outputs_aux = model(inputs,simclr=True,shift=True)
            _, outputs_aux2 = model(images_pair, simclr=True,shift=True)
            outputs = outputs_aux['simclr']
            dist = torch.sum((outputs - c) ** 2, dim=1)
            simclr = normalize(outputs_aux2['simclr'])
            # loss_shift = criterion(outputs_aux2['shift'], shift_labels)
            sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
            loss_sim = NT_xent(sim_matrix, temperature=0.5)
            #更改损失函数  加上对比损失

            loss1 = torch.mean(dist)
            loss = loss1 + 0.1 * loss_sim
            # loss=loss1+0.1*loss_sim+0.1*loss_shift
            # loss=loss1
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            n_batches += 1

        # log epoch statistics

                #开始画图
                # lab1 = [2]*1000
                # lab2 = np.ones(9000)
                # lab = np.append(lab1, lab2)
                #
                # result = tsne.fit_transform(ls)
                # print(len(result))
                # fig = plot_embedding(data=result, label=lab, title='111')
                # plt.show()
        scheduler.step()
        epoch_train_time = time.time() - epoch_start_time
        # print(loss_epoch / n_batches)
        logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                    .format(epoch + 1, P.svdd_epochs, epoch_train_time, loss_epoch / n_batches))

    train_time = time.time() - start_time
    logger.info('Training time: %.3f' % train_time)

    logger.info('Finished training.')
    logger.info("可以进行到这")
    # print(score_sum)
    logger.info(max_score)


    # with torch.no_grad():
    #     scores_in=None
    #     for data in test_loader:
    #         inputs, _ = data
    #         inputs = inputs.to(device)
    #         _, outputs_aux = model(inputs, simclr=True)
    #         outputs = outputs_aux['simclr']
    #         score = torch.sum((outputs - c) ** 2, dim=1)
    #         if scores_in==None:
    #             scores_in=score
    #         else:
    #             scores_in=torch.cat((scores_in,score),dim=0)
    #     print(scores_in.size())
    #     scores_ood=None
    #     for data in test_loader_ood:
    #         inputs, _ = data
    #         inputs = inputs.to(device)
    #         _, outputs_aux = model(inputs, simclr=True)
    #         outputs = outputs_aux['simclr']
    #         score = torch.sum((outputs - c) ** 2, dim=1)
    #         if scores_ood == None:
    #             scores_ood = score
    #         else:
    #             scores_ood = torch.cat((scores_ood, score), dim=0)
    #     print(scores_ood.size())
    # auroc_dict=get_auroc(scores_ood.cpu(),scores_in.cpu())
    # print(score_sum)
    # print(auroc_dict)



else:
    raise NotImplementedError()


