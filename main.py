# coding=UTF-8
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ToolScripts.TimeLogger import log
import pickle
import os
import sys
import random
import scipy.sparse as sp
from scipy.sparse import csr_matrix
import time
from model import SAMN
import argparse
import time
from process import loadData
from BPRData import BPRData
import evaluate
import torch.utils.data as dataloader
modelUTCStr = str(int(time.time()))
device_gpu = t.device("cuda")


class Model():
    def __init__(self, args):#, train, test, cv, trust):
        self.args = args
        trainMat, testData, validData, trustMat = self.getData(args)
        self.testData = testData
        
        self.trainMat = (trainMat!=0)*1
        self.trustMat = trustMat
        self.userNum, self.itemNum = self.trainMat.shape
        assert self.trustMat.shape[0] == self.userNum
        self.hide_dim = self.args.hide_dim
        self.curEpoch = 0
        test_dataset = BPRData(testData)
        self.test_loader  = dataloader.DataLoader(test_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)
        # valid_dataset = BPRData(validData)
        # self.valid_loader  = dataloader.DataLoader(valid_dataset, batch_size=args.test_batch*101, shuffle=False, num_workers=0)


    #初始化参数
    def prepareModel(self):
        np.random.seed(self.args.seed)
        t.manual_seed(self.args.seed)
        t.cuda.manual_seed(self.args.seed)

        self.model = SAMN(self.userNum, self.itemNum, self.args.mem_size, self.args.att_size, self.hide_dim).cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.reg)

    def run(self):
        #判断是导入模型还是重新训练模型
        self.prepareModel()
        cvWait = 0
        best_HR = 0.1
        for e in range(self.curEpoch, self.args.epochs+1):
            self.curEpoch = e
            log("**************************************************************")
            log("begin train")
            epoch_loss = self.trainModel(self.trainMat, self.trustMat)
            log("end train")
            log("epoch %d/%d, epoch_loss=%.2f"% (e,self.args.epochs, epoch_loss))
            #验证
            # valid_HR, valid_NDCG = self.testModel(self.valid_loader)
            # log("epoch %d/%d, valid_HR=%.2f, valid_NDCG=%.4f"%(e, self.args.epochs, valid_HR, valid_NDCG))
            #测试
            if e > 20:
                test_HR, test_NDCG = self.testModel(self.test_loader)
                log("epoch %d/%d, test_HR=%.4f, test_NDCG=%.4f"%(e, self.args.epochs, test_HR, test_NDCG))
            else:
                test_HR, test_NDCG = 0, 0
                cvWait = 0

            
            # self.sparseTest(self.trainMat, self.testMat)

            if test_HR > best_HR:
                best_HR = test_HR
                cvWait = 0
            else:
                cvWait += 1
                log("cvWait = %d"%(cvWait))

            if cvWait == 5:
                HR, NDCG = self.testModel(self.test_loader,save=True)
                # with open(self.datasetDir + "/test_data.csv".format(self.args.cv), 'rb') as fs:
                #     test_data = pickle.load(fs)
                uids = np.array(self.testData[::101])[:,0]
                data = {}
                assert len(uids) == len(HR)
                assert len(uids) == len(np.unique(uids))
                for i in range(len(uids)):
                    uid = uids[i]
                    data[uid] = [HR[i], NDCG[i]]

                with open("SAMN-{0}-cv{1}-test.pkl".format(self.args.dataset, self.args.cv), 'wb') as fs:
                    pickle.dump(data, fs)
                break


    def trainModel(self, trainMat, trustMat):
        batch = self.args.batch
        num = trainMat.shape[0]
        shuffledIds = np.random.permutation(num)
        steps = int(np.ceil(num / batch))
        epoch_loss = 0
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            user_idx = batch_ids

            pred_pos, pred_neg = self.model(trainMat, trustMat, user_idx)

            loss = - (pred_pos.view(-1) - pred_neg.view(-1)).sigmoid().log().sum()
            
            epoch_loss += loss.item()

            loss = loss/trainMat[user_idx].nnz
            # loss = loss/batch_ids.size

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            log('setp %d/%d, step_loss = %f'%(i,steps, loss.item()), save=False, oneline=True)
        return epoch_loss
    

    def sparseTest(self, trainMat, testMat):
        interationSum = np.sum(trainMat != 0)
        flag = int(interationSum/3)
        user_interation = np.sum(trainMat != 0, axis=1).reshape(-1).A[0]
        sort_idx = np.argsort(user_interation)
        user_interation_sort = user_interation[sort_idx]
        
        tmp = 0
        idx = []
        for i in range(user_interation_sort.size):
            if tmp >= flag:
                tmp = 0
                idx.append(i)
                continue
            else:
                tmp += user_interation_sort[i]
        print("<{0}, <{1}, <{2}".format(user_interation_sort[idx[0]], \
                                            user_interation_sort[idx[1]], \
                                            user_interation_sort[-1]))
        print("{0}, {1}, {2}".format(idx[0], idx[1]-idx[0], self.userNum-idx[1]))
        splitUserIdx = [sort_idx[0:idx[0]], sort_idx[idx[0]: idx[1]], sort_idx[idx[1]:]]
        self.sparseTestModel(sort_idx)
        for i in splitUserIdx:
            self.sparseTestModel(i)

    def sparseTestModel(self, uid):
        batch = self.args.batch
        num = uid.size
        shuffledIds = uid
        steps = int(np.ceil(num / batch))
        epoch_rmse_loss = 0
        epoch_rmse_num = 0
        epoch_mae_loss = 0
        for i in range(steps):
            ed = min((i+1) * batch, num)
            batch_ids = shuffledIds[i * batch: ed]
            user_idx = batch_ids
            label = t.from_numpy(self.testMat[user_idx].data).float().to(device_gpu)

            pred = self.model(self.testMat, self.trustMat, user_idx)
            
            loss = self.loss_rmse(pred.view(-1), label.view(-1))

            epoch_rmse_loss += loss.item()
            epoch_mae_loss += t.sum(t.abs(pred.view(-1) - label)).item()
            epoch_rmse_num += self.testMat[user_idx].data.size

        rmse = np.sqrt(epoch_rmse_loss / epoch_rmse_num)
        mae = epoch_mae_loss / epoch_rmse_num

        print("sparse test : user num = %d, rmse = %.4f, mae = %.4f"%(uid.size, rmse, mae))
    
    def testModel(self, data_loader, save=False):
        HR, NDCG = [], []
        user_embed, item_embed = self.model(self.trainMat, self.trustMat, np.arange(self.userNum), isTrain=False)
        for user, item_i in data_loader:
            user = user.long().cuda()
            item_i = item_i.long().cuda()
            userEmbed = user_embed[user]
            testItemEmbed = item_embed[item_i]
            pred_i = t.sum(t.mul(userEmbed, testItemEmbed), dim=1)
            batch = int(user.cpu().numpy().size/101)
            assert user.cpu().numpy().size % 101 ==0
            for i in range(batch):
                batch_scores = pred_i[i*101: (i+1)*101].view(-1)
                _, indices = t.topk(batch_scores, self.args.top_k)
                tmp_item_i = item_i[i*101: (i+1)*101]
                recommends = t.take(tmp_item_i, indices).cpu().numpy().tolist()
                gt_item = tmp_item_i[0].item()
                HR.append(evaluate.hit(gt_item, recommends))
                NDCG.append(evaluate.ndcg(gt_item, recommends))
        if save:
            return HR, NDCG
        else:
            return np.mean(HR), np.mean(NDCG)


    def getModelName(self):
        title = "SAMN_"
        ModelName = title + dataset + "_" + modelUTCStr + \
        "_CV" + str(self.args.cv) + \
        "_reg_" + str(self.args.reg)+ \
        "_hide_" + str(self.hide_dim) + \
        "_batch_" + str(self.args.batch) + \
        "_mem_" + str(self.args.mem_size) + \
        "_att_" + str(self.args.att_size) + \
        "_lr_" + str(self.args.lr) +\
        "_top_k_" + str(self.args.top_k)
        return ModelName
    
    def getData(self, args):
        data = loadData(args.dataset, args.cv)
        if args.dataset == "Tianchi_time":
            _, _, _, _, interatctMat, testData, validData, trustMat = data
            trainMat = interatctMat
        else:
            trainMat, testData, validData, trustMat = data

        return trainMat, testData, validData, trustMat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SR-GMI main.py')
    parser.add_argument('--reg', type=float, default=0.005)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--test_batch', type=int, default=1024)

    parser.add_argument('--hide_dim', type=int, default=16)
    parser.add_argument('--mem_size', type=int, default=8)
    parser.add_argument('--att_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--early', type=int, default=1)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--seed', type=int, default=29)
    parser.add_argument('--top_k', type=int, default=10)

    parser.add_argument('--dataset', type=str, default="Epinions")
    parser.add_argument('--cv', type=int, default=1)

    args = parser.parse_args()
    args.dataset = args.dataset + "_time"
    print(args)
    dataset = args.dataset

    # trainMat, testMat, cvMat, trustMat = loadData(dataset, args.cv)
    # print("train num=%d, cv num=%d, test num=%d"%(trainMat.nnz, testMat.nnz, cvMat.nnz))

    # with open(r'D:\Work\baseline\dataset\CiaoDVD\mats\0.8_user1\train.pickle', 'rb') as fs:
    #     trainData = pickle.load(fs)

    # a = trainMat + testMat + cvMat
    # assert a.nnz == (trainMat.nnz + testMat.nnz + cvMat.nnz)

    hope = Model(args) #trainMat, testMat, cvMat, trustMat)

    modelName = hope.getModelName()
    
    print('ModelNmae = ' + modelName)

    hope.run()

