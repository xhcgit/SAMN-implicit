import torch as t
import torch.nn as nn
import numpy as np

class SAMN(nn.Module):
    def __init__(self, userNum, itemNum, mem_size, att_size, hide_dim):
        super(SAMN, self).__init__()
        self.userEmbed = nn.Embedding(userNum, hide_dim)
        self.itemEmbed = nn.Embedding(itemNum, hide_dim)
        nn.init.xavier_normal_(self.userEmbed.weight)
        nn.init.xavier_normal_(self.itemEmbed.weight)
        self.hide_dim = hide_dim
        self.mem_size = mem_size
        self.att_size = att_size

        key = t.empty(hide_dim, mem_size, dtype=t.float32)
        nn.init.xavier_normal_(key)
        self.key = nn.Parameter(key, requires_grad=True)

        mem = t.empty(mem_size, hide_dim, dtype=t.float32)
        nn.init.xavier_normal_(mem)
        self.mem = nn.Parameter(mem, requires_grad=True)

        self.w1 = nn.Linear(hide_dim, att_size)
        nn.init.xavier_normal_(self.w1.weight)

        self.w3 = nn.Linear(hide_dim, att_size)
        nn.init.xavier_normal_(self.w3.weight)

        b = t.empty(1, att_size, dtype=t.float32)
        nn.init.zeros_(b)
        self.b = nn.Parameter(b, requires_grad=True)

        self.h = nn.Linear(att_size, 1, bias=False)
        nn.init.xavier_normal_(self.h.weight)
    
    #return negative item id
    def negSample(self, trainMat, uid):
        userNum, itemNum = trainMat.shape
        uidx, iidx = trainMat[uid].nonzero()
        tmp_trainMat = trainMat[uid].todok()
        length = iidx.size
        neg_data = np.random.randint(low=0, high=itemNum, size=length)
        for i in range(length):
            tmp_uid = uidx[i]
            tmp_iid = neg_data[i]
            if (tmp_uid, tmp_iid) in tmp_trainMat:
                while (tmp_uid, tmp_iid) in tmp_trainMat:
                    tmp_iid = np.random.randint(low=0, high=itemNum)
                neg_data[i] = tmp_iid
        return neg_data

    
    def forward(self, trainMat, trustMat, uid, isTrain=True):
        #neg sample
        if isTrain:
            item_neg_idx = self.negSample(trainMat, uid)
        uid_t = t.from_numpy(uid).long().cuda()

        user_idx = trainMat[uid].tocoo().row
        item_pos_idx = trainMat[uid].tocoo().col

        # for i,j in zip(uid[user_idx], item_neg_idx):
        #     assert trainMat[i, j] == 0

        trust_num = np.sum(trustMat[uid], axis=1).A.reshape(-1)
        # u_e = self.userEmbed(uid_t)

        trustid = uid_t[trustMat[uid].tocoo().row]
        trusteeid = t.from_numpy(trustMat[uid].tocoo().col).long().cuda()

        trust_e = self.userEmbed(trustid)
        trustee_e = self.userEmbed(trusteeid)

        norm = t.norm(trust_e, dim=1, keepdim=True) * t.norm(trustee_e, dim=1, keepdim=True)
        s = (trust_e * trustee_e) / norm
        a = t.softmax(t.mm(s, self.key), dim=1)

        F = trustee_e.view(-1, 1, self.hide_dim) * self.mem.view(1,-1,self.hide_dim)

        f = t.sum(F * a.unsqueeze(2), dim=1)

        beta = self.h(t.relu(self.w1(trust_e) + self.w3(f) + self.b)).view(-1)

        start = 0
        end = 0
        tmp = []
        for i in trust_num:
            start = end
            end += i
            att = t.softmax(beta[int(start): int(end)], dim=0).view(-1, 1)
            friend_embed = t.sum(f[int(start): int(end)] * att, dim=0)
            tmp.append(friend_embed)

        final_friend_embed = t.stack(tmp)

        if isTrain:
            user_embed = final_friend_embed[user_idx] + self.userEmbed(t.LongTensor(uid[user_idx]).cuda())
            item_pos_embed = self.itemEmbed(t.LongTensor(item_pos_idx).cuda())
            pred_pos = t.sum(user_embed*item_pos_embed, dim=1).view(-1)
            item_neg_embed = self.itemEmbed(t.LongTensor(item_neg_idx).cuda())
            pred_neg = t.sum(user_embed*item_neg_embed, dim=1).view(-1)
            return pred_pos, pred_neg
        else:
            user_embed = final_friend_embed + self.userEmbed(t.LongTensor(uid).cuda())
            return user_embed, self.itemEmbed.weight
        

