import torch
import numpy as np
import random
from tqdm import tqdm
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

KG_path = 'MetaQA/'
print('Load Knowledge Graph...')
with open(KG_path + 'entities.dict') as file_e:
    lines = file_e.readlines()
    entity_dict = {}
    entityind_dict = {}
    for line in lines:
        ent, ind = line.strip().split('\t')
        entity_dict[ent] = int(ind)
        entityind_dict[int(ind)] = ent

with open(KG_path + 'relations.dict') as file_r:
    lines = file_r.readlines()
    relation_dict = {}
    relationind_dict = {}
    for line in lines:
        rel, ind = line.strip().split('\t')
        relation_dict[rel] = int(ind)
        relationind_dict[int(ind)] = rel
        
with open(KG_path + 'train.txt') as file_train:
    lines = file_train.readlines()
    train_triple_list = []
    for line in lines:
        h, r, t = line.strip().split('\t')
        train_triple_list.append([entity_dict[h], relation_dict[r], entity_dict[t]])

with open(KG_path + 'valid.txt') as file_valid:
    lines = file_valid.readlines()
    valid_triple_list = []
    for line in lines:
        h, r, t = line.strip().split('\t')
        valid_triple_list.append([entity_dict[h], relation_dict[r], entity_dict[t]])
        
with open(KG_path + 'test.txt') as file_valid:
    lines = file_valid.readlines()
    test_triple_list = []
    for line in lines:
        h, r, t = line.strip().split('\t')
        test_triple_list.append([entity_dict[h], relation_dict[r], entity_dict[t]])
print(f'Finish loading entity:{len(entity_dict)}, relation:{len(relation_dict)}, \
train triples:{len(train_triple_list)}, valid triples:{len(valid_triple_list)}, test triples:{len(test_triple_list)}')

class TransE(torch.nn.Module):
    def __init__(self, entity_num, relation_num, dim, margin):
        super(TransE, self).__init__()
        self.ent_emb = torch.nn.Embedding(num_embeddings=entity_num,
                             embedding_dim=dim,
                             device=0)
        self.rel_emb = torch.nn.Embedding(num_embeddings=entity_num,
                            embedding_dim=dim,
                            device=0)
        self.margin = margin
        self.marginloss = torch.nn.MarginRankingLoss(self.margin, reduction='mean')
        self.ent_normalization()
        
        
        
        self.rel_normalization()
    
    def dist(self, h, r, t, norm=2):
        if(norm == 2):
            return (h + r - t).square().sum(axis=1).sqrt()
        elif(norm == 1):
            return (h + r - t).abs().sum(axis=1)
    
    def forward(self, current_list, corrupt_list):
        h, r, t = torch.tensor(current_list).cuda().T.chunk(3)
        h_ = self.ent_emb(h)
        r_ = self.rel_emb(r)
        t_ = self.ent_emb(t)
        hc, rc, tc= torch.tensor(corrupt_list).cuda().T.chunk(3)
        hc_ = self.ent_emb(hc)
        rc_ = self.rel_emb(rc)
        tc_ = self.ent_emb(tc)
        return self.marginloss(self.dist(h_, r_, t_, 2), self.dist(hc_, rc_, tc_, 2), target=torch.tensor([-1]).cuda())
    
    def test_avg_rank(self, triple_list):
        test_ent_ind = torch.tensor(list(range(len(entity_dict)))).view(-1,1).cuda()
        ans = []
        for triple in tqdm(triple_list):
            h, t, r = triple
            test_ent = self.ent_emb(torch.tensor(h).cuda())
            test_rel = self.rel_emb(torch.tensor(t).cuda())
            dist = np.array((test_ent + test_rel - self.ent_emb.weight.data).square().sum(dim=1).detach().cpu())
            ans.append(dist.argsort().argsort()[r])
        return np.mean(ans)
    
    def ent_normalization(self):
        norm = self.ent_emb.weight.detach()
        self.ent_emb.weight.data.copy_(norm / norm.square().sum(dim=1).sqrt().view(-1, 1))
        
    def rel_normalization(self):
        norm = self.rel_emb.weight.detach()
        self.rel_emb.weight.data.copy_(norm / norm.square().sum(dim=1).sqrt().view(-1, 1))

model = TransE(len(entity_dict), len(relation_dict), 30, 2)

triple_sr_o_dict = {}
for triple in train_triple_list:
    if tuple(triple[:2]) in triple_sr_o_dict:
        triple_sr_o_dict[tuple(triple[:2])].append(triple[2])
    else:
        triple_sr_o_dict[tuple(triple[:2])] = [triple[2]]
triple_s_ro_dict = {}
for triple in train_triple_list:
    if tuple(triple[1:]) in triple_s_ro_dict:
        triple_s_ro_dict[tuple(triple[1:])].append(triple[0])
    else:
        triple_s_ro_dict[tuple(triple[1:])] = [triple[0]]

optim = torch.optim.Adagrad(params=model.parameters(), lr=0.1)
batch_size=400
best_mean_rank = 9999
for epoch in tqdm(range(1000)):
    indices = list(range(len(train_triple_list)))
    random.shuffle(indices)
    total_loss = 0
    model.ent_normalization()
    for i in range(0, len(indices), batch_size):
        ind = indices[i:i+batch_size]
        current_list = []
        corrupt_list = []
        for j in ind:
            current_list.append(train_triple_list[j].copy())
            flag = random.randint(0, 1)
            corrupt = train_triple_list[j].copy()
            if flag == 0:
                corrupt[0] = random.randint(0, len(entity_dict) - 1)
                while corrupt[0] in triple_s_ro_dict[tuple(corrupt[1:])]:
                    corrupt[0] = random.randint(0, len(entity_dict) - 1)
            else:
                corrupt[2] = random.randint(0, len(entity_dict) - 1)
                while corrupt[2] in triple_sr_o_dict[tuple(corrupt[:2])]:
                    corrupt[2] = random.randint(0, len(entity_dict) - 1)
            corrupt_list.append(corrupt.copy())
        loss = model(current_list, corrupt_list)
        with torch.no_grad():
            total_loss += loss
        optim.zero_grad()
        loss.backward()
        optim.step()
#     val_mean_rank = model.test_avg_rank(valid_triple_list)
#     if val_mean_rank < best_mean_rank:
#         best_ent_emb = model.ent_emb.weight.data.clone().detach()
#         best_rel_emb = model.rel_emb.weight.data.clone().detach()
#         best_mean_rank = val_mean_rank
#     print(f'epoch:{epoch+1}, total_loss:{loss}, average val rank:{val_mean_rank}, average test rank:{model.test_avg_rank(test_triple_list)},')
#     if (epoch+1) % 20 ==0:
print(f'epoch:{epoch+1}, total_loss:{total_loss}')
# print(f'average test rank:{model.test_avg_rank(test_triple_list)}')

model.test_avg_rank(test_triple_list)

model.rel_emb.weight.square().sum(dim=1)

model.rel_emb.weight

model.ent_emb.weight.data

torch.save(best_ent_emb, 'best_entity_embedding_TransE_rank_' + str(round(best_mean_rank, 2)))
torch.save(best_ent_emb, 'best_relation_embedding_TransE_rank_' + str(round(best_mean_rank, 2)))