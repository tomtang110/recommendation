from torch.nn import functional as F
import torch
from SVDPP.metrics import cal_precision_at_k,cal_Recall_at_k_for_each_user,cal_ndcg_at_k_for_each_user



def train(config,model,train_iter=None,dev_iter=None,dev_index=None):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    total_batch = 0
    dev_best_loss = float('inf')
    last_improve = 0
    flag = True
    for ep in range(config.epoch):
#         print('Epoch [{}/{}]'.format(ep+1,config.epoch))
        output = model(train_iter)
        model.zero_grad()
        loss = F.mse_loss(output,train_iter)
        loss.backward()
        optimizer.step()

        if total_batch % 50 == 0:
            dev_loss = evaluate(output,dev_index,dev_iter)
            if dev_loss < dev_best_loss:
                dev_best_loss = dev_loss
                torch.save(model.state_dict(),config.save_path+'svdpp.ckpt')
                improve = '*'
                last_improve = total_batch
            else:
                improve = ''
            print('Iter:{0:>6}, Train Loss:{1:>5.2}, Val Loss:{2:>6.2}, {3:>4.3}'.format(total_batch,loss,dev_loss,improve))
        total_batch += 1
        if total_batch - last_improve > config.require_improvement:
            print("No optimization for a long time, auto-stopping...")
            flag = True
            break





def evaluate(R_matrix,dev_index,dev_iter):

    output = R_matrix[torch.where(dev_iter==0)]
    loss =F.mse_loss(dev_iter[torch.where(dev_iter==0)],output)

    return loss

def test(config,model,test_index,K):

    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    R_matrix = model.Pre.numpy()
    prec = 0.0
    recal = 0.0
    ndgc = 0.0
    for user_ in test_index.keys():
        result = R_matrix[user_].argsort()[::-1][:K]
        prec += cal_precision_at_k(K,result,test_index[user_])
        recal += cal_Recall_at_k_for_each_user(K,result,test_index[user_])
        ndgc += cal_ndcg_at_k_for_each_user(K,result,test_index[user_])

    print('Precision: {}, Recall: {} NDCG: {}'.format(prec/len(test_index),recal/len(test_index),ndgc/len(test_index)))