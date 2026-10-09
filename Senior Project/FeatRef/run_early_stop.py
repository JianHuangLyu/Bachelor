# from torchsummary import  summary
from genericpath import isfile
import os
from pickle import NONE
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from data.mydata import MyDataset,  get_sub_num
# from utils.parameters import *
# from torchviz import make_dot
from model.mymodel import FeatRef
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score
import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
# import EarlyStopping

patience = 3
save_epoch = 0


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    
    def __init__(self, patience=7, verbose=False, delta=0, path = '', path1 = '', trace_func=print, save_epoch = 0, save_old_best_epoch = 0,sub=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.path1 = path1
        self.sub = sub
        self.save_epoch = save_epoch
        
        self.save_old_best_epoch = save_old_best_epoch
        self.trace_func = trace_func
    def __call__(self, val_loss, model, optimizer, epoch,sub,save_epoch,best_epoch):

        score = -val_loss
        epoch = epoch
        if(sub>0):
            if(epoch==0): # sub>0 才會執行此式判斷
                
                    if os.path.isfile('model'+str(best_epoch)+'.pth'):
                        print("Loading the best for last " + "model"+str(best_epoch)+ ".pth" + " ...") 
                        model.load_state_dict(torch.load('model'+str(best_epoch)+'.pth'))
                            
                                    
                    if os.path.isfile('optimizer'+str(best_epoch)+'.pth'):
                        print("Loading the best for last " + "optimizer"+str(best_epoch)+ ".pth" + " ...")
                        optimizer.load_state_dict(torch.load('optimizer'+str(best_epoch)+'.pth'))
                

        if(score!=0):
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model, optimizer,epoch,sub,save_epoch)
                
            elif score > self.best_score + self.delta:
                self.best_score = score
                self.save_checkpoint(val_loss, model, optimizer,epoch,sub,save_epoch)
                #self.counter = 0
            best_epoch = save_epoch
            return best_epoch
                    
            #elif score < self.best_score + self.delta:
                #self.counter += 1
                #self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                
                #if self.counter >= self.patience:
                    #self.early_stop = True
                #if val_loss >= (abs(self.best_score - 0.60)) : #
                    #print("Because of the Val_Loss >=" + str(abs(self.best_score - 0.60)) + " , End of this epoch trainning ...")
                    #self.early_stop = True
                #if self.counter >= self.patience:
                    #self.counter = 0
       
    def save_checkpoint(self, val_loss, model, optimizer, epoch,sub, save_epoch):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving ' + 'model ' + str(epoch) + ' and optimizer ' + str(epoch)+ ' ...')
            self.save_epoch = epoch #將最好的epoch編號模型儲存
            save_epoch = epoch

            self.valloss = val_loss 
            if((val_loss < self.val_loss_min) and  (epoch!=0)):
                os.remove('model' + str(self.save_old_best_epoch) + '.pth')
                os.remove('optimizer' + str(self.save_old_best_epoch) + '.pth')
            self.save_old_best_epoch = self.save_epoch
        
        self.path = 'model' + str(self.save_epoch) + '.pth'
        self.path1 = 'optimizer' + str(self.save_epoch) + '.pth'
        torch.save(model.state_dict(), self.path)
        torch.save(optimizer.state_dict(), self.path1)
        if((self.valloss > val_loss) and (sub>0)):
            torch.save(model.state_dict(), self.path)
            torch.save(optimizer.state_dict(), self.path1)
        
        #if((val_loss < self.val_loss_min) and  (epoch!=0)):
            #if os.path.isfile('model'+str(self.save_epoch)+'.pth'):
                    #model.load_state_dict(torch.load('model'+str(self.save_epoch)+'.pth'))
                    #print("Loading the best for last " + "model"+str(self.save_epoch)+ ".pth" + " ...") 
                            
            #if os.path.isfile('optimizer'+str(self.save_epoch)+'.pth'):
                #optimizer.load_state_dict(torch.load('optimizer'+str(self.save_epoch)+'.pth'))
                #print("Loading the best for last " + "optimizer"+str(self.save_epoch)+ ".pth" + " ...")
        self.val_loss_min = val_loss
        return save_epoch


parser = argparse.ArgumentParser(description='PyTorch Cifar Training')
parser.add_argument('--dataset', default='orignal flow ', help='dataset setting')
parser.add_argument('--path', default='data/database/', help='dataset path')
parser.add_argument('--root_path', default='', help='project root path')
parser.add_argument('--result_path', default='result/', help='result path')
parser.add_argument('--result_metrics', default='metric_samm.csv', help='result metrics file name')
parser.add_argument('--ev_dim', default=1024, type=int, metavar='N',
                    help='dimension of feature')
parser.add_argument('--width', default=28, type=int, metavar='N',
                    help='sample width')
parser.add_argument('--class_num', default=4, type=int, metavar='N',
                    help='number of total classes')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='number of total epochs to run') 
parser.add_argument('--train_round', default=1, type=int, metavar='N',
                    help='number of total train round')
parser.add_argument('--batch_size', default=4, type=int, metavar='N',
                    help='number of batch size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--lambada', default=0.9, type=float, metavar='N',
                    help='weight for rois loss')

# acc
train_num_list = []
# acc
full_acc_list = []
# uf1
full_uf1_list = []
# uar
full_uar_list = []
# matrix
full_matrix_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from torchviz import make_dot
def save_parameters(args):
    file_path = args.result_path + '/' + args.result_metrics.replace('.csv', '.txt')
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    file = open(file_path, 'w+')
    pram_str = '--dataset: ' + str(args.dataset) + '\n' + \
               '--ev_dim: ' + str(args.ev_dim) + '\n' + \
               '--path: ' + str(args.path) + '\n' + \
               '--result_path:' + str(args.result_path) + '\n' + \
               '--result_metrics:' + str(args.result_metrics) + '\n' + \
               '--width:' + str(args.width )+ '\n' + \
               '--class_num' + str(args.class_num) + '\n' + \
               '--epochs' + str(args.epochs) + '\n' + \
               '--train_round' + str(args.train_round )+ '\n'
    file.write(pram_str)
    file.close()
# acc
train_num_list = []
# acc
full_acc_list = []
casme2_acc_list = []
samm_acc_list = []
smic_acc_list = []

# uf1
full_uf1_list = []
casme2_uf1_list = []
samm_uf1_list = []
smic_uf1_list = []

# wf1
full_wf1_list = []
casme2_wf1_list = []
samm_wf1_list = []
smic_wf1_list = []

# f1
full_f1_list = []
casme2_f1_list = []
samm_f1_list = []
smic_f1_list = []

# uar
full_uar_list = []
casme2_uar_list = []
samm_uar_list = []
smic_uar_list = []

# matrix
full_matrix_list = []
casme2_matrix_list = []
samm_matrix_list = []
smic_matrix_list = []
def draw_model(model):
    print(model)
    x1 = torch.randn(6,9,1,1).requires_grad_(True)
    x1 = x1.to(device)
    y = model(x1)
    # vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x1', x1)]+ [('x2', x1)]))
    vis_graph = make_dot(y)
    vis_graph.view()

def evaluate_metrics(predict_label, true_label, num, args):
    casme2_label_num=10
    samm_label_num=20
    #smic_label_num=25
    full_acc = accuracy_score(true_label, predict_label)

    full_acc_list.append(full_acc)

    # uf1 macro_f1
    full_uf1 = f1_score(true_label, predict_label, average='macro')

    full_uf1_list.append(full_uf1)

    # uf1 weighted_f1
    full_wf1 = f1_score(true_label, predict_label, average='weighted')

    full_wf1_list.append(full_wf1)
   # matrix
    confusion_matrixs = confusion_matrix(true_label, predict_label)

    full_matrix_list.append(confusion_matrixs)

    d = np.diag(confusion_matrixs)
    m = np.sum(confusion_matrixs, axis=1)
    uar = np.sum(d / m / args.class_num)

    print('full_acc:', full_acc_list)
    # print('full_f1:', full_f1)
    print('full_uf1:', full_uf1)
    print('full_wf1:', full_wf1)
    print('full_uar:', uar)

    print('full_cm:', confusion_matrixs)

    full_uar_list.append(uar)

    train_num_list.append(num)
    df = pd.DataFrame(data={'train_num': train_num_list,
                            'full_acc_list': full_acc_list, 'full_uf1_list': full_uf1_list,
                            'full_uar_list': full_uar_list, 'full_wf1_list:': full_wf1_list,
                            'full_matrix_list': full_matrix_list
                            })

    df.to_csv(args.result_path + args.result_metrics)

def train_flow():
    
    args = parser.parse_args()
    save_parameters(args)
    print(args)
    # model_path = args.root_path + '/result/mi_gcn_test/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sub_num = get_sub_num(args.path)
    # print(model)
    
    for round_time in range(args.train_round):
        
        print(args)
        numth_test_sample_list = []
        predict_label_list = []
        true_label_list = []
        acc_list = []
        early_stopping = EarlyStopping(patience=patience, verbose=True)
       
        
        
        for sub in range(sub_num):
            sub = sub
            print(args)
            model = FeatRef(args.class_num, args).to(device)
            print("gcn model have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
            print('Loading new model for new sub...')
            #print(model)
            #draw_model(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            print('Loading new optimizer for new sub...')
            loss_func = torch.nn.CrossEntropyLoss()
            train_data = MyDataset(args.path, 'train', sub, model='3d', width=args.width, class_num=args.class_num)
            train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
            val_data = MyDataset(args.path, 'val', sub, model='3d', width=args.width, class_num=args.class_num)
            val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
            test_data = MyDataset(args.path, 'test', sub, model='3d', width=args.width, class_num=args.class_num)
            test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
            for index in range(np.shape(test_data.test_sample_list)[0]):
                numth_test_sample_list.append(test_data.test_sample_list[index])
            
            
            
            print('********subject',sub, ':', test_data.test_sub[sub])
            print('********val subject',sub, ':', test_data.val_sub[sub])
            # 保存验证集上acc最高的模型作为测试模型,或者loss最低的模型
            best_val_acc = 0
            best_val_loss = 100
            tic = time.time()

            epoch_list = []
            train_loss_list=[]
            train_acc_list=[]
            val_loss_list=[]
            val_acc_list=[]

            for epoch in range(args.epochs):
                if(sub>0):
                    early_stopping( 0, model, optimizer , epoch, sub,0,0)
                    
                #if(epoch!=0):
                 #   if os.path.isfile('model'+str(epoch)+'.pth'):
                  #      model.load_state_dict(torch.load('model.pth'))
                   #     if((epoch+1)==2):
                    #        print("\n")
                     #       print("Loading the best for last model ...")
                            
                   # if os.path.isfile('optimizer.pth'):
                    #    optimizer.load_state_dict(torch.load('optimizer'+str(epoch)+'.pth'))
                     #   if((epoch+1)==2):
                      #      print("Loading the best for last optimizer ...")

                print('epoch {}'.format(epoch + 1))

                

                # training
                epoch_list.append(epoch + 1)
                train_loss = 0
                train_acc = 0.0

                model.train()
                
                c_train = 0
                
                
                for i, data in enumerate(train_loader):
                    
                    batch_x1, batch_y = data
                    batch_x1, batch_y = Variable(batch_x1), Variable(batch_y)
                    batch_x1,  batch_y = batch_x1.to(device), batch_y.to(device)
                    out = model(batch_x1)
                    loss = loss_func(out, batch_y)
                    train_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    train_correct = (pred==batch_y).sum()
                    train_acc += train_correct.item()
                    optimizer.zero_grad()
                    loss_rois = 0
                    loss.backward()
                    optimizer.step()
                    c_train = c_train+1

                     
                # evalutation
                model.eval()
                val_loss = 0
                val_acc = 0
                Acc = 0
                c = 0  #count
                
                for i, data in enumerate(val_loader):
                    batch_x1, batch_y = data
                    batch_x1, batch_y = Variable(batch_x1), Variable(batch_y)
                    batch_x1, batch_y = batch_x1.to(device),  batch_y.to(device)
                    out = model(batch_x1)
                    loss = loss_func(out, batch_y)
                    val_loss += loss.item()
                    pred = torch.max(out, 1)[1]
                    val_correct = (pred == batch_y).sum()
                    val_acc += val_correct.item()
                    Acc = float(val_acc) / (len(val_data))
                    c = c + 1
                print('Train_Loss: {:.6f}, Train_Acc: {:.6f}'.format(train_loss / c_train, float(train_acc) / (len(train_data))),\
                      'Val_Loss: {:.6f}, Val_Acc: {:.6f}'.format(val_loss / c, float(val_acc) / (len(val_data))))

                
                train_loss_list.append(train_loss / c_train)
                train_acc_list.append(float(train_acc) / (len(train_data)))
                val_loss_list.append(val_loss / c)
                val_acc_list.append(float(val_acc) / (len(val_data)))
                
                early_stopping(val_loss / c,model,optimizer,epoch,sub,save_epoch,NONE)
                  
                    
                if best_val_loss > (val_loss/c):
                    best_val_loss = (val_loss/c)
                    model.eval()
                    test_loss = 0
                    # best_val_acc = 0
                    test_acc = 0
                    Acc = 0
                    c = 0
                    cur_predict_label_list = []
                    cur_true_label_list = []
                    for i, data in enumerate(test_loader):
                        batch_x1, batch_y = data
                    # for batch_x1, batch_y, rois_label in test_loader:
                        batch_x1,  batch_y = Variable(batch_x1), Variable(batch_y)
                        # print('test label:', batch_y)
                        batch_x1, batch_y = batch_x1.to(device), batch_y.to(device)
                        out = model(batch_x1)
                        loss = loss_func(out, batch_y)
                        test_loss += loss.item()
                        pred = torch.max(out, 1)[1]
                        test_correct = (pred == batch_y).sum()
                        test_acc += test_correct.item()
                        c = c+1
                        Acc = float(test_acc) / (len(test_data))
                        pred = pred.cpu().numpy()
                        batch_y = batch_y.cpu().numpy()
                        cur_predict_label_list += list(pred)
                        cur_true_label_list += list(batch_y)
                    print('Test Loss: {:.6f}, Test Acc: {:.6f}'.format(test_loss / c, Acc))

                if early_stopping.early_stop:
                    print("Early stopping")
                    early_stopping = EarlyStopping(patience=patience, verbose=True)
                    
                    break
                
            early_stopping = EarlyStopping(patience=patience, verbose=True)   
            toc = time.time()
            # print(toc-tic)
            # 计算所有subject的pre与ture
            true_label_list += list(cur_true_label_list)
            predict_label_list += list(cur_predict_label_list)
            temp_acc = accuracy_score(np.array(cur_true_label_list), np.array(cur_predict_label_list))
            acc_list.append(temp_acc)
            acc_avg = accuracy_score(np.array(true_label_list), np.array(predict_label_list))
            
            #y_pred=["ant", "ant", "cat", "cat", "ant", "cat"]   #預測
            #y_true=["cat", "ant", "cat", "cat", "ant", "bird"]   #真實

            #C=confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
            

            
            print('_______________________________________________')
            print('round time:', round_time)
            print('subject:', sub)
            print('test acc:', temp_acc)
            print(sub + 1, 'subjects avg_acc:', acc_avg)
            print('test samples:', np.shape(test_data.test_y_data)[0])
            print(np.array(cur_true_label_list))
            print(np.array(cur_predict_label_list))
            print('真實: ')
            print(true_label_list)
            print('預測: ')
            print(predict_label_list)
            print("sub" + str(sub+1) + " end ...")
            #print("Remove the best model ...")
            #print("Remove the best optimizer ...")
            #os.remove('model.pth')
            #os.remove('optimizer.pth')
            print('_______________________________________________')

            


            if sub %8 == 0:
                fig = plt.figure()
                plt.plot(epoch_list, train_loss_list, color='blue')
                plt.plot(epoch_list, val_loss_list, color='red')
                plt.legend(['Train Loss', 'val Loss'], loc='upper right')
                plt.xlabel('number of training examples seen')
                plt.ylabel('negative log likelihood loss')
                plt.savefig('./result/'+str(sub)+'_loss.png')
                fig = plt.figure()
                plt.plot(epoch_list, train_acc_list, color='blue')
                plt.plot(epoch_list, val_acc_list, color='red')
                plt.legend(['Train acc', 'val acc'], loc='upper right')
                plt.xlabel('number of training examples seen')
                plt.ylabel('average accuracy')
                plt.savefig('./result/'+str(sub)+'_accuracy.png')
                plt.legend.cla()
                #y_pred = ["0","0","1"]
                #y_true = ["0","0","0"]
                   #預測
                #y_true+= predict_label_list   #真實
                #print(true_label_list)

                
            del (train_data, val_data, test_data, train_loader, val_loader, test_loader)
            
            del(model)
            
        

        

        round_result_dir = args.result_path + '/round_' + str(round_time)
        if (os.path.exists(round_result_dir)):
            shutil.rmtree(round_result_dir)  #移除資料夾
        os.makedirs(round_result_dir)    #創建
        # save acc_list
        acc_list = np.array(acc_list)
        result_f = open(round_result_dir + '/acc_' + str(round_time) + '.txt', 'w')  
        #打開一個文件只用於寫入，如果該文件已存在則打開文件，並從開頭開始編輯，原有內容會被刪除。如果該文件不存在，創建新文件。
        result_f.write(str(acc_list))
        result_f.close()
        # save_predict_label
        predict_label_list = np.array(predict_label_list)
        result_f = open(round_result_dir + '/predict_' + str(round_time) + '.txt', 'w')
        result_f.write(str(predict_label_list))
        result_f.close()
        # save_true label
        true_f = open(round_result_dir + '/true_' + str(round_time) + '.txt', 'w')
        true_label_list = np.array(true_label_list)
        true_f.write(str(true_label_list))
        true_f.close()
        # save file names
        file_f = open(round_result_dir + '/file_names' + str(round_time) + '.txt', 'w')
        # numth_test_sample_list = np.ndarray(numth_test_sample_list)
        numth_test_sample_list = np.array(numth_test_sample_list)

        # print(numth_test_sample_list)
        file_f.write(str(numth_test_sample_list))
        file_f.close()
        evaluate_metrics(predict_label_list, true_label_list, round_time,args)

        numth_test_sample_list = numth_test_sample_list.tolist()
        true_label_list = true_label_list.tolist()
        predict_label_list = predict_label_list.tolist()
       
        
        da = pd.DataFrame(data={'file_list': numth_test_sample_list, 'true_label': true_label_list,
                                'predict_label': predict_label_list
                                })
        da.to_csv(round_result_dir + '/results_' + str(round_time) + '.csv')
        



if __name__ == '__main__':
    # train()
    train_flow()
