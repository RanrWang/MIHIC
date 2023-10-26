import os
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from utils.utils import get_lr

def fit_one_epoch(device, model, loss_history, optimizer, epoch, gen, gen_val, Epoch, save_dir,backbone):
    model = model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
    total_loss      = 0
    total_accuracy  = 0

    val_loss        = 0
    val_accuracy    = 0
    criterion       = nn.CrossEntropyLoss()
    #-------------------------------#
    #   训练
    #-------------------------------#
    print("Start Train")
    pbar = tqdm(total=len(gen), desc=f'Eopch {epoch+1}/{Epoch}',postfix=dict, mininterval=0.2)
    model.train()
    for iteration, (images, targets) in enumerate(gen):
        images  = images.to(device)
        targets =  targets.to(device)
        #----------------------#
        #   清零梯度
        #----------------------#
        optimizer.zero_grad()
        #----------------------#
        #   前向传播
        #----------------------#
        outputs     = model(images)
        #----------------------#
        #   计算损失
        #----------------------#
        loss_value  = criterion(outputs, targets)
        loss_value.backward()
        optimizer.step()
        
        total_loss += loss_value.item()
        with torch.no_grad():
            accuracy = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            total_accuracy += accuracy.item()
        pbar.set_postfix(**{'train_loss': total_loss / (iteration + 1), 
                            'accuracy'  : total_accuracy / (iteration + 1), 
                            'lr'        : get_lr(optimizer)})
        pbar.update(1)
    pbar.close()
    print('Finish Train')

    #---------------------------------#
    #   验证
    #---------------------------------#
    print('Start Validation')
    pbar = tqdm(total=len(gen_val), desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.2)
    model.eval()
    with torch.no_grad(): 
        for iteration, (images, targets) in enumerate(gen_val):
            images  = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs     = model(images)
            loss_value  = criterion(outputs, targets)
            val_loss    += loss_value.item()
            accuracy    = torch.mean((torch.argmax(F.softmax(outputs, dim=-1), dim=-1) == targets).type(torch.FloatTensor))
            val_accuracy+= accuracy.item()
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'accuracy'  : val_accuracy / (iteration + 1)})
            pbar.update(1)
        pbar.close()
    print('Finish Validation')

    #---------------------------------#
    #   保存loss
    #---------------------------------#
    loss_history.append_loss(epoch + 1, total_loss / len(gen), val_loss / len(gen_val))
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Train Loss: %.3f || Val Loss: %.3f ' % (total_loss / len(gen), val_loss / len(gen_val)))

    #-----------------------------------------------#
    #   保存 val loss 最小的权值文件
    #-----------------------------------------------#
    if len(loss_history.val_loss) <= 1 or (val_loss / len(gen_val)) <= min(loss_history.val_loss):
        print('\033[1;33;44mSave best model to best_epoch_weights.pth\033[0m')
        torch.save(model.state_dict(), os.path.join(save_dir, "{}_best_epoch_weights.pth".format(backbone)))
