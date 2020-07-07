from GCN_RE import GCNRE
import sys
import os

if __name__ == '__main__':
    data_name = ['TacRED']
    data_train_TAC = "./train_balance.json"
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    GCNRE.train_and_save(dataset=data_train_TAC, saving_dir='./data/TACRED/eval', data_name = data_name[0] ,
                         epochs=15, bucket_size = 100)