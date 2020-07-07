from GCN_RE import GCNRE

if __name__ == '__main__':
        checkpoint = 14
        data_name = ['TacRED']
        RE_file = './data/TACRED/eval/gcn-re-param-'+str(checkpoint)+'.pkl'
        RE = GCNRE()
        RE.test(RE_filename=RE_file, dataset='./test_mini.json', data_name = data_name[0], threshold=0.6, batch_size = 32)