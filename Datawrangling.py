import os
import sys

def data_wrangling():
    data_dir = 'data'
    # Make directory for data
    os.makedirs('data',exist_ok=True)
    # make subdirectories for data
    os.makedirs('data/train',exist_ok=True)
    os.makedirs('data/test',exist_ok=True)

    # Move file to another directory
    classifications = ['colon_aca','colon_n','lung_aca','lung_n','lung_scc']
    directory_mappings = {'colon_aca':'colon',
                     'colon_n':'colon',
                     'lung_aca':'lung',
                     'lung_n':'lung',
                     'lung_scc':'lung'}
    name_mappings  = {'colon_aca':'colonaca',
                      'colon_n':'colonn',
                      'lung_aca':'lungaca',
                      'lung_n':'lungn',
                      'lung_scc':'lungscc'}

    for classification in classifications:
        #make directory for classification
        os.makedirs(f'data/train/{classification}',exist_ok=True)
        os.makedirs(f'data/test/{classification}',exist_ok=True)
        for i in range(1,5001):
            if i <= 5000*.8:
                dest = 'data/test'
            else:
                dest = 'data/train'
            


        
    os.rename('data.csv','data/train/data.csv')