import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

data_dir = '../data/train/'
output_dir = '../data/fixed_train1/'

def load_strip_pkl(path: str)-> pd.DataFrame: 
    return pd.read_pickle(path)

def save_strip_pkl(data: pd.DataFrame, outdir: str, outname: str):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)    

    data.to_pickle(fullname)

def load_strip_csv(path: str)-> pd.DataFrame: 
    return pd.read_csv(path, sep=',')

def save_strip_csv(data: pd.DataFrame, outdir: str, outname: str):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    fullname = os.path.join(outdir, outname)    

    data.to_csv(fullname, sep=',')

def add_missing_data(df: pd.DataFrame):
    tmp = df[['run_number', 'frame_number', 'node_id']]
    near = pd.DataFrame(df['near'].values.tolist()).mean(1)
    print("Add missing data")
    for index, row in tmp.iterrows():
        nodes = row['node_id']
        length = 0
        if isinstance(nodes, list): 
            length = len(nodes)
        if length < 15:
            run = row['run_number']
            frame = row['frame_number']
            #print("Missing val in %i %i" % (run, frame))
            for i in range(1,16):
                if i not in nodes:
                    df.loc[-1] = [run, frame, np.nan, i, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, near, np.nan, np.nan]



def transform_data(strip_name: str):
    input_path = "%s%s.csv" % (data_dir, strip_name)
    if os.path.exists(input_path):
        
        # load strip
        strip = load_strip_csv(input_path)

        # drop transform strip
        strip = strip.groupby(['run_number','frame_number']).agg(pd.Series.tolist)
        strip.reset_index(inplace=True)

        # add missing data
        add_missing_data(strip)

        # save to output dir
        save_strip_pkl(strip, output_dir, "%s.pkl" % strip_name)
        print('Finish %s' % strip_name)
        return strip
    else: 
        print('Strip is missing %s' % strip_name)
        return None


def prepare_all_data(reprepare: bool = False):
    for i in range(1, 24):
        file_name = 'strip_%i_train' % i
        if reprepare or not os.path.exists('%s%s.pkl' % (output_dir, file_name)):
            transform_data(file_name)
        else:
            print(file_name, " already exists")
