import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

data_dir = './data/train/'
output_dir = './data/fixed_train1/'

data_dir_test = './data/test/'
output_dir_test = './data/fixed_train_no_labels/'

model_path = './classification_model/'
rmodel_path = './regression_model/'

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

def add_missing_data(df: pd.DataFrame, strip_id: int, testData: bool= False):
    print("Add missing data")
    for index, row in df.iterrows():
        nodes = row['node_id']
        length = 0
        if isinstance(nodes, list): 
            length = len(nodes)
        # print(df.head())
        if length < 15:
            frame = row['frame_number']
            for i in range(1,16):
                if i not in nodes:
                    df.at[index, 'strip_id'] = list(map(lambda x: strip_id, range(1,16)))
                    df.at[index, 'node_id'] = list(map(lambda x: x, range(1,16)))
                    cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r']
                    if not testData:
                        df.at[index, 'near'] = list(map(lambda x: row['near'][0], range(1,16)))
                        cols = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'r', 'vicon_x', 'vicon_y']
                    for s in cols:
                        l = row[s].copy()
                        l.insert(i-1, np.nan)
                        df.at[index, s] = l
                    if (len(df.at[index, 'node_id']) < 15):
                        print("Hier")


def transform_data(strip_name: str, strip_id: int):
    input_path = "%s%s.csv" % (data_dir, strip_name)
    if os.path.exists(input_path):
        
        # load strip
        strip = load_strip_csv(input_path)

        # transform strip
        strip = strip.groupby(['run_number','frame_number']).agg(pd.Series.tolist)
        strip.reset_index(inplace=True)

        # add missing data
        add_missing_data(strip, strip_id)
        print(strip.head(n=1))
        # save to output dir
        save_strip_pkl(strip, output_dir, "%s.pkl" % strip_name)
        print('Finish %s' % strip_name)
        return strip
    else: 
        print('Strip is missing %s' % strip_name)
        return None

def prepare_all_data(reprepare: bool = False):
    for i in range(1, 24):
        print('Prepare %i' % i)
        file_name = 'strip_%i_train' % i
        if reprepare or not os.path.exists('%s%s.pkl' % (output_dir, file_name)):
            transform_data(file_name, i)
        else:
            print(file_name, " already exists")


# test data

def transform_test_data(strip_name: str, strip_id: int):
    input_path = "%s%s.csv" % (data_dir_test, strip_name)
    if os.path.exists(input_path):
        
        # load strip
        strip = load_strip_csv(input_path)

        strip = strip.groupby(['frame_number']).agg(pd.Series.tolist)
        strip.reset_index(inplace=True)

        # add missing data
        add_missing_data(strip, strip_id, True)

        # save to output dir
        save_strip_pkl(strip, output_dir_test, "%s.pkl" % strip_name)
        print('Finish %s' % strip_name)
        return strip
    else: 
        print('Strip is missing %s' % input_path)
        return None

# no label data 

def load_no_label_data(reprepare: bool = False):
    for i in range(1, 24):
        file_name = "strip_%i_test_no_labels" % i
        if reprepare or not os.path.exists('%s%s.pkl' % (output_dir_test, file_name)):
            print('Prepare %i' % i)
            transform_test_data(file_name, i)
        else:
            print(file_name, " already exists")



# helper functions
def split_data(df: pd.DataFrame, test_data: bool= False) -> pd.DataFrame:
    columns = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "r"] # , 
    newColumns = range(1,16)
    df2 = pd.DataFrame()
    # df2['frame_number'] = df['frame_number']
    for c in columns:
        cols = list(map(lambda x: c+str(x), newColumns))
        tmp = df[c].to_list()
        new_df = pd.DataFrame(tmp, columns=cols)
        df2 = pd.concat([df2, new_df], axis=1)
    # add near column 
    if not test_data:
        df2['vicon_x'] = pd.DataFrame(df['vicon_x'].values.tolist()).agg('max', axis=1)
        df2['vicon_y'] = pd.DataFrame(df['vicon_y'].values.tolist()).agg('max', axis=1)
        df2['near'] = pd.DataFrame(df['near'].values.tolist()).agg('max', axis=1)
    return df2

def prepare_training_data(df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    df = split_data(df)
    columns = ["ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "r"] # "strip_id", "timestamp", 
    newColumns = range(1,16)
    for c in columns:
        cols = list(map(lambda x: c+str(x), newColumns))
        for c2 in cols:
            df[c2].fillna(df[c2].mean() if not math.isnan(df[c2].mean()) else 0, inplace=True)
    #Alle frames entfernen, wo near = 0
    df = df[df.near != 0]
    X_train = df.drop('vicon_x',axis = 1).drop('vicon_y',axis = 1).drop('near',axis = 1)
  
    X_train.fillna(X_train.mean(), inplace=True)
    Y_train = df[['vicon_x', 'vicon_y']]
    Y_train.fillna(Y_train.mean(), inplace=True)


    sc = StandardScaler()
    sc.fit(X_train)
    X_train_transformed = sc.transform(X_train)

    return X_train_transformed, Y_train

def saveModel(model, file_name):
    f = open("./regression_model/%s.bin" % file_name, "wb")
    f.write(model)
    f.close()


def readClassificationModel(strip_id: int):
    f = open("%smodel_%i.bin" % (model_path, strip_id), "rb")
    model = pickle.load(f)
    f.close()
    return model

def readRegressionModel(strip_id):
    f = open("%smodel_%i.bin" % (rmodel_path, strip_id), "rb")
    model = pickle.load(f)
    f.close()
    return model