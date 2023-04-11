import pandas as pd
import os
from mult_stat import doPCA, doLDA, loo_cv
from helpers import Helpers as hp
from os.path import join
from datetime import datetime

path = 'C:\\Users\\sverme-adm\\Desktop\\data_split2'
eval_ts = datetime.now().strftime("_%m-%d-%Y_%H-%M-%S")
os.environ["ROOT_PATH"] = hp.mkdir_ifnotexits(join(path, 'result' + eval_ts))

"""
files = hp.get_all_files_in_dir_and_sub(path)
files = [f for f in files if f.find('chromatogramm.csv') >= 0]

results = []
info = []

i = 0
for file in files:
    # data = file.spit('\\')[-2]
    df = hp.read_file(file, skip_header=11)[
        ['Ret.Time', 'Absolute Intensity']]

    data_round = hp.round_data(df, file.split('\\')[5])
    i += 1
    results.append(data_round)
    info.append({'name': file.split('\\')[5], 'date': file.split('\\')[6]})
df_info = pd.DataFrame(info)

df_concat = hp.concat_df(results)


hp.save_df(df_concat, join(os.environ["ROOT_PATH"], 'data'), 'extracted_features')
hp.save_df(df_info, join(
    os.environ["ROOT_PATH"], 'data'), 'extracted_features_info')
    """

df = pd.read_csv(
    'extracted_features.csv', sep=';', decimal=',')
df_info = pd.read_csv(
    'extracted_features_info.csv', sep=';', decimal=',')
df_info.drop(df_info.columns[0],axis=1,inplace=True)
df.drop('time', axis=1, inplace=True)


# df_PC = doPCA(df.T, df_info)
# dfLDA = doLDA(df.T, df_info)#
loo_cv(df.T, df_info)
