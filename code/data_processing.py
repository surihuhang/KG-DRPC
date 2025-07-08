import pandas as pd
import numpy as np
import os
df = pd.read_csv('../Dataset/All.csv')
folder_path = "../Dataset/Disease_dataset"
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
i=0
print(csv_files)
for csv_file in csv_files:
    need = pd.read_csv('../Dataset/Disease_dataset/'+csv_file)
    index = np.array(need.iloc[:,0])

    other = df.drop(index)
    t = need.iloc[:,1:]
    other.to_csv('../Dataset/train_fd_' + str(i) + '.csv',index=False)
    t.to_csv('../Dataset/test_fd_' + str(i) + '.csv', index=False)
    i = i+1
