import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from ampligraph.latent_features import ComplEx,TransE,DistMult
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from deepctr.models import NFM
from deepctr.feature_column import SparseFeat,DenseFeat,get_feature_names
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.optimizers import Adam,Adagrad,Adamax
from sklearn.decomposition import PCA
from tensorflow import keras


#load data
data_all = pd.read_csv('../KG_data/KG_all.txt',delimiter='\t',header=None)
data_all.columns = ['head','relation','tail']

#kg
kg1 = pd.read_csv('../KG_data/Cell_Property.txt',delimiter='\t',header=None)
kg2 = pd.read_csv('../KG_data/Drug_Property.txt',delimiter='\t',header=None)
kg3 = pd.read_csv('../KG_data/Gene_GO.txt',delimiter='\t',header=None)
kg = pd.concat([kg1,kg2,kg3])

kg.index = range(len(kg))
kg.columns = ['head','relation','tail']

#for nfm input
head_le = LabelEncoder()
tail_le = LabelEncoder()
head_le.fit_transform(data_all['tail'].values)
tail_le.fit_transform(data_all['head'].values)

mms = MinMaxScaler(feature_range=(0,1))

#descriptors preparation
Cell_copynmb = pd.read_csv('../KG_data/Cell_feature.csv')
Cell_id = Cell_copynmb['name']
Cell_feats = np.array(Cell_copynmb.iloc[:,1:])

Cell_feats_scaled = mms.fit_transform(Cell_feats)
Cell_feats_scaled2 = PCA(n_components=100).fit_transform(Cell_feats_scaled)
Cell_feats_scaled3 = mms.fit_transform(Cell_feats_scaled2)

fp_df = pd.read_csv('../KG_data/finger__out.csv')
Celldes_df = pd.concat([Cell_id,pd.DataFrame(Cell_feats_scaled3)],axis=1)



#Function
################################################################

# If you want to test other scenarios, just change the data path.
# But it should be noted that the hypermeters in nfm need to be adjusted.

data_path = './data_fold/'

def load_data(i):
    train = pd.read_csv(data_path+'train_fd_'+str(i)+'.csv')[['head','relation','tail','label']]
    train_pos = train[train['label']==1]
    test = pd.read_csv(data_path+'test_fd_'+str(i)+'.csv')[['head','relation','tail','label']]
    data = pd.concat([train_pos,kg])[['head','relation','tail']]
    return train,train_pos,test,data

def roc_auc(y,pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, pred)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def pr_auc(y,pred):
    precision, recall, thresholds = metrics.precision_recall_curve(y, pred)
    pr_auc = metrics.auc(recall, precision)
    return pr_auc

def get_scaled_embeddings(model,train_triples,test_triples,get_scaled,n_components):
    [train_sub_embeddings,test_sub_embeddings] = [model.get_embeddings(x['head'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    [train_obj_embeddings,test_obj_embeddings] = [model.get_embeddings(x['tail'].values, embedding_type='entity') for x in [train_triples,test_triples]]
    train_feats = np.concatenate([train_sub_embeddings,train_obj_embeddings],axis=1)

    test_feats = np.concatenate([test_sub_embeddings,test_obj_embeddings],axis=1)
    train_dense_features = mms.fit_transform(train_feats)
    test_dense_features = mms.transform(test_feats)
    if get_scaled:
        pca = PCA(n_components=n_components)
        scaled_train_dense_features = pca.fit_transform(train_dense_features)
        scaled_pca_test_dense_features = pca.transform(test_dense_features)
    else:
        scaled_train_dense_features = train_dense_features
        scaled_pca_test_dense_features = test_dense_features
    return scaled_train_dense_features,scaled_pca_test_dense_features


def get_features(data,fp_df,Celldes_df,use_Cell):
    drug_features = pd.merge(data,fp_df,how='left',left_on='head',right_on='drugbank_id').iloc[:,4:885].values
    Cell_features = pd.merge(data,Celldes_df,how='left',left_on='tail',right_on='name').iloc[:,4:105].values
    if use_Cell:
        feature = np.concatenate([drug_features,Cell_features],axis=1)
    else:
        feature = drug_features

    return feature

def get_nfm_input(re_train_all,re_test_all,train_feats,test_feats,train_des,test_des,embedding_dim,pca_components):
    train_all_feats = np.concatenate([train_feats,train_des],axis=1)
    test_all_feats = np.concatenate([test_feats,test_des],axis=1)
    train_all_feats_scaled = mms.fit_transform(train_all_feats)
    test_all_feats_scaled = mms.transform(test_all_feats)
    feature_columns = [SparseFeat('head',re_train_all['head'].unique().shape[0],embedding_dim=embedding_dim),
                        SparseFeat('tail',re_train_all['tail'].unique().shape[0],embedding_dim=embedding_dim),
                        DenseFeat("feats",train_all_feats_scaled.shape[1]),
                        #DenseFeat("des",train_des.shape[1])
                        ]
    train_model_input = {'head':head_le.fit_transform(re_train_all['head'].values),
                    'tail':tail_le.fit_transform(re_train_all['tail'].values),
                     'feats':train_all_feats_scaled,
                     #'des':train_des
                    }
    test_model_input = {'head':head_le.fit_transform(re_test_all['head'].values),
                    'tail':tail_le.fit_transform(re_test_all['tail'].values),
                    'feats':test_all_feats_scaled,
                    # 'des':test_des
                    }
    return feature_columns,train_model_input,test_model_input

def train_nfm(feature_columns,train_model_input,train_label,test_model_input,y,patience):
    re_model = NFM(feature_columns,feature_columns,task='binary',dnn_hidden_units=(128,128),
                    l2_reg_dnn=1e-7,l2_reg_linear=1e-7,
                    )
    re_model.compile(Adam(1e-6), "binary_crossentropy",
                metrics=[keras.metrics.Precision(name='precision'),], )
    es = EarlyStopping(monitor='loss',patience=patience,min_delta=0.0001,mode='min',restore_best_weights=True)

    data = pd.DataFrame(train_model_input['feats'],columns=None,index=None)
    data = data.fillna(0)
    train_model_input['feats'] = np.array(data)


    re_model.fit(train_model_input, train_label,
                        batch_size=2000, epochs=100,
                        verbose=2,
                        callbacks=[es]
                        )
    data = pd.DataFrame(test_model_input['feats'], columns=None, index=None)
    data = data.fillna(0)
    test_model_input['feats'] = np.array(data)

    pred_y = re_model.predict(test_model_input, batch_size=512)

    roc_nfm = roc_auc(y,pred_y[:,0])
    pr_nfm = pr_auc(y,pred_y[:,0])

    return roc_nfm,pr_nfm,pred_y[:,0]

def train(i,embedding_dim,n_components,use_cell,patience):

    train,train_pos,test,data = load_data(i)
    model = DistMult(batches_count=1,
        seed=1,
        epochs=50,
        k=500,
        embedding_model_params={'corrupt_sides':'o'},
        optimizer='adam',
        optimizer_params={'lr':1e-5},
        loss='pairwise', #pairwise
        regularizer='LP',
        regularizer_params={'p':3, 'lambda':1e-5},
        verbose=True)


    model.fit(data.values, early_stopping =True,early_stopping_params=
                {
                    'x_valid': train_pos[['head','relation','tail']].values,       # validation set, here we use training set for validation
                    'criteria':'mrr',         # Uses mrr criteria for early stopping
                    'burn_in': 10,              # early stopping kicks in after 10 epochs
                    'check_interval':2,         # validates every 2th epoch
                    'stop_interval':3,           # stops if 3 successive validation checks are bad.
                    'x_filter': train_pos[['head','relation','tail']].values,          # Use filter for filtering out positives
                    'corrupt_side':'o'         # corrupt object (but not at once)
                })

    columns = ['head','relation','tail']
    test_score = model.predict(test[columns])
    test_label = test['label'].values


    roc = roc_auc(test_label,test_score)
    pr = pr_auc(test_label,test_score)
    # print("DistMult:",roc)
    # print("Distmult",pr)

    re_train_all = train[columns]
    re_test_all = test[columns]
    train_label = train['label']

    train_dense_features,test_dense_features = get_scaled_embeddings(model,re_train_all,re_test_all,False,n_components)

    train_des = get_features(re_train_all,fp_df,Celldes_df,use_cell)
    test_des = get_features(re_test_all,fp_df,Celldes_df,use_cell)

    feature_columns,train_model_input,test_model_input = get_nfm_input(re_train_all,re_test_all,
                                                                    train_dense_features,test_dense_features,
                                                                    train_des,test_des,
                                                                    embedding_dim,n_components)

    roc_nfm, pr_nfm, pred_y = train_nfm(feature_columns, train_model_input, train_label, test_model_input, test_label,
                                        patience)

    y_pred = (pred_y >= 0.5).astype(int)
    acc = metrics.accuracy_score(test_label, y_pred)
    precision = metrics.precision_score(test_label, y_pred)
    recall = metrics.recall_score(test_label, y_pred)
    f1 = metrics.f1_score(test_label, y_pred)

    return roc, pr, roc_nfm, pr_nfm, acc, precision, recall, f1


#train and test
################################################################

for i in range(30):
    roc,pr,roc_s,pr_s,acc,precision,recall,f1 = train(i,50,200,True,10)
    stable_metrics = pd.DataFrame()
    stable_metrics['ROC'] = [roc_s]
    stable_metrics['AUPR'] = [pr_s]
    stable_metrics['ACC'] = [acc]
    stable_metrics['Precision'] = [precision]
    stable_metrics['Recall'] = [recall]
    stable_metrics['F1'] = [f1]
    print(stable_metrics)

