import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, ParameterGrid, StratifiedKFold, LeaveOneOut
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import matplotlib
from sklearn.neighbors import KNeighborsClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve,auc
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
import xgboost as xgb
from sklearn import svm
import pandas as pd
import math
SVM = svm.SVC
from sklearn.ensemble import RandomForestClassifier as RF, RandomForestClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from GIS_yyzw import MaxEnt
from print_data import print_data
import jackknife
import os

from ctrl_models import xlf_dict

def get_corr(x_data, corr_csv_filename='./corr/x_corr.csv', corr_img_filename='./corr/corr.png' ):
    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[: ,4:])
    if not os.path.exists('./corr/'):
        os.mkdir('./corr')

    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min()) / (x_data.max() - x_data.min())
    x_max_min = pd.DataFrame(x_max_min)
    x_corr = x_max_min.corr().abs()

    font0 = {
    'weight' : 'medium',
    'size' : 21,
    "fontweight":'bold',
    }
    font1 = {
    'weight' : 'medium',
    'size' : 13,
    "fontweight":'bold',
    }

    x_corr_df = pd.DataFrame(np.array(x_corr), columns = [x for x in range(1, x_corr.shape[1]+1)], index = [x for x in range(1, x_corr.shape[1]+1)])
    x_corr_df.to_csv(corr_csv_filename)

    fig, ax = plt.subplots(figsize = (12,10),dpi=600)
    #二维的数组的热力图，横轴和数轴的ticklabels要加上去的话，既可以通过将array转换成有column
    #和index的DataFrame直接绘图生成，也可以后续再加上去。后面加上去的话，更灵活，包括可设置labels大小方向等。
    cmap = sns.cubehelix_palette(start = 1.5, rot = 3, gamma=0.8, as_cmap = True)

    # sns.heatmap(df, linewidths = 0.05, vmax=1, vmin=0)
    sns.heatmap(x_corr_df, annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True, cmap="YlGnBu", fmt='.1f')

    #sns.heatmap(np.round(a,2), annot=True, vmax=1,vmin = 0, xticklabels= True, yticklabels= True,
    #            square=True, cmap="YlGnBu")
    # ax.set_title('二维数组热力图', fontsize = 18)

    plt.yticks(rotation=360)
    plt.tick_params(labelsize=15)
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]
    [label.set_fontsize(17) for label in labels]
    [label.set_fontweight('bold') for label in labels]
    [label.set_fontstyle('italic') for label in labels]
    # ax.set_ylabel('Models', fontdict=font0)
    # ax.set_xlabel('Data', fontdict=font0) #横变成y轴，跟矩阵原始的布局情况是一样的
    plt.savefig(corr_img_filename,bbox_inches='tight')
    # x_corr.to_csv('./x_corr_.csv')
    return x_corr_df


def get_jickknife(x_data, y_data, jk_csv_filename='./jk_value.csv', clf=xlf_dict['rf']):
    x_max_min = np.array( (x_data - x_data.min()) / (x_data.max() - x_data.min()))
    pred_dict = {}
    y_test_dict = {}
    auprc_dict = {}
    x_data = pd.DataFrame(x_data)
    header_range = x_data.keys().shape[0]
    loo_numbers = x_max_min.shape[0]
    for header_name in range(header_range):
        print(str(header_name) + '正在运行')
        pred_dict[header_name] = []
        y_test_dict[header_name] = []

        loo = LeaveOneOut()
        i = 0
        for train_index, test_index in loo.split(x_max_min, y_data):
            train_x, test_x = x_max_min[train_index, header_name:header_name+1], x_max_min[test_index,header_name:header_name+1]
            train_y, test_y = y_data[train_index], y_data[test_index]

            clf.fit(train_x, train_y)
            y_pred = (clf.predict_proba(test_x))[:,1]
            # y_score = clf.predict_proba(test_x)
            pred_dict[header_name].append(y_pred.item())
            y_test_dict[header_name].append(test_y.item())
            i +=1
            print('进行特征{}/{}, 此特征完成loo {}/{}'.format(header_name+1, header_range, i, loo_numbers))

        precision, recall, thresholds = precision_recall_curve( np.array(y_test_dict[header_name]),np.array(pred_dict[header_name]))

        auprc = auc(recall, precision)
        auprc_dict[header_name+1] = [auprc]

    print(auprc_dict)
    auprc_df = pd.DataFrame(auprc_dict)
    auprc_df.to_csv(jk_csv_filename, index=None)
    return auprc_df


def get_best_features(corr_data, jk_scores, best_features_filename='./best_features.csv'):  # use

    corr_data = pd.DataFrame(corr_data)
    idx_scores = np.array(jk_scores)
    idx_scores = idx_scores.reshape(-1)
    if len(idx_scores) != corr_data.keys().__len__():
        raise Exception('jk_scores 与 corr_data 的长度不匹配...')

    corr_data = corr_data > 0.8

    corr_list = {}
    arr_list = []
    for i in range(corr_data.shape[0]):
        if i not in arr_list:
            corr_list[i] = [i]
            for j in range(i+1, corr_data.shape[1]):
                if corr_data.iloc[i,j] == True:
                    corr_list[i].append(j)
                    arr_list.append(j)

    best_list = []
    for i in corr_list.keys():
            idx_arg = np.argsort(idx_scores[corr_list[i]])
            best_list.append(corr_list[i][idx_arg[-1]])

    best_list_features_numbers = np.sort(best_list)
    save_data = pd.DataFrame(best_list_features_numbers)
    save_data.to_csv(best_features_filename,index=None)
    return best_list_features_numbers


def get_best_model(x_data, y_data, jk_list):
    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    save_file = './get_best_model_auc/'
    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min()) / (x_data.max() - x_data.min())
    jk_list = list(jk_list)
    x_max_min = x_max_min[:,jk_list]

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    auc_dict = {}
    for clf_name in xlf_dict.keys():
        print(clf_name + '正在运行')
        clf = xlf_dict[clf_name]
        auc_dict[clf_name] = []
        for tr_idx, val_idx in kfold.split(x_max_min, y_data):
            train_x, train_y = x_max_min[tr_idx], y_data[tr_idx]
            test_x, test_y = x_max_min[val_idx], y_data[val_idx]
            clf.fit(train_x, train_y)
            y_pred = clf.predict(test_x)
            y_score = clf.predict_proba(test_x)
            idx_all = print_data(test_y, y_pred, y_score)
            auc_dict[clf_name].append(idx_all['AUROC'])
        print(auc_dict[clf_name])
    auc_data = pd.DataFrame(auc_dict)

    print(auc_data)
    print()
    print(auc_data.mean(axis=0))
    if not os.path.exists(save_file):
        os.mkdir(save_file)
    auc_data.to_csv(save_file + 'fold_pred_auc.csv')
    auc_mean = auc_data.mean(axis=0)
    auc_mean.to_csv(save_file + 'mean_pred_auc.csv')

    best_model_name = auc_mean[auc_mean == auc_mean.max()].keys().item()
    print('最优算法是：' + best_model_name)
    save_best_model(x_data, y_data, best_model_name, jk_list)
    return best_model_name


def save_best_model(x_data, y_data, best_model_name,jk_list, save_file_name='', best_param='' ):
    # data = pd.read_excel('./data/data_.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    y_data = np.array(y_data)
    x_data = np.array(x_data)
    x_max_min = (x_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    clf = xlf_dict[best_model_name]
    if best_param != '':
        for i in best_param.keys():
            if not (hasattr(clf, i)):
                raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, clf()))
            setattr(clf, i, best_param[i])
            ##最优参数示例： {'a':1,'b':2}

    jk_list = list(jk_list)
    x_max_min = x_max_min[:, jk_list]
    clf.fit(x_max_min, y_data)
    print(clf.predict(x_max_min))  # 自测准确度
    if not os.path.exists('./models/'):
        os.mkdir('./models/')

    joblib.dump(clf, './models/'+save_file_name +  best_model_name +'.pkl')
    print('保存完成，请在models文件夹里查看..')


def use_best_model(x_sits, x_pred_data, x_data, model_path, jk_list):
    data = pd.read_excel('./data/data_.xlsx')
    pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    # y_data = np.array(data['y'])
    # x_data = np.array(data.iloc[:, 4:])
    # x_pred_data = np.array(pred_data.iloc[:, 3:])
    x_max_min = (x_pred_data - x_data.min(axis=0)) / (x_data.max(axis=0) - x_data.min(axis=0))
    jk_list = list(jk_list)
    x_max_min = x_max_min[:, jk_list]

    clf = joblib.load(model_path)
    y_p = clf.predict(x_max_min)
    y_proba = clf.predict_proba(x_max_min)
    y_p = pd.DataFrame(y_p)
    y_proba = pd.DataFrame(y_proba)
    y_proba_cls = np.array(y_proba.iloc[:,1])

    y_proba_cls[y_proba_cls > 0.75] = 3
    y_proba_cls[(0.75>=y_proba_cls) & (y_proba_cls >= 0.5)] = 2
    y_proba_cls[(0.5>=y_proba_cls) & (y_proba_cls>= 0.25)] = 1
    y_proba_cls[0.25>=y_proba_cls] = 0

    y_proba_cls = pd.DataFrame(y_proba_cls, columns=['等级'])
    if not os.path.exists('./results'):
        os.mkdir('./results')
    y_p.to_csv('./results/y_pred.csv')
    y_proba.to_csv('./results/y_scores.csv')
    y_proba_cls.to_csv('./results/y_class.csv', index=None)
    print(y_p)
    print(y_proba)
    print('预测完成..')
    x_sits = pd.DataFrame(x_sits)
    y_proba_cls = pd.concat((x_sits, y_proba_cls), axis=1)
    return y_proba_cls


def grid_xlf(key, feature_i, xlf,param_dict,x_train, y_train,x_test, y_test):
    for i in param_dict.keys():
        if not (hasattr(xlf, i)):
            raise Exception('xlf_append: {} 属性在{}中不存在..'.format(i, xlf()))
    param_grid_dict = list(ParameterGrid(param_dict))

    param_EI_list = []
    for param_sigle in param_grid_dict:
        for keys in param_dict.keys():
            setattr(xlf, keys, param_sigle[keys])

        xlf.fit(x_train, y_train)
        yy_pred = xlf.predict_proba(x_test)

        y_pred = np.argmax(yy_pred, axis=1)
        true_values = np.array(y_test)
        y_scores = yy_pred[:, 1]
        TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
        Sensitivity = TP / (TP + FN)
        Specificity = TN / (TN + FP)
        ACC = (TP + TN) / (TP + FP + FN + TN)
        Precision = TP / (TP + FP)
        F1Score = 2 * TP / (2 * TP + FP + FN)
        MCC = ((TP * TN) - (FP * FN)) / ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) ** 0.5
        AUC = roc_auc_score(true_values, y_scores)
        pre, rec, thresholds = precision_recall_curve(y_test, y_scores)
        prc_area = auc(rec, pre)
        EI = [key,feature_i, param_sigle, TN, FP, FN, TP, Precision, Sensitivity, Specificity, F1Score, MCC, ACC, prc_area, AUC]
        param_EI_list.append(EI)
    return param_EI_list


def grid_search_best_model():
    pass


if __name__ == '__main__':
    data = pd.read_excel('./data/data_.xlsx')
    pred_data = pd.read_excel('./pred_data/2030l.xlsx')
    y_data = np.array(data['y'])
    x_data = np.array(data.iloc[:, 4:])
    x_pred_data = np.array(pred_data.iloc[:, 3:])
    sites = pred_data.iloc[:,1:3]

    x_corr = get_corr(x_data)
    jk_value = get_jickknife(x_data, y_data)
    best_feature = get_best_features(x_corr, jk_value)
    best_model = get_best_model(x_data, y_data, best_feature)
    best_model_path = './models/' + best_model + '.pkl'
    y_proba_cls = use_best_model(sites, x_pred_data, x_data,best_model_path , best_feature)
    print('结束...')





