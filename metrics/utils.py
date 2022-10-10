import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm


def load_predictions(folder, split, fold_mask, file_prefix='val_prediction_fold'):
    preds = []
    for i in range(5):
        x = pd.read_csv(folder+'/%s_%d.csv'%(file_prefix, i), index_col=0)
        x.index = x.index.astype(str)
        preds.append(x)
    
    if split=='lco': # leave cell out
        preds_df = pd.DataFrame()
        for i in range(5):
            preds_df = pd.concat([preds_df, preds[i]])
        preds_df = preds_df.sort_index()

    else:
        if fold_mask is None:
            print("fold mask should not be None when loading for leave-pairs-out")

        drugs = preds[0].columns

        if len(drugs) > len(fold_mask.columns):
            drugs = list(fold_mask.columns)

        samples = set()
        for i in range(5):
            samples = samples.union(set(preds[i].index))
        samples = sorted(list(samples)) # fix the order

        preds_df = pd.DataFrame(np.zeros((len(samples), len(drugs))), index=samples, columns=drugs)
        for i in range(5):
            temp = preds[i][drugs].replace(np.nan, 0)
            missing = set(samples) - set(temp.index) # the fold doesn't have these samples
            if len(missing) > 0:
                # print('fold %d does not have samples: '%i, missing)
                for m in missing:
                    temp.loc[m] = np.zeros(len(drugs))

            fm = ((fold_mask == i)*1).loc[samples, drugs]
            preds_df += temp.loc[samples, drugs]*fm # make sure that only those in the fold are added

    return preds, preds_df


def get_per_drug_metric(df, y, y_bin=None):
    """
        df: DataFrame containing the predictions with drug as columns and CCLs as rows
        y: DataFrame containing the true responses
        y_bin: DataFrame containing the true responses in binary
    """

    y0 = y.replace(np.nan, 0)
    drugs = df.columns
    if y_bin is not None:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE', 'AUROC'])
        calc_auroc = True
    else:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE'])

    for drug in drugs:
        mask = y0[drug].values.nonzero()
        prediction = df[drug].values[mask]
        true_label = y[drug].values[mask]
        
        rmse = np.sqrt(((prediction-true_label)**2).mean())
        scc = spearmanr(true_label, prediction)[0]
        pcc = pearsonr(true_label, prediction)[0]

        if calc_auroc:
            true_bin = y_bin[drug].values[mask]
            true_bin = true_bin.astype(int)
            if true_bin.mean() != 1:
                auroc = roc_auc_score(true_bin, prediction)
            else:
                auroc = np.nan
            metrics.loc[drug] = [scc,pcc,rmse,auroc]
        else:
            metrics.loc[drug] = [scc,pcc,rmse]

    return metrics

def get_per_drug_fold_metric(df, y, fold_mask, y_bin=None):
    """
        df: DataFrame containing the predictions with drug as columns and CCLs as rows
        y: DataFrame containing the true responses
        fold_mask: DataFrame containing the designated folds
        y_bin: DataFrame containing the true responses in binary
    """

    drugs = df.columns

    if y_bin is not None:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE', 'AUROC'])
        calc_auroc = True
    else:
        metrics = pd.DataFrame(columns=['SCC', 'PCC', 'RMSE'])

    for drug in tqdm(drugs):

        temp = np.zeros((5, len(metrics.columns)))
        for i in range(5):
            mask = ((fold_mask[drug] == i)*1).values.nonzero()
            prediction = df[drug].values[mask]
            true_label = y[drug].values[mask]

            rmse = np.sqrt(((prediction-true_label)**2).mean())
            scc = spearmanr(true_label, prediction)[0]
            pcc = pearsonr(true_label, prediction)[0]

            if calc_auroc:
                true_bin = y_bin[drug].values[mask]
                true_bin = true_bin.astype(int)
                if true_bin.mean() != 1:
                    auroc = roc_auc_score(true_bin, prediction)
                else:
                    auroc = np.nan
                temp[i] = [scc,pcc,rmse,auroc]
            else:
                temp[i] = [scc,pcc,rmse]

        metrics.loc[drug] = temp.mean(axis=0)
    return metrics