import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from utils import load_predictions, get_per_drug_metric, get_per_drug_fold_metric
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--folder', default='results/big_plus/', help='folder containing the predictions')
parser.add_argument('-s', '--split', default='lco', help='[lco], lpo')
parser.add_argument('-o', '--outfolder', default='results/', help='output folder')
parser.add_argument('-d', '--drugset', default='metrics/lists/drug_list.txt', help='file containing the list of drugs')
parser.add_argument('-r', '--response', default='ln_ic50', help='[ln_ic50], auc')
parser.add_argument('-m', '--mode', default='collate', help='[collate], per_fold')
parser.add_argument('-l', '--labels', default='../../drp-data/grl-preprocessed/drug_response/gdsc_tuple_labels_folds.csv', help='labels and folds file')
args = parser.parse_args() 

folder = args.folder

# load labels
y_tup = pd.read_csv(args.labels, index_col=0)
if args.split == 'lpo':
	y_tup['fold'] = y_tup['pair_fold']
else:
	y_tup['fold'] = y_tup['cl_fold']

y_tup = y_tup.loc[y_tup['fold']>=0]
y = y_tup.pivot(index='cell_line', columns='drug', values='ln_ic50')
y_bin = y_tup.pivot(index='cell_line', columns='drug', values='resistant')

samples = list(y_tup['cell_line'].unique())

# load drugs
drugs = open(args.drugset).read().split('\n')
if drugs[-1] == '': drugs=drugs[:-1]

# filter out unnecessary samples/drugs
y_tup = y_tup.loc[y_tup['drug'].isin(drugs)]
y = y.loc[samples, drugs]
y_bin = y_bin.loc[samples, drugs] # binary response

y0 = y.replace(np.nan, 0)
null_mask = y0.values.nonzero()
y_norm = (y - y.mean())/y.std()   # normalized response

print("calculating for %d drugs and %d cell lines..."%(len(drugs), len(samples)))

# create mask for folds
# NOTE: This code assumes that all (drug, CCL) pairs can only exist in 1 fold
fold_mask = y_tup.pivot(index='cell_line', columns='drug', values='fold')
fold_mask = fold_mask.loc[samples, drugs]

# load predictions
_, df = load_predictions(folder, split=args.split, fold_mask=fold_mask)
df = df.loc[samples, drugs]
preds_norm = df                      # actual prediction for normalized response
preds_unnorm = df*y.std() + y.mean() # if we revert back to unnormalized response

# Calculate overall metrics
print('calculating overall metrics...')

mets = ["spearman (fold.%d)"%i for i in range(5)]
overall = pd.DataFrame(index=mets, columns=['normalized %s'%args.response, 'raw %s'%args.response])

s = np.zeros((5, 2))
for i in range(5):
	m  = ((fold_mask == i)*1).values.nonzero()
	s[i, 0] = spearmanr(y_norm.values[m], preds_norm.values[m])[0]
	s[i, 1] = spearmanr(y.values[m], preds_unnorm.values[m])[0]
overall.loc[mets] = s
overall.loc['spearman (fold.mean)'] = s.mean(axis=0)
overall.loc['spearman (fold.stdev)'] = s.std(axis=0)
print(overall)

outfile = '%s/%s_performance_%d_drugs.xlsx'%(args.outfolder, args.split, len(drugs))
exwrite = pd.ExcelWriter(outfile)#, engine='xlsxwriter')
overall.to_excel(exwrite, sheet_name='Overall')

if args.mode == 'collate':
	per_drug_metric = get_per_drug_metric(preds_norm, y_norm, y_bin)
elif args.mode == 'per_fold':
	per_drug_metric = get_per_drug_fold_metric(preds_norm, y_norm, fold_mask, y_bin)
per_drug_metric = per_drug_metric.sort_values('SCC', ascending=False)


drug_summary = pd.DataFrame(index=per_drug_metric.columns, columns=['mean', 'stdev'])
drug_summary['mean'] = per_drug_metric.mean()
drug_summary['stdev'] = per_drug_metric.std()
print(drug_summary)

per_drug_metric.to_excel(exwrite, sheet_name='Drug')
drug_summary.to_excel(exwrite, sheet_name='Summary Drug')
exwrite.save()

print("Results written to: %s"%outfile)