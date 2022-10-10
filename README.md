# BiG-DRP: Bipartite Graph-based Drug Response Predictor

Implementation of Bipartite Graph-represented Drug Response Predictor (BiG-DRP and BiG-DRP+) as described in:

>David Earl Hostallero, Yihui Li, Amin Emad, Looking at the BiG picture: incorporating bipartite graphs in drug response prediction, Bioinformatics, Volume 38, Issue 14, 15 July 2022, Pages 3609–3620, https://doi.org/10.1093/bioinformatics/btac383

## Dependencies
This repository has been tested on python 3.7. To install the dependencies run the following on the terminal
```
pip install -r requirements.txt
```

## Running BiG-DRP
```
python main.py
```

## Running BiG-DRP+
To run BiG-DRP+, you must first run BiG-DRP while specifying the results subfolder (`--folder=<folder_name>`). Then run BiG-DRP with the `--weight_folder` specified as the results subfolder in the previous run.
```
python main.py --mode=train --folder=big
python main.py --mode=extra --weight_folder=big --folder=big_plus
```

## Additional Parameters

- `--split`: the type of data-splitting to use (`lco` or `lpo`, default: `lco`)
- `--dataroot`: the root directory of your data (file names for input files can me modified in `utils/constants.py`) (default: `../`)
- `--outroot`: the root directory of your outputs (default: `./`)
- `--folder`: subdirectory you want to save your outputs (optional)
- `--weight_folder`: subdirectory for the saved weights and encodings (for BiG-DRP+ only)
- `--mode`: `train` means BiG-DRP, `extra` means BiG-DRP+ (default: `train`)
- `--seed`: the seed number for 5-fold CV (default: 0)
- `--drug_feat`: type of drug feature (`desc`, `morgan`, or `mixed`, default: `desc`)
- `--network_perc`: percentile used for the bipartite graph threshold (default: 1)

## Data Availability
Preprocessed data can be accessed here: https://dx.doi.org/10.6084/m9.figshare.20022947

## Performance Metrics
Note that when you run ``main.py``, the output performance metrics do not correspond to the ones we presented in the paper because ``main.py`` only shows the overall performance (i.e. performance for all (drug, CCL) pairs in the test set is calculated as a whole). In the paper, we calculated the performance **per drug then averaged the per-drug performances**. To run the per-drug calculation, use ``metrics/calculate_metrics.py``. Example:

```
python metrics/calculate_metrics.py --folder=results/big_plus/ --outfolder=results/big_plus
```

## BibTex Citation
```
@article{hostallero2022looking,
  title={Looking at the BiG picture: incorporating bipartite graphs in drug response prediction},
  author={Hostallero, David Earl and Li, Yihui and Emad, Amin},
  journal={Bioinformatics},
  volume={38},
  number={14},
  pages={3609--3620},
  year={2022},
  publisher={Oxford University Press}
}
```