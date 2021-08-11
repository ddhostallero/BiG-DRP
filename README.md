# BiG-DRP: Bipartite Graph-based Drug Response Predictor

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
- `--dataroot`: the root directory of your data (file names for input files can me modified in `constants.py`) (default: `./data/`)
- `--outroot`: the root directory of your outputs (default: `./results/`)
- `--folder`: subdirectory you want to save your outputs (optional)
- `--weight_folder`: subdirectory for the saved weights and encodings (for BiG-DRP+ only)
- `--mode`: `train` means BiG-DRP, `extra` means BiG-DRP+ (default: `train`)
- `--seed`: the seed number for 5-fold CV (default: 0)
- `--drug_feat`: type of drug feature (`desc`, `morgan`, or `mixed`, default: `desc`)
- `--network_perc`: percentile used for the bipartite graph threshold (default: 1)

## Data Availability

Preprocessed data can be accessed here: https://bit.ly/3yHTyCX