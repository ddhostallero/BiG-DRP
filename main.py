import argparse
from utils.utils import save_flags, mkdir
import os
os.environ['NUMEXPR_MAX_THREADS']='6'
import numexpr as ne

def main(FLAGS):

    if FLAGS.mode == 'train':
        import bigdrp.main_cv as main_fxn

    elif FLAGS.mode == 'extra':
        import bigdrp_plus.train_extra as main_fxn

    main_fxn.main(FLAGS)


if __name__ == '__main__':

    print("started main function")
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="lco", help="leave-cells-out [lco], leave-pairs-out (lpo)")
    parser.add_argument("--dataroot", default="../", help="root directory of the data")
    parser.add_argument("--folder", default="chk", help="directory of the output")
    parser.add_argument("--weight_folder", default="", help="directory of the weights")
    parser.add_argument("--outroot", default="./", help="root directory of the output")
    parser.add_argument("--mode", default="train", help="[train], extra")
    parser.add_argument("--seed", default=0, help="seed number for pseudo-random generation", type=int)
    parser.add_argument("--drug_feat", default="desc", help="type of drug feature ([morgan], desc, mixed)")
    parser.add_argument("--network_perc", default=1, help="percentile for network generation", type=float)

    norm_parser = parser.add_mutually_exclusive_group(required=False)
    norm_parser.add_argument('--normalize_response', dest='normalize_response', action='store_true')
    norm_parser.add_argument('--no-normalize_response', dest='normalize_response', action='store_false')
    parser.set_defaults(normalize_response=True)

    args = parser.parse_args() 
    mkdir(args.outroot + "/results/" + args.folder)
    save_flags(args)
    main(args)
