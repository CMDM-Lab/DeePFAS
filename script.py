import argparse
import warnings

from models.inference import inference

warnings.filterwarnings('ignore', category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_path',
        type=str,
        required=True,
        help='dataset used to search compounds library',
    )
    parser.add_argument(
        '--dataset_id_path',
        type=str,
        required=True,
        help='dataset ids used to search compounds library',
    )
    parser.add_argument(
        '--topk',
        type=int,
        required=False,
        default=20,
        help='topk candidates',
    )
    parser.add_argument(
        '--data_file',
        type=str,
        required=True,
        help='input .mzxml or .mgf file'
    )
    parser.add_argument(
        '--out_dir',
        type=str,
        required=True,
        help='dictionary stored prediction'
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        help="if \'inference\' mode => unknown compound elif \'eval\' mode => known comound"
    )
    parser.add_argument(
        '--ignore_MzRange',
        type=str,
        required=False,
        default='not ignore',
        help='optional abandon limitation of peaks mz range (50 ~ 1000 Da)'
    )
    parser.add_argument(
        '--ignore_CE',
        type=str,
        required=False,
        default='not ignore',
        help='optional abandon limitation of collision energy range (20 ~ 46 eV)'
    )
    args = parser.parse_args()

    results = inference(dataset_path=args.dataset_path,
                        data_id_path=args.dataset_id_path,
                        topk=args.topk,
                        data_file=args.data_file,
                        out_dir=args.out_dir,
                        mode=args.mode,
                        ignore_MzRange=args.ignore_MzRange,
                        ignore_CE=args.ignore_CE)
    print(results)
