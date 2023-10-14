import warnings
warnings.filterwarnings("ignore")

import scenario
import utils
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def main():
    args = utils.get_args()
    method = getattr(scenario, args.scenario)
    method(args)

if __name__ == '__main__':
    main()