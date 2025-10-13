# basic imports
import pickle
import argparse
import sys
import os

# imports from other modules
from D5 import D5
from validator import Validator
from lm_proposer import GPT3_Proposer
from get_representative import return_extreme_values
from run_example_problem import subsample

def problem_parser():
    parser = argparse.ArgumentParser(prog=sys.argv[0].replace(".py", ""))

    parser.add_argument(
        '--verifier',
        default='ruiqi-zhong/d5_t5_validator',
        choices=['ruiqi-zhong/d5_t5_validator', 'ruiqi-zhong/d5_t5_validator_700M', 'ruiqi-zhong/d5_t5_validator_3B'],
        help='The trained verifier to use. The default is ruiqi-zhong/d5_t5_validator - there are also smaller alternatives called ruiqi-zhong/d5_t5_validator_700M and ruiqi-zhong/d5_t5_validator_3B.'
    )
    parser.add_argument(
        '--problem_folder',
        default='./problem_set',
        help='The location to search for problem files.'
    )
    parser.add_argument(
        '--output_folder',
        default='./problem_output',
        help='The location to save output.pkl files.'
    )
    parser.add_argument(
        '--lo',
        type=int,
        default=0,
        help='The number of the problem set to begin at (inclusive). The first problem is numbered 0.'
    )
    parser.add_argument(
        '--hi',
        type=int,
        default=622,
        help='The number of the problem set to end at (exclusive). The last problem is numbered 621.'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=-1,
        help='The number of randomly selected samples to verify the hypotheses on. Defaults to testing on all representative samples (as -1).'
    )

    return parser.parse_args()

if __name__ == '__main__':
    args = problem_parser()
    #Makes sure that the lo-hi range is within the correct range
    args.lo = max(args.lo, 0)
    args.hi = min(args.hi, 622) 
    print(args)

    verifier = Validator(args.verifier, batch_size=32)

    for i in range(args.lo, args.hi):
        problem_path = f'{args.problem_folder}/task_{i}.pkl'
        problem = pickle.load(open(problem_path,'rb'))

        #Finding the extreme values
        extreme_vals = return_extreme_values(problem['split']['research']['A_samples'], problem['split']['research']['B_samples'])
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = extreme_vals['sorted_A_samples'], extreme_vals['sorted_B_samples']

        #Building the representative subsample
        if args.subsample > 0:
            problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = subsample(problem['split']['research']['A_samples'], args.subsample), subsample(problem['split']['research']['B_samples'], args.subsample)
        
        proposer = GPT3_Proposer(problem)

        d5 = D5(
            problem['split']['research']['A_samples'], 
            problem['split']['research']['B_samples'], 
            verifier,
            proposer,
            total_hypotheses_count=60,
            early_stop=True
        )
        h2h_dicts = d5.run()

        h_sorted = sorted(h2h_dicts, key=lambda h: h2h_dicts[h]['diff_w_significance']['mu'], reverse=True)
        for h in h_sorted:
            h_dict = h2h_dicts[h]
            # print out the example hypothesis along with their V' score
            print(h_dict['hypothesis'], 'V\'', h_dict['diff_w_significance']['mu'])
        output_path = f'{args.output_folder}/output_{i}.pkl'
        # creates the folder if it doesn't already exist
        try:
            os.mkdir(args.output_folder)
        except FileExistsError:
            pass
        pickle.dump(h2h_dicts, open(output_path, 'wb'))
