# imports
import pickle
from sklearn.semi_supervised import LabelPropagation
from sklearn.metrics import precision_recall_curve, average_precision_score
import argparse
import sys
import os
import random
import matplotlib.pyplot as plt

# sklearn needs a label array with all samples and a label vector with labels for some of them and not others
# each entry in the sample array X should be the embedding for the RHx of that that sample (the difference between the hypothesis and the sample's embeddings)
#	rebuild the pickle file I made so that each hypothesis also includes the embeddings for that problem set
# each entry in the label array y should be either a label for that sample or -1 as a placeholder value
#	have a preparation function that builds these arrays, rounding the samplescore to 0 or 1 as the label value
#   it should end up running 3 times, with 10%, 5%, and 1% of the entries labeled.
# then make the model:		model = LabelPropagation(kernel=KERNEL, gamma=GAMMA, max_iter=1000)
#	three possible kernels
#		rbf (radial basis function)
#		knn (k-nearest neighbors)
#		poly (polynomial)
#	gamma - kernal coefficient, will have notable effect. Test some values
# run the model:		model.fit(X, y) followed by model.transduction_
# plot things: plot the results. Talk to dad about how to represent?
#	use a precision recall curve - sklearn should have examples of how to do that

def label_propagation_parser():
# argutil function to let it use different inputs
    parser = argparse.ArgumentParser(prog=sys.argv[0].replace(".py", ""))
#	one for sample input file
    parser.add_argument(
        '--infile',
        default='test_subset.pkl',
        help='location of the file to look for hypothesis data'
    )
#	one for embedding input file
#		looks for embeddings in sample input if not given
#     parser.add_argument(
#        '--embeddings',
#        default='',
#        help='location of the file to look for embedding data. If no file is specified, embedding data will be searched for in hypothesis data'
#    )
#	one for output location
    parser.add_argument(
        '--outfolder',
        default='./lprop_outputs',
        help='the folder to save outputs to.'
    )
#	one for kernel
    parser.add_argument(
        '--kernel',
        default='rbf',
        choices=['rbf','knn','poly'],
        help='the kernel function to use, from rbf (radial basis function), knn (k-nearest neighbors), and poly (polynomial)'
    )
#	one for gamma
    parser.add_argument(
        '--gamma',
        default=1,
        type=float,
        help='the kernel coefficient for an rbf kernel'
    )

    return parser

def load_data(file: str):
# function to load the data files
    with open(file, 'rb') as infile:
        data = pickle.load(infile)
    # convert to an array to make things a little easier later
    ret = []
    for entry in data:
        ret.append(data[entry])
    return ret

def prepare_data(hypothesis: dict):
# function to prepare the data
    X = [] # all samples
    Y0 = [] # all labels

    embeds = hypothesis['embeddings']
    for key, value in hypothesis['sample2score'].items():
        # this builds the X array and an array of all matching labels.
        X.append(embeds[key])
        Y0.append(round(value))

    fifty_per_len = round(len(X) * 0.5)
    twentyfive_per_len = round(len(X) * 0.25)
    ten_per_len = round(len(X) * 0.1)
    five_per_len = round(len(X) * 0.05)
    one_per_len = round(len(X) * 0.01)

    fifty_per_indices = set()
    twentyfive_per_indices = set()
    ten_per_indices = set()
    five_per_indices = set()
    one_per_indices = set()
    # sets that will be used to track what indexes are used
    while len(fifty_per_indices) <= fifty_per_len:
        new_index = random.randint(0, len(Y0) // 2)
        fifty_per_indices.add(new_index)
        if len(twentyfive_per_indices) <= twentyfive_per_len:
            twentyfive_per_indices.add(new_index)
            if len(ten_per_indices) <= ten_per_len:
                ten_per_indices.add(new_index)
                if len(five_per_indices) <= five_per_len:
                    five_per_indices.add(new_index)
                    if len(one_per_indices) <= one_per_len:
                        one_per_indices.add(new_index)

    Y50 = []
    Y25 = []
    Y10 = []
    Y5 = []
    Y1 = []
    for i in range(len(Y0)):
        val = Y0[i]
        if i in one_per_indices:
            Y1.append(val)
            Y5.append(val)
            Y10.append(val)
            Y25.append(val)
            Y50.append(val)
        else:
            Y1.append(-1)
            if i in five_per_indices:
                Y5.append(val)
                Y10.append(val)
                Y25.append(val)
                Y50.append(val)
            else:
                Y5.append(-1)
                if i in ten_per_indices:
                    Y10.append(val)
                    Y25.append(val)
                    Y50.append(val)
                else:
                    Y10.append(-1)
                    if i in twentyfive_per_indices:
                        Y25.append(val)
                        Y50.append(val)
                    else:
                        Y25.append(-1)
                        if i in fifty_per_indices:
                            Y50.append(val)
                        else:
                            Y50.append(-1)

    return (X, Y0, Y1, Y5, Y10, Y25, Y50)

if __name__ == '__main__':
# get args
    args = label_propagation_parser().parse_args()
    print(args)
# load data files (func)
    data = load_data(args.infile)
# create model
    model = LabelPropagation(kernel=args.kernel, gamma=args.gamma, max_iter=1000)
    for hyp in data:
        print(f"Beginning on hypothesis: {hyp['hypothesis']}")
# for each hypothesis in the data file
#   prepare the data (func)
        X, Y0, Y1, Y5, Y10, Y25, Y50 = prepare_data(hyp)
        #print(f"Data prepared for {len(Y0)} embeddings, from 1% ({len(Y0) - Y1.count(-1)}) to 50% ({len(Y0) - Y50.count(-1)})")
#	fit the model
#	run it
#	plot the results
        plt.figure(figsize=(8, 6))

        half_start = len(Y0) // 2
        Y0_back = Y0[half_start:]

        try:
            print("Running model for 1%")
            model.fit(X, Y1)
            Y1_score = model.label_distributions_[:, 1]
            Y1_back = Y1_score[half_start:]
            p1, r1, _ = precision_recall_curve(Y0_back, Y1_back)
            avg1 = average_precision_score(Y0_back, Y1_back)
            plt.plot(r1, p1, label=f"1% Labeled (AP={avg1:.3f})")
        except Exception as e:
            print(f"Failed to render 1% model for hyp {hyp['hypothesis']}")
            print(f"Error: {e}")

        try:
            print("Running model for 5%")
            model.fit(X, Y5)
            Y5_score = model.label_distributions_[:, 1]
            Y5_back = Y5_score[half_start:]
            p5, r5, _ = precision_recall_curve(Y0_back, Y5_back)
            avg5 = average_precision_score(Y0_back, Y5_back)
            plt.plot(r5, p5, label=f"5% Labeled (AP={avg5:.3f})")
        except Exception as e:
            print(f"Failed to render 1% model for hyp {hyp['hypothesis']}")
            print(f"Error: {e}")

        try:
            print("Running model for 10%")
            model.fit(X, Y10)
            Y10_score = model.label_distributions_[:, 1]
            Y10_back = Y10_score[half_start:]
            p10, r10, _ = precision_recall_curve(Y0_back, Y10_back)
            avg10 = average_precision_score(Y0_back, Y10_back)
            plt.plot(r10, p10, label=f"10% Labeled (AP={avg10:.3f})")
        except Exception as e:
            print(f"Failed to render 1% model for hyp {hyp['hypothesis']}")
            print(f"Error: {e}")

        try:
            print("Running model for 25%")
            model.fit(X, Y25)
            Y25_score = model.label_distributions_[:, 1]
            Y25_back = Y25_score[half_start:]
            p25, r25, _ = precision_recall_curve(Y0_back, Y25_back)
            avg25 = average_precision_score(Y0_back, Y25_back)
            plt.plot(r25, p25, label=f"25% Labeled (AP={avg25:.3f})")
        except Exception as e:
            print(f"Failed to render 1% model for hyp {hyp['hypothesis']}")
            print(f"Error: {e}")

        try:
            print("Running model for 50%")
            model.fit(X, Y50)
            Y50_score = model.label_distributions_[:, 1]
            Y50_back = Y50_score[half_start:]
            p50, r50, _ = precision_recall_curve(Y0_back, Y50_back)
            avg50 = average_precision_score(Y0_back, Y50_back)
            plt.plot(r50, p50, label=f"50% Labeled (AP={avg50:.3f})")
        except Exception as e:
            print(f"Failed to render 1% model for hyp {hyp['hypothesis']}")
            print(f"Error: {e}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curves\n{hyp['hypothesis']}")
        plt.legend()
        plt.grid(True)

        if not os.path.exists(args.outfolder):
            os.makedirs(args.outfolder)
        plt.savefig(f"{args.outfolder}/precision_recall_curves{hyp['hypothesis'][:20]}.png", dpi=300, bbox_inches="tight")

        #print("Plot figures saved.")
