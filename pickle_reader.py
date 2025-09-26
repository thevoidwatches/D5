import pickle
import pprint

INPUT_FILE = "output.pkl"

with open(INPUT_FILE, "rb") as infile:
    data = pickle.load(infile)

    output = ""

    for hyp in data:
#        pprint.pprint(data[hyp])

#        if not data[hyp]['active']: continue

        output += f"Hypothesis: {data[hyp]['hypothesis']}\n"
        output += f"  Active: {data[hyp]['active']}\n"
        output += f"  Mu: {data[hyp]['diff_w_significance']['mu']}\n"
        output += f"  P value: {data[hyp]['diff_w_significance']['p_value']}\n"

#        break

    print(output)
