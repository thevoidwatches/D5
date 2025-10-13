# How to use run_problems.py

First, you need to acquire the dataset. OpenD5.pkl can be acquired [from this site](https://zenodo.org/records/7683302). Then run `python3 problem_splitter.py`, which will break it apart into individual files found in the ./problem_set folder.

Once that's done `run_problems.py` will be able to load task_[NUMBER].pkl files from the ./problem_set folder and run them in order. For each file loaded, it will find the representative values from both the A and B corpuses, generate hypothesis based on the samples, and then test the hypotheses using a verifier.

Arguments can be seen by running `python3 run_problems.py --help`, but the most important ones to know are --lo, --hi, and --subsample.

The first two define the range of problems to test. --lo defines the problem number to start with (inclusive) and defaults to the first problem, 0. --hi defines the problem to end with (exclusive) and defaults to ending after the last problem, 621. It is important to define the intended range correctly.

The last reduces how much of the representative corpus to test the hypotheses against. Using it can cut down runtime significantly, but will also reduce accuracy.
