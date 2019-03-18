# Compositional pre-training for neural semantic parsing
Stanford CS224N project

# Running the code
Use Python 3.6+.

1- Get the following two directories from [CodaLab](https://worksheets.codalab.org/bundles/0xfe58a4a712054c8ab5a585596647d1ed) (look under dependencies) and place them in the root of this directory:
```
evaluator/
lib/
```
These are required for evaluation.   

2- Get the data
Get the data from the same CodaLab link. We need these files:
```
# files for the geoquery domain
data/geo880_train600.tsv
data/geo880_test260.tsv
```

3- Install dependencies:
```bash
pip3 install -r requirements.txt
```

4- Run the experiments
```bash
python3 experiments.py
```

The experiment parameters are defined in `params.py`. The cross-product of all the parameters is run and the results are collected in a csv file under `results\`. A lot of intermediate artifacts are generated for analysis in the same folder.

### References
- [`OpenNMT-py`](http://opennmt.net/OpenNMT-py/) is heavily used in this project.
- [Jia et. al's code for Data Recombination for Neural Semantic Parsing (ACL 2016)](https://worksheets.codalab.org/worksheets/0x50757a37779b485f89012e4ba03b6f4f/)
 