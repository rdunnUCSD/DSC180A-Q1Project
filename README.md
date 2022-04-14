# Label Corruptions in Kernel Machines

Script for running different types of kernels with added label noise.

The applicable targets for running the script are:

`python run.py` : runs the script according to `config/script-params.json`

`python run.py test` : runs the script on testing data found in the `test` folder

`python run.py build` : builds distance matrices according to `config/script-params.json`

`python run.py clean` : removes any files produced by the script

The current output is sent to `out/results.json`. These can currently be viewed by running the cell in `notebooks/Visualization.ipynb`.
