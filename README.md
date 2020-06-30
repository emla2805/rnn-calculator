# RNN Calculator

Adding two numbers in the coolest possible way, by training an RNN on generated data:

Input | Target
---|---
1+1 | 2
771+352	| 1123
196+547	| 743

## Requirements

Create a Python 3.7 virtual environment and activate it:

```bash
virtualenv -p python3.7 venv
source ./venv/bin/activate
```

Next, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Train model

To train a model using generated data simply run:

```bash
python train.py
```

To track metrics and predictions, start `Tensorboard`

```bash
tensorboard --logdir logs/
```

and navigate to [localhost:6006](localhost:6006).

### Use the model

To load the model, run an interactive Python REPL:

```bash
python -i eval.py
```
and then try some expressions, like:

```bash
>>> eval('123+321')
'444'
```
