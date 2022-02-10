# Training Data Debugging for the Fairness of Machine Learning Software


This repository stores our experimental codes for the paper “Training Data Debugging for the Fairness of Machine Learning Software”，LTDD is short for the method we propose in this paper: *Linear-regression
based Training Data Debugging*.

<br/>

## Datasets

We use 9 datasets, all of which are widely used in fairness research: **Adult, COMPAS, German Credit, Default, Heart Disease, Bank Marketing, Student Performance, MEPS15 and MEPS16**. We provide these data sets in the "datasets" folder.
For MEPS15 and MEPS16, they can be loaded through python's aif360 package:

```python
from aif360.datasets import MEPSDataset19,MEPSDataset21
```

<br/>

## Codes for LTDD

You can easily reproduce our method, we provide it in the LTDD folder. 

The codes in the folder are named for the applicable scenarios. The Adult and COMPAS data sets include two protected attributes, so we divide them into two scenarios: Adult_sex (COMPAS_sex) and Adult_race (COMPAS_race). 

The code contains data preprocessing, our method and the calculation of indicators. You can run these codes directly to get the experimental results.

<br/>

## Baseline methods

We compare our method with four baseline methods:

**Fair-Smote:** Proposed in the paper: *Bias in Machine Learning Software: Why? How? What to Do?* Fair-Smote is a pre-processing method that uses the modified SMOTE method to make the distribution of sensitive features  in the data set consistent, and then deletes biased data through situation testing.

We use the code they provided in the code repository: <https://github.com/joymallyac/Fair-SMOTE>

**Fairway:** Proposed in the paper: *Fairway: A Way to Build Fair ML Software*. Fairway is a hybrid algorithm that combines pre-processing and in-processing methods. It trains the models separately according to the protected attributes to remove biased data points. Then it uses Flash technique for multi-objective optimization, including model performance indicators and fairness indicators.

We use the code they provided in the code repository: <https://github.com/joymallyac/Fairway>

**Reweighing:** Reweighing is a pre-processing method that calculates a weight value for each data point based on the expected probability and the observed probability, to help the unprivileged class have a greater chance of obtaining favorable prediction results.

We use python's AIF360 module to achieve it: 

```python
from aif360.algorithms.preprocessing import Reweighing
```

**Disparate Impact Remover (DIR):** This method is a pre-processing technology. Its main goal is to eliminate Disparate Impact and increase fairness between groups by modifying attribute values except the protected attributes.

We also use python's AIF360 module to achieve it: 

```python
from aif360.algorithms.preprocessing import DisparateImpactRemover
```

## Experimental settings
* When training the model, we refer to Fairway's attribute selection before constructing ML software, which removes some inappropriate attributes, e.g., 'native-country' on Adult dataset.
* In Discussion 3, when we use AOD and EOD, we refer to Fairway's usage and report after **taking the absolute value**, less absolute values |AOD| and |EOD| mean more fairness.
