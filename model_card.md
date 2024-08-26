# Model Card


## Model Details
The model is a simple **logistic regression** model. 

## Intended Use
The model can be used as a baseline to benchmark more complex models against.

## Training Data
The training data consists of 80% of the provided dataset. It contains categorical and numerical features. The target variable is a categorical variable, i.e. we are faced with a **multi-class classification** problem. 

## Evaluation Data
The model performance has been calculated using 20% of the labeled data.

## Metrics
We are looking at the performance of the baseline model, using **micro** and **macro f1 scores** overall as well as on slices of the categorical features. The model does not perform very well with macro scores below .5 for many slices.
### F1 Score Overall
Overall score f1 macro: **0.63**\
Overall score f1 micro: **0.80**
### F1 Score on Slices of Features
#### Workclass
Min f1 macro: 0.46, label: *"Unknown"*\
Max f1 macro: 0.50, label: *"Self-emp-inc"*
#### Education
Min f1 macro: 0.42, label: *"5th-6th"*\
Max f1 macro: 0.78, label: *"7th-8th"*
#### Marital Status
Min f1 macro: 0.45, label: *"Divorced"*\
Max f1 macro: 0.78, label: *"Widowed"*
#### Occupation
Min f1 macro: 0.44, label: *"Protective-serv"*\
Max f1 macro: 0.49, label: *"Craft-repair"*
#### Relationship
Min f1 macro: 0.45, label: *"Other-relative"*\
Max f1 macro: 0.49, label: *"Husband"*
#### Race
Min f1 macro: 0.43, label: *"Amer-Indian-Eskimo"*\
Max f1 macro: 0.55, label: *"Asian-Pac-Islander"*
#### Sex
Min f1 macro: 0.47, label: *"Female"*\
Max f1 macro: 0.47, label: *"Male"*
#### Native Country
Min f1 macro: 0.43, label: *"Mexico"*\
Max f1 macro: 0.52, label: *"Unknown"*

## Ethical Considerations
The model may provide some insights for describing affects of sociodemografic features on salary but we need to be careful, not to use any model results for normative implications. 

## Caveats and Recommendations
We should test additional models and their performances. Also we may want to group some of the smaller category-subsets in order to obtain larger groups.
