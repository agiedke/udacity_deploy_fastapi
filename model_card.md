# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
The model is a simple logistic regression model.

## Intended Use
The model can be used as a baseline to benchmark more complex models against.

## Training Data
The training data consists of 80% of the provided dataset. It contains categorical and numerical features. The target variable is a categorical variable, i.e. we are faced with a multi-class classification problem. 

## Evaluation Data

## Metrics
We are looking at the performance of the baseline model, using micro and macro f1 scores. We are first looking at the overall score and then at the score for slices of the dataset for each of the categorical variable. The model does not perform very well with macro scores below .5 for many slices.

## Ethical Considerations
The model may provide some insights for describing affects of sociodemografic features on salary but we need to be careful, not to use any model results for normative implications. 

## Caveats and Recommendations
We should test additional models and their performances. Also we may want to group some of the smaller category-subsets in order to obtain larger groups.
