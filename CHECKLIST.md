# Machine learning for fNIRS classification checklist

## Methodology
- [ ] pay attention to the study design when selecting classes
- [ ] use nested cross-validation, also called double cross-validation with the outer cross-validation (leaving out the test sets) for evaluation and the inner cross-validation (leaving out the validation sets) for the optimisation of models
- [ ] optimise the hyperparameters (with grid-search for instance) on validation sets
- [ ] use the test sets for evaluation and nothing else (no optimisation should be performed with the test set)
- [ ] create the train, validation and test sets in accordance with what the model is hypothesised to generalise (eg. unseen subject, unseen session, etc.), thanks to group k-fold cross-validation for example
- [ ] pay attention to not include test data when performing normalisation
- [ ] take extra care to not have any of the sets overlap (train, validation and test sets), the test set used to report results more than anything must consist of unseen data only
- [ ] pay attention to class imbalance (using metrics more appropriate than accuracy such as F1 score for example)
- [ ] perform a statistical analysis to find significance of the results when comparing results to chance level and classifiers to each other

## Reporting
- [ ] describe what data is used as input of the classifier and its shape
- [ ] describe the number of input examples in the dataset
- [ ] describe the details of the cross-validations implementations
- [ ] describe the details of each model used including the architecture of the model and every hyperparameter
- [ ] describe which hyperparameters have been optimised and how
- [ ] clearly state the number of classes and the chance level
- [ ] provide all necessary information related to the statistical analysis of the results, including the name of the tests, the verification of their assumptions and the p-values
