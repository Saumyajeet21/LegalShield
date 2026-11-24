# Metrics

NLI uses accuracy, per-label and averaged precision, recall and F1, specificity,
balanced accuracy, MCC, confusion, ROC AUC and average precision. ROC and PR
curves consume probabilities. A class missing either outcome has null ROC AUC
and average precision. Undefined P/R/F1 follows scikit-learn zero division.
Report support alongside these values.

For fixed eligible pairs, CS is the fraction of correct originals with
correct authorized variants, divided by all eligible change pairs. IS uses
correct originals and correctly preserved answers across invariance pairs.
VRS is 2 × CS × IS / (CS + IS), zero for two defined zeros, null for an empty
pair group. Report abstention and answerable-case coverage separately.
Group paired bootstrap samples by family.

For verifier temperature T, probabilities are softmax(logits/T).
Calibration reports multivariate squared probability error (Brier),
negative log likelihood, bin-count weighted calibration error, and
risk-coverage. Incorrect-answer AUROC is null with only one error class.
Empirical answer entropy is minus the sum of p log2 p over sampled answer
clusters. The implementation uses structured semantic keys and material-token
checks. It requires empirical validation of semantic cluster reliability.
