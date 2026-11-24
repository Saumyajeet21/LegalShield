# Research methodology

Test whether governing amendments change answers correctly and unrelated or
unverified variations preserve correct answers under comparable coverage.
Use deterministic fictional ground truth and independent review. Preserve raw
source spans and original digests. Keep model precision/recall, retrieval
coverage, authorization and symbolic checks as separate outcomes.

The enhanced answer path checks typed Qwen claims against exact retained
sources, calibrated independent evidence inference and bounded Z3 rules.
Z3 checks premise consistency before testing negated and affirmative
conclusions. A supported solver result holds under the encoded premises.
Compare the complete version-aware path to simpler paths, and report
qualifying abstention, uncertainty failures and negative results.
