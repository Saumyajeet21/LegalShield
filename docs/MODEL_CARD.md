# LAVEN legal evidence model card

Design: a separate three-way verifier with Entailment, Contradiction and
NotMentioned. The neural candidates are a scratch-trained shared BiLSTM
and Microsoft DeBERTa-v3 with a supervised classification head. The general
comparison uses the trained MoritzLaurer MNLI/FEVER/ANLI model. Domain
fine-tuning weights, legal test scores and validated confidence limits
are now available for the imported Colab run below. Two CPU smoke models executed successfully on fictional
fixtures. Smoke accuracy is only an execution check.

Use the general model to measure transfer and the calibrated legal model
to score claims within the approved scope after validation. A verifier
score alone supplies neither legal authority nor exact evidence. Limitations
include weak support for rare class/exception patterns, source shift and
long evidence sets. The runtime can reject oversized contexts and cases
without a loaded verifier. No jurisdiction-general effectiveness is established.



## Actual ContractNLI comparison

The four-window retrieved track contained 2,091 eligible test pairs. Seed
13 TF-IDF achieved accuracy .6863 and macro F1 .6404. The RTX 4050-trained
BiLSTM achieved .6772 and .6319, respectively. Frozen general-domain DeBERTa
achieved .5189 and .3572. Its neutral class was mapped to NotMentioned for
this task. The models used the same test IDs and gold labels. The observed
ranking applies to this extraction, hypothesis, split and checkpoint
configuration. Class imbalance, lexical retrieval and single seed limit
conclusions. The legal DeBERTa benchmark is complete. Deployment validation is pending.


FAST_TRAIN selected epoch 1 at 86.33% validation accuracy and 80.87% macro
F1. On 2,091 official test pairs it achieved 87.18% accuracy and 82.98%
macro F1. Learned retrieval and pooling differ from the three baselines.
Test ECE rose from .0301 to .0546 after benchmark temperature scaling.
