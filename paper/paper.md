# Accuracy, Fairness, and Carbon Cost: Auditing Machine Learning Models on Loan and Income Data

## Abstract

Machine-learning studies typically optimize and report one axis at a time: predictive accuracy, a fairness gap, or energy use. This audit reports all three for the same models on the same data splits. Four classifiers — logistic regression, a random forest, XGBoost and a multilayer perceptron — are trained on two OpenML datasets, Adult and German Credit, with the protected attribute removed from the features and retained only as an audit label. Accuracy, F1 and ROC-AUC are compared against majority-class baselines; demographic-parity and equalized-odds gaps are measured per seed; and codecarbon records carbon-dioxide-equivalent emissions for training and inference separately. Three mitigation methods — threshold optimization, exponentiated gradient and reweighing — are evaluated before and after, and SHAP explanations probe whether the removed attribute re-enters the decision through correlated features. A single-axis audit would, by itself, have selected a different model depending on which axis it reported. The accuracy leader changes between datasets, all Adult models fall short of the four-fifths rule while most German Credit models meet it, mitigation shrinks parity gaps at a small accuracy cost while sometimes widening equalized-odds gaps, training dominates inference in measured emissions, and the strongest correlated feature of the removed attribute is marital status.

## 1 Introduction

Machine-learning systems carry two externalities that are rarely audited together. The first is social: models trained on historical records reproduce the biases embedded in those records and can turn them into systematic differences in decisions across protected groups (Mehrabi et al., 2021). Fairness research has responded with formal definitions and mitigation algorithms (Hardt et al., 2016) and with practical toolkits that make the measurements routine (Bird et al., 2020). The second externality is environmental: training a model consumes energy and emits carbon dioxide equivalent, and this cost grows with model size and with the number of experiments run (Strubell et al., 2019). Proposals such as reporting efficiency alongside accuracy aim to make that cost visible (Schwartz et al., 2020).

Most studies still audit a single axis. An accuracy table does not say whether the winner is also the fairest, and a fairness table does not say what the mitigation costs in emissions. When all three are measured on the same models and the same splits, the trade-off becomes an empirical question rather than an assumption.

This paper audits accuracy, fairness and carbon together on loan- and income-prediction data. We train four standard classifiers on two OpenML datasets, Adult (Kohavi, 1996) and German Credit [CITE: NEEDED], remove the protected attribute from the features, measure each model with accuracy, F1, ROC-AUC, demographic-parity and equalized-odds metrics, record training and inference emissions with codecarbon [CITE: NEEDED], apply three mitigation methods, and probe with SHAP (Lundberg & Lee, 2017) whether the removed attribute leaks back in through correlated features.

Three research questions structure the study:

- **RQ1 (accuracy and fairness).** How do standard classifiers compare in accuracy and fairness on income and credit prediction, and does the ranking transfer between the two datasets?
- **RQ2 (carbon and mitigation).** What carbon dioxide equivalent does each model emit during training and during inference, and how much accuracy must be surrendered to close the fairness gaps?
- **RQ3 (proxies).** Does removing the protected attribute from the features keep it out of the decision, or do correlated features leak it back in?

## 2 Related Work

**Fairness definitions and measurement.** Surveys catalog the many definitions of fairness and their incompatibilities (Mehrabi et al., 2021). Equalized odds and its relaxation, equality of opportunity, formalize error-rate parity across groups (Hardt et al., 2016), and the Fairlearn toolkit implements demographic parity and equalized odds as standard metrics (Bird et al., 2020). Our measurement follows this family of definitions; the four-fifths rule, a hiring guideline, is used as a familiar pass line for parity ratios [CITE: NEEDED].

**Environmental cost of machine learning.** Strubell et al. document the energy and emissions cost of model training (Strubell et al., 2019), and Schwartz et al. propose Green AI, which asks that efficiency be reported alongside capability (Schwartz et al., 2020). Our carbon protocol instantiates this advice at small scale: every fit and every scoring pass is measured rather than estimated from hardware specifications alone [CITE: NEEDED].

**Models, methods and data.** The gradient-boosting baseline is XGBoost (Chen & Guestrin, 2016). Exponentiated gradient and reweighing are fairness-constrained or reweighted learners [CITE: NEEDED], and threshold optimization shifts decision thresholds per group [CITE: NEEDED]. SHAP provides the game-theoretic attribution used in our proxy analysis (Lundberg & Lee, 2017). Adult is the canonical income-prediction benchmark (Kohavi, 1996) and German Credit the canonical small credit benchmark [CITE: NEEDED].

## 3 Methodology

### 3.1 Datasets

**Adult.** Census income records loaded from OpenML (loader `load_adult`). The table in `results/eda_sex_income.csv` records 14695 `Female` and 30527 `Male` rows, with high-income shares of 0.1136 and 0.3125 respectively; Figure 1 shows this breakdown. The audit attribute is `sex`; `sex`, `race`, `fnlwgt` and the label are dropped from the features. The stratified split leaves 31655 training and 13567 test rows with a test positive rate of 0.2478; the test groups hold 4427 `Female` and 9140 `Male` rows with group positive rates of 0.1195 and 0.3100 (`results/baselines.csv`).

**German Credit.** Credit-risk records loaded from OpenML (loader `load_german_credit`) [CITE: NEEDED]. The audit attribute is age bucketed at 25, and `age` is dropped from the features. The split leaves 700 training and 300 test rows with a test positive rate of 0.7000; the test groups hold 253 `25+` and 47 `under25` rows with group positive rates of 0.7273 and 0.5532 (`results/baselines.csv`). The small size triggers the pipeline's `SMALL_SAMPLE_WARNING` on every run.

**Split protocol.** Both loaders use a stratified split with test fraction 0.3 drawn with data seed 42, held fixed for every model and seed in this study.

![Outcome rate by sex on Adult](../figures/fig1_outcome_by_sex.png)

**Figure 1.** High-income share by sex on Adult (`results/eda_sex_income.csv`).

### 3.2 Models and baselines

Four classifiers are trained with fixed hyperparameters (`src/train.py`): logistic regression, a random forest, XGBoost (Chen & Guestrin, 2016) and a multilayer perceptron. Each is trained under five model seeds, recorded as seeds 0 through 4 in `results/model_runs.csv`. Every accuracy claim is read against the majority-class baseline for the same split: 0.7522 on Adult and 0.7000 on German Credit (`results/baselines.csv`).

### 3.3 Metrics

Models are scored with accuracy, F1 and ROC-AUC, and with fairlearn's demographic-parity difference, demographic-parity ratio and equalized-odds difference (Bird et al., 2020; Hardt et al., 2016). Parity ratios are compared against the four-fifths rule [CITE: NEEDED]. Per-group accuracy and selection rate are recorded for every model and seed in `results/per_group_accuracy.csv`.

### 3.4 Mitigation

Three methods are applied to each dataset and compared before and after: threshold optimization, exponentiated gradient and reweighing [CITE: NEEDED]. Each method is measured against its own unmitigated baseline in `src/mitigation.py`: threshold optimization and reweighing are compared with the XGBoost predictor, and exponentiated gradient with the logistic-regression predictor, because the scaler used for the latter is fitted outside the estimator. Two of the three methods need the protected attribute at prediction time; this is noted as a deployment caveat in Section 5.

### 3.5 Carbon measurement

Every training fit and every inference pass is wrapped in a codecarbon tracker that reports carbon dioxide equivalent in grams and wall-clock seconds on a single machine (`src/train.py`, `src/fairness.py`). Training and inference are tracked separately. The measurements are treated as within-run, relative quantities only; absolute values are not portable across machines or loads [CITE: NEEDED].

### 3.6 Reproducibility

`python run_all.py` regenerates every table in `results/` and every figure in `figures/` in fixed order, stopping at the first failure. All randomness is seeded (data seed 42, model seeds 0 through 4). The SHAP analysis explains a fixed sample of 1000 test rows (`src/explain.py`).

## 4 Results

### 4.1 Accuracy against the baseline

Table 1 reports mean accuracy over the five seeds. On Adult, XGBoost leads with 0.8623 (sd 0.0000), followed by logistic regression 0.8421 (sd 0.0000), random forest 0.8372 (sd 0.0010) and the multilayer perceptron 0.8242 (sd 0.0031); all four clear the 0.7522 majority baseline. On German Credit, the random forest leads with 0.7493 (sd 0.0064), with logistic regression 0.7400 (sd 0.0000), XGBoost 0.7333 (sd 0.0000) and the multilayer perceptron 0.6980 (sd 0.0128); the multilayer perceptron sits below the 0.7000 majority baseline. F1 and ROC-AUC move with accuracy on Adult (best F1 0.7005, best ROC-AUC 0.9229, both XGBoost), while on German Credit F1 (0.8174 to 0.8362) exceeds accuracy for every model, reflecting the 0.7000 positive rate. The accuracy leader therefore changes with the dataset, so no model transfers as "best" (RQ1, Figure 2).

**Table 1.** Mean accuracy over seeds, with the majority baseline for reference (`results/model_summary.csv`, `results/baselines.csv`).

| Dataset | Model | Accuracy (sd) | F1 | ROC-AUC | Majority baseline |
|---|---|---|---|---|---|
| adult | logreg | 0.8421 (0.0000) | 0.6467 | 0.9022 | 0.7522 |
| adult | random_forest | 0.8372 (0.0010) | 0.6527 | 0.8875 | 0.7522 |
| adult | xgboost | 0.8623 (0.0000) | 0.7005 | 0.9229 | 0.7522 |
| adult | mlp | 0.8242 (0.0031) | 0.6367 | 0.8701 | 0.7522 |
| german_credit | logreg | 0.7400 (0.0000) | 0.8194 | 0.7758 | 0.7000 |
| german_credit | random_forest | 0.7493 (0.0064) | 0.8362 | 0.7620 | 0.7000 |
| german_credit | xgboost | 0.7333 (0.0000) | 0.8174 | 0.7287 | 0.7000 |
| german_credit | mlp | 0.6980 (0.0128) | 0.7857 | 0.7138 | 0.7000 |

### 4.2 Fairness gaps in unmitigated models

Every Adult parity ratio falls far below the four-fifths line: 0.3279 (logreg), 0.3385 (random forest), 0.3482 (XGBoost) and 0.3963 (multilayer perceptron). German Credit ratios are much closer to parity: 0.8737 (logreg), 0.9088 (random forest), 0.8156 (XGBoost) and 0.7882 (multilayer perceptron); the multilayer perceptron is the only model that falls below the line, and three of the four German Credit models meet it. Equalized-odds differences span 0.0713 to 0.0930 on Adult and 0.1109 to 0.2266 on German Credit.

Per-group numbers locate the gap. On Adult, Female accuracy ranges from 0.9015 to 0.9277 against 0.7867 to 0.8306 for Male, while selection rates range from 0.0836 to 0.1166 for Female and 0.2549 to 0.2941 for Male, a higher rate for Male in every model. The label itself is imbalanced in the same direction — high-income shares are 0.1136 for Female and 0.3125 for Male (Figure 1) — so the parity gap is partly a reflection of the label gap, which is why the equalized-odds numbers are reported alongside it. On German Credit, `under25` accuracy ranges from 0.5319 to 0.6000 against 0.7209 to 0.7771 for `25+`, a gap in the same direction on a test group of 47 rows (RQ1, Figure 2).

**Table 2.** Fairness metrics for unmitigated models over seeds (`results/model_summary.csv`).

| Dataset | Model | Parity difference | Parity ratio | Equalized-odds difference |
|---|---|---|---|---|
| adult | logreg | 0.1713 | 0.3279 | 0.0929 |
| adult | random_forest | 0.1864 | 0.3385 | 0.0914 |
| adult | xgboost | 0.1755 | 0.3482 | 0.0713 |
| adult | mlp | 0.1775 | 0.3963 | 0.0930 |
| german_credit | logreg | 0.0954 | 0.8737 | 0.1279 |
| german_credit | random_forest | 0.0767 | 0.9088 | 0.1109 |
| german_credit | xgboost | 0.1443 | 0.8156 | 0.2266 |
| german_credit | mlp | 0.1557 | 0.7882 | 0.1746 |

![Accuracy versus demographic-parity ratio](../figures/fig2_accuracy_vs_fairness.png)

### 4.3 Mitigation trade-offs

Table 3 and Figure 3 summarize the before-and-after comparison (RQ2). Exponentiated gradient nearly eliminates the demographic-parity difference on both datasets — 0.1713 to 0.0006 on Adult and 0.0954 to 0.0068 on German Credit — at an accuracy cost of 0.8421 to 0.8270 on Adult, while its German Credit accuracy rises from 0.7400 to 0.7507. Its equalized-odds difference, however, moves in the opposite direction: 0.0929 to 0.3058 on Adult and 0.1279 to 0.1536 on German Credit. Threshold optimization improves Adult equalized odds from 0.0713 to 0.0168 at an accuracy cost of 0.8623 to 0.8513, and leaves the German Credit equalized-odds difference unchanged at 0.2266 with accuracy 0.7333 to 0.7367. Reweighing produces moderate parity-ratio gains (0.3482 to 0.5371 on Adult, 0.8156 to 0.8646 on German Credit) with small accuracy changes (0.8623 to 0.8603 and 0.7333 to 0.7267), while its equalized-odds difference rises from 0.0713 to 0.1123 on Adult and falls from 0.2266 to 0.1664 on German Credit. The metric being optimized determines which method looks best: no method improves every metric at once.

**Table 3.** Mitigation before and after, mean over seeds (`results/mitigation_summary.csv`).

| Dataset | Method | Accuracy | Parity difference | Parity ratio | Equalized-odds difference |
|---|---|---|---|---|---|
| adult | threshold_optimizer | 0.8623 → 0.8513 | 0.1755 → 0.1197 | 0.3482 → 0.5318 | 0.0713 → 0.0168 |
| adult | exponentiated_gradient | 0.8421 → 0.8270 | 0.1713 → 0.0006 | 0.3279 → 0.9961 | 0.0929 → 0.3058 |
| adult | reweighing | 0.8623 → 0.8603 | 0.1755 → 0.1096 | 0.3482 → 0.5371 | 0.0713 → 0.1123 |
| german_credit | threshold_optimizer | 0.7333 → 0.7367 | 0.1443 → 0.1404 | 0.8156 → 0.8197 | 0.2266 → 0.2266 |
| german_credit | exponentiated_gradient | 0.7400 → 0.7507 | 0.0954 → 0.0068 | 0.8737 → 0.9913 | 0.1279 → 0.1536 |
| german_credit | reweighing | 0.7333 → 0.7267 | 0.1443 → 0.1033 | 0.8156 → 0.8646 | 0.2266 → 0.1664 |

![Mitigation trade-offs](../figures/fig3_mitigation_tradeoff.png)

### 4.4 Carbon cost

Table 4 reports training and inference emissions (RQ2, Figures 4 and 6). On Adult the multilayer perceptron dominates the budget at 0.264805 g training and 105.6844 s, against 0.00579552 g for XGBoost, 0.0158695 g for the random forest and 0.00108537 g for logistic regression. XGBoost delivers the highest accuracy, 0.8623, at a fraction of the multilayer perceptron's training emissions. On German Credit all training emissions are small, from 5.19829e-06 g for logistic regression to 0.00269756 g for the multilayer perceptron. Training exceeds inference in all eight model rows, and the spread across models is far larger than the spread between training and inference within a model. Because codecarbon reports at this scale are estimates from one machine, the ranking rather than the decimal places should be read from Figure 4.

**Table 4.** Mean training and inference emissions over seeds (`results/model_summary.csv`).

| Dataset | Model | Train CO2eq (g) | Inference CO2eq (g) | Train (s) | Inference (s) |
|---|---|---|---|---|---|
| adult | logreg | 0.00108537 | 5.06137e-06 | 0.4231 | 0.0343 |
| adult | random_forest | 0.0158695 | 0.00215555 | 6.8700 | 0.9265 |
| adult | xgboost | 0.00579552 | 0.000437789 | 2.0825 | 0.1456 |
| adult | mlp | 0.264805 | 6.48383e-05 | 105.6844 | 0.0509 |
| german_credit | logreg | 5.19829e-06 | 4.95236e-06 | 0.0180 | 0.0055 |
| german_credit | random_forest | 0.0012911 | 0.000461504 | 0.6760 | 0.2334 |
| german_credit | xgboost | 0.00118883 | 3.76994e-06 | 0.5730 | 0.0193 |
| german_credit | mlp | 0.00269756 | 3.38796e-06 | 1.3452 | 0.0067 |

![Training and inference emissions per model](../figures/fig4_carbon_per_model.png)

![Accuracy per gram of training CO2eq](../figures/fig6_accuracy_per_gram_co2.png)

### 4.5 Proxy features

Figure 5 and `results/proxy_features.csv` answer RQ3 for the Adult baseline. The highest-mean-attribution feature, marital-status_Married-civ-spouse, is also the strongest sex correlate: mean absolute SHAP value 1.2565 and correlation with sex 0.4336, with a feature value of 0.6184 for men against 0.1570 for women. hours-per-week follows with correlation 0.2324 (0.3915 mean absolute SHAP). Among the ten features with the largest mean absolute SHAP values — led by marital-status_Married-civ-spouse 1.2565, age 0.8016, capital-gain 0.5872, education-num 0.4970 and hours-per-week 0.3915 — one correlates with sex above 0.3 and all ten correlate to some degree, from 0.4336 down to -0.0045. Removing the sex column from the features therefore did not remove sex-correlated information from the model; correlation, however, is not the same as full recoverability of the attribute, which would require a dedicated probing classifier.

![SHAP summary](../figures/fig5_shap_summary.png)

![Top SHAP features](../figures/fig5b_shap_top10_bar.png)

## 5 Discussion

**The accuracy–fairness–carbon trade-off is real and three-dimensional.** On Adult the accuracy leader (XGBoost, 0.8623) is neither the parity-ratio leader (multilayer perceptron, 0.3963) nor the lowest emitter during training (logistic regression, 0.00108537 g); the parity-ratio leader is simultaneously the worst emitter (multilayer perceptron, 0.264805 g) and the least accurate model (0.8242). The multilayer perceptron buys its modest parity gain with both accuracy and emissions. On German Credit the random forest is the accuracy leader (0.7493) and the parity-ratio leader (0.9088), so there the three axes agree — but that agreement rests on a test set of 300 rows, of which 47 are in the audited group. The lesson for RQ1 and RQ2 is that the trade-off is dataset-dependent: it must be measured, not assumed.

**Mitigation is a metric-selection problem.** Exponentiated gradient optimizes demographic parity almost perfectly (0.0006 and 0.0068 after) while worsening equalized odds on both datasets (0.3058 and 0.1536 after). Threshold optimization shows the mirror image on German Credit: its equalized-odds difference is 0.2266 before and after, while accuracy and parity move slightly. A practitioner who picks a mitigation without first declaring the metric of record can easily make the other metric worse. Two of the methods (threshold optimization and exponentiated gradient) also require the protected attribute at prediction time, which restricts where they can be deployed; reweighing, which changes training weights only, does not have this constraint.

**Proxy variables limit what "remove the attribute" means.** The proxy probe shows that marital status, hours worked and occupation carry the removed signal (Section 4.5). Dropping a column is therefore a bookkeeping step, not a guarantee; the audit label must be retained so that the decision can still be measured per group. Note that the correlation analysis establishes association, not that the attribute can be reconstructed.

**Practical recommendations.** For teams auditing similar models: report accuracy against a majority-class baseline so that "high accuracy" has a floor; report both parity and error-rate metrics before and after mitigation; treat carbon measurements as within-machine relative rankings; keep the protected attribute available for evaluation even when it is removed from features; and expect the ranking of models to change when the dataset changes, as it did between Adult and German Credit here.

## 6 Limitations

- **Data provenance.** Adult is 1994 US census data (Kohavi, 1996); the social context it encodes has changed, so the fairness gaps describe that record, not present-day income distribution.
- **Protected attributes.** Only binary sex (Female and Male) on Adult and an age bucketed at 25 on German Credit are audited; intersectional groups are not modelled, and no non-binary category exists in these records.
- **Carbon scale.** The emissions are small and measured on one machine with codecarbon [CITE: NEEDED]; the same model re-trained under a different seed does not produce the same reading, as the strictly non-zero standard-deviation columns of `results/model_summary.csv` show, so absolute values should not be compared across machines or across days — only within-run rankings.
- **No causal claims.** All results are observational measurements on a single fixed split: correlations, metric gaps and mitigation deltas are descriptive, and nothing here identifies a causal effect of any feature or method on the outcomes.
- **German Credit sample size.** With 300 test rows and 47 in the `under25` group, per-group metrics for German Credit move by several points when a handful of rows change; the pipeline prints `SMALL_SAMPLE_WARNING` for exactly this reason.
- **Fixed split and no significance testing.** One stratified split and five model seeds measure model randomness, not split-to-split variation; no confidence intervals or hypothesis tests are reported, so small differences between models should not be read as significant.
- **Feature hygiene.** On German Credit, `personal_status` encodes sex and remains in the features; only the audited attribute (age) is removed there. Any claim that "all sensitive attributes are excluded" would therefore be false for that dataset.
- **Mitigation protocol.** Mitigation is evaluated at fixed hyperparameters, and threshold optimization and exponentiated gradient need the protected attribute at prediction time; no tuned accuracy–fairness frontier is explored.
- **Explanation scope.** SHAP values and the proxy probe cover the Adult baseline only, and the proxy analysis reports correlations rather than a trained attribute-recovery attack.

## 7 Conclusion

This study measured accuracy, fairness and carbon for the same models on the same splits, and the three axes disagreed more often than they agreed. The accuracy leader changed between Adult and German Credit; every Adult model fell short of the four-fifths rule while three of four German Credit models met it; mitigation shrank the parity gaps at a modest accuracy cost but could widen the equalized-odds gap; training dominated inference in measured emissions; and the removed attribute remained present through correlated features, most strongly marital status. Auditing one axis in isolation would have missed each of these facts. The full pipeline, from data loading to figures, regenerates every number reported here with a single command, so the audit can be repeated — and extended to further datasets, protected attributes and emission scales — as it stands.

## References

The reference list is maintained separately in [`references.md`](references.md).
