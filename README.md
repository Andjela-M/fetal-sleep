# fetal-sleep
This repository contains the code and non-identifiable source data to quantify associations between maternal and fetal rhythms using multi-level models, and to identify predictors of early sleep regulation from the fetal period to 6 months of age using random forest analyses. 

Archived release: https://doi.org/10.5281/zenodo.21842441

For questions regarding reuse or collaboration, please contact: andjela.markovic@unibe.ch

## Multi-level models
### `fetalSleep_multiLevelModel.RData`

Contains the input data required to run `fetalSleep_multiLevelModel.R` and reproduce the multi-level linear mixed-effects analyses of fetal activity.

This participant-level input dataset is not included in this public repository. It is available under restricted access on Zenodo at https://doi.org/10.5281/zenodo.22826738.

The file contains the following object:

| Object | Description |
|---|---|
| `allData` | Long-format dataset containing repeated observations of fetal and maternal signals across recording days and participants. Variables used in the models include fetal activity (`bellyAbsZ`), maternal activity (`watchZ`), maternal wrist temperature (`watchtemperatureZ`), maternal vigilance state (`state`), gestational age (`age`), recording day (`day`), participant identifier (`subject`), and the covariates twin pregnancy (`twins`), in-vitro fertilization (`ivf`), diabetes (`diabetes`), and nulliparity (`nulliparity`). |

The corresponding script, `fetalSleep_multiLevelModel.R`, fits a series of linear mixed-effects models using `lmerTest`, with fetal activity (`bellyAbsZ`) as the outcome. Models of increasing complexity are evaluated, including a null model with a participant-level random intercept, a fixed-effects model, a model including participant-specific random intercepts and slopes for recording day, and a final interaction model.

The final interaction model includes interactions between maternal vigilance state, gestational age, and maternal wrist temperature, as well as between maternal vigilance state, gestational age, and maternal activity. Twin pregnancy, in-vitro fertilization, diabetes, and nulliparity are included as additional covariates. Model fit is summarized using AIC and functions from the `performance` package.

## Random forest analyses
### `fetalSleep_randomForest.mat`

Contains the input data required to run `fetalSleep_randomForest.m` and reproduce the random forest analyses, including feature-importance estimates and partial dependence plots for early postnatal, 3-month, and 6-month day/night sleep ratios.

This participant-level input dataset is not included in this public repository. It is available under restricted access on Zenodo at https://doi.org/10.5281/zenodo.22826738.

The file contains the following variables:

| Variable | Description |
|---|---|
| `newbornDetTab` | Predictor table for the early postnatal sleep model (`32 × 9`). Includes sex, birth mode, breastfeeding, age at assessment, fetal day/night sleep ratio, maternal sleep regularity index (SRI), maternal wrist-temperature phase, breast milk melatonin, and infant stool melatonin. |
| `ratioNewbornSleep` | Early postnatal day/night sleep ratio used as the outcome variable (`32 × 1`). |
| `mo3DetTab` | Predictor table for the 3-month sleep model (`30 × 10`). Includes sex, birth mode, breastfeeding, age at assessment, fetal day/night sleep ratio, maternal SRI, maternal wrist-temperature phase, early postnatal day/night sleep ratio, breast milk melatonin, and infant stool melatonin. |
| `ratio3moSleep` | 3-month day/night sleep ratio used as the outcome variable (`30 × 1`). |
| `mo6DetTab` | Predictor table for the 6-month sleep model (`29 × 11`). Includes sex, birth mode, breastfeeding, age at assessment, fetal day/night sleep ratio, maternal SRI, maternal wrist-temperature phase, early postnatal day/night sleep ratio, breast milk melatonin, infant stool melatonin, and 3-month day/night sleep ratio. |
| `ratio6moSleep` | 6-month day/night sleep ratio used as the outcome variable (`29 × 1`). |

The corresponding script, `fetalSleep_randomForest.m`, performs 10-fold cross-validation, random forest feature selection and model fitting, and generates the partial dependence plots shown in the manuscript.

## Additional source data

Additional source data generated in this study are provided with the published article. The corresponding DOI will be added upon publication.

Source data underlying the exemplary abdominal acceleration signals shown in Figure 2 are provided in this repository as `exemplaryAbdominalSignals.zip`.

The archive contains:

- `Figure2_left_measured.mat` – abdominal acceleration data underlying the left example shown in Figure 2.
- `Figure2_right_measured.mat` – abdominal acceleration data underlying the right example shown in Figure 2.

## Citation

The archived version of this repository is available on Zenodo:

Markovic, A. (2026). Andjela-M/fetal-sleep: fetal-sleep v1.0.0 (Version v1.0.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.21842442
