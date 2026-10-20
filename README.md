# Predicting the labour force status of young Rwandans from survey microdata

Machine Learning summative project — African Leadership University.

Young Rwandans aged 16–30 have an unemployment rate well above the adult rate
and more than a quarter are not in employment, education or training. This
project asks whether a young person's labour force status — employed,
unemployed, or outside the labour force — can be predicted from the
education, vocational training, disability, migration, household and
unpaid-work information collected in the **Rwanda Labour Force Survey**
(NISR, annual rounds 2024 and 2025), and which of those factors matter most.
The answer is useful for anyone designing career-readiness programmes for
students in Rwanda.

Two modelling paradigms are compared on the same preprocessing pipeline:

* **Traditional ML (scikit-learn):** logistic regression, random forest,
  gradient boosting (XGBoost, LightGBM) and an RBF SVM, with randomised
  hyper-parameter search, class weighting and a leakage ablation (9 runs).
* **Deep learning (TensorFlow / Keras):** multilayer perceptrons built with the
  Functional API and fed through `tf.data` pipelines, varying regularisation
  (dropout, batch normalisation, L2, AdamW), capacity, learning rate, optimiser,
  entity embeddings and class weights (9 runs). Every run is logged to TensorBoard.

## Links

| Resource | Link |
|----------|------|
| Google Drive (raw data zip, saved models, TensorBoard logs, figures) | https://drive.google.com/drive/folders/1Ol9Xh2oEHdOokqBWSNi8RFORV1d6g2q_?usp=sharing |
| Video presentation | https://drive.google.com/file/d/1GS4BKHDEHfGgvfhpU4ekYHS1iEtH5SWQ/view?usp=sharing |
| Notebook on Colab (executed copy) | https://drive.google.com/file/d/1MXTtlccln8AZtB4gBdI_N_89mthOVT65/view?usp=sharing |

## Dataset

Rwanda Labour Force Survey public-use microdata, National Institute of
Statistics of Rwanda (NISR): 2024 round (102,562 persons, 197 variables) and
2025 round (100,852 persons, 207 variables). The modelling population is every
person aged 16–30 with a valid ILO labour force status: 53,503 people. The
raw files are **not** committed to this repository; the zipped dataset is in
the Google Drive folder above.

To download the files yourself (free, about five minutes):

1. Register at <https://microdata.statistics.gov.rw/index.php/auth/register> and confirm your email.
2. Open the catalogue entries
   [114 (2024)](https://microdata.statistics.gov.rw/index.php/catalog/114) and
   [125 (2025)](https://microdata.statistics.gov.rw/index.php/catalog/125),
   click **Get Microdata**, accept the public-use terms and download the Stata files.
3. Save them as `data/raw/RW_LFS2024.dta` and `data/raw/RW_LFS2025.dta`.

Access policy (from the catalogue): *Public use files, accessible to all. By
downloading, data users agree to maintain the confidentiality of the data and
data subjects.* Cite as: National Institute of Statistics of Rwanda, Rwanda
Labour Force Survey 2024 (2025), Version 0.1.

## Repository contents

```
notebooks/youth_labour_status_classification.ipynb   the whole project, runs top to bottom
requirements.txt                                     pinned dependencies
data/raw/                                            put the two .dta files here (git-ignored)
```

The notebook is self-contained. Section 0 defines the helper functions
(configuration, preprocessing, evaluation, traditional and deep models); the
remaining sections cover data description, exploratory analysis, preprocessing
and feature engineering, the 18 experiments, the experiment table, confusion
matrices, ROC and precision–recall curves, learning curves, permutation
importance, subgroup error analysis, model saving and the TensorBoard evidence
export. Running it creates `reports/` (experiment table, figures, logs
archive), `models/` and `logs/`, all git-ignored.

## Running locally

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/
```

Place the two Stata files in `data/raw/`, open the notebook and run all cells
(about 45 minutes on an 8-core laptop, CPU only). The random seed is fixed to
42 in the configuration cell. View the experiment logs with
`.venv/bin/tensorboard --logdir logs/`.

## Running on Google Colab

1. In Google Drive create a folder *ALU_ML_Summative* with a sub-folder `data/`
   containing the zipped raw dataset (or the two `.dta` files).
2. Open the notebook in Colab (File → Open notebook → GitHub, paste this
   repository's URL) and choose Runtime → Run all.
3. The first cell mounts Drive, installs the two packages Colab lacks and copies
   the data; the last cells copy every output (experiment table, figures,
   TensorBoard logs, saved models) back to *ALU_ML_Summative/outputs/* as
   evidence of the run.
