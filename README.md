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

## Status

Work in progress: data pipeline and first experiments. Run notes and links will follow.
