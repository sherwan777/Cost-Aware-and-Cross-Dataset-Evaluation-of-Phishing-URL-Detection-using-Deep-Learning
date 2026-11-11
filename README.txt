PHISHING DETECTION PROJECT: COST BASED THRESHOLDS AND GENERALIZATION
Abdullah Jamil and Nausherwan Baig

data/phishing.csv                 original dataset used by Karim et al. (Kaggle copy of UCI)
data/dataset_B_05_2020.csv        Mendeley dataset (Hannousse and Yahiouche), 11,430 raw URLs
common.py                         shared settings, models and metric functions (do not run this one)
url_features.py                   extracts the 8 URL only features from a raw URL (do not run this one)
phase1_data.py                    Phase 1: clean the data and make leakage safe splits
phase2_models.py                  Phase 2: rebuild the paper's models, tune, calibrate, evaluate
phase3_threshold.py               Phase 3: cost based threshold, infeasible cases, sensitivity, error analysis
phase4_generalization.py          Phase 4: train on original data, test on the Mendeley dataset
requirements.txt                  Python packages needed


Phase 1  Data and leakage control  (proposal: "Data Splitting and Leakage Control")
         - recodes labels so phishing = 1 (in the file, -1 means phishing)
         - removes the 5,205 exact duplicate rows
         - flags the 64 feature patterns that appear with both labels (kept, not relabelled)
         - makes train / validation / test splits (60 / 20 / 20), stratified,
           with identical feature patterns always kept in the same split, for 5 seeds
         - shows how much leakage the paper style random 70/30 split has (about 62 percent)
           and that our split has 0 percent

Phase 2  Rebuild the models  (proposal: "Rebuilding and Improving the Models")
         - logistic regression (replaces linear regression), naive Bayes, KNN, decision tree,
           random forest, gradient boosting, SVC (trained until it converges), a small MLP,
           and the LSD ensemble (LR + SVC + DT) with soft and hard voting
         - hyperparameters tuned by 5 fold cross validation inside the training set only
         - probabilities calibrated with Platt scaling on the validation set
         - test results at threshold 0.5 (for comparison with the paper), plus ROC AUC,
           PR AUC, Brier score and calibration error (ECE) before and after calibration
         - canopy feature selection is left out (the paper does not describe it)

Phase 3  Cost based threshold  (proposal: "Cost Based Threshold Selection" and "Sensitivity and Error Analysis")
         - cost of a false alarm C_FP = 1, cost of a missed phishing site C_FN = 2, 10 or 50
         - compares 5 ways of picking the threshold tau:
              default_0.5                               the usual cutoff
              analytic_tau_star                         tau* = C_FP / (C_FP + C_FN)
              searched_cost_only                        lowest cost on validation
              searched_with_recall_floor                lowest cost with recall >= 0.95
              searched_with_recall_and_precision_floor  lowest cost with recall >= 0.95 and precision >= 0.90
         - tau is always chosen on the validation set, the test set is only used to report
         - if no tau meets the floors, it is reported as infeasible (not forced)
         - checks whether the most accurate model is also the cheapest
         - draws the ROC curve with the chosen operating points
         - sensitivity: changes only cheap URL features of test phishing sites to their
           legitimate value and measures how much recall drops at the chosen tau
         - error analysis: what the still missed phishing sites have in common

Phase 4  Generalization  (proposal: "Generalization to an Independent Dataset")
         - computes the 8 URL only features from the raw Mendeley URLs with one extractor,
           using the original rules (for example URL length cut points 54 and 75)
         - trains URL only models on the original data (same deduplicated rows and splits)
         - picks tau on the ORIGINAL validation set, then applies it to Mendeley without retuning
         - reference: trains and tests inside Mendeley, with splits grouped by domain
         - reports the gap between the datasets and a feature distribution comparison

STEP BY STEP: HOW TO RUN 

You only need to do steps 1 to 4 once.

1. Check that Python 3 is installed. Open the Terminal app and type:

       python --version

   You should see Python 3.9 or newer. If you get an error, install Python from
   https://www.python.org/downloads/ and try again.

2. Unzip the project and go into the folder. If you unzipped it in Downloads:

       cd ~/Downloads/phishing_project

   (Tip: type "cd " with a space and then drag the folder into the Terminal window.)

3. Create a virtual environment (a private space for the packages) and turn it on:

       python -m venv venv
       source venv/bin/activate

   Your prompt should now start with (venv). Every time you open a new Terminal window,
   go to the folder again and run "source venv/bin/activate" before running anything.

4. Install the packages:

       pip install -r requirements.txt

5. Run the phases IN ORDER. Each one saves a log you can send me.

       python phase1_data.py | tee results_phase1_log.txt
       python phase2_models.py | tee results_phase2_log.txt
       python phase2_models.py --drop-ambiguous | tee results_phase2_noamb_log.txt
       python phase3_threshold.py | tee results_phase3_log.txt
       python phase4_generalization.py | tee results_phase4_log.txt

   Approximate time on a laptop: Phase 1 a few seconds, Phase 2 about 2 minutes,
   Phase 3 under 1 minute, Phase 4 about 5 minutes.

   The second Phase 2 command is the extra experiment from the proposal: the same
   models trained without the 128 ambiguous rows, to see how much they matter.

OUTPUT FILES (all in the results folder)

phase1_summary.csv                    counts after cleaning
phase1_leakage_check.csv              test rows that also appear in training, paper split vs ours
phase1_split_sizes.csv                size and phishing share of each split

phase2_summary.csv                    main results table, mean and std over 5 seeds
phase2_metrics_per_seed.csv           the same per seed
phase2_predictions.csv                validation and test probabilities (used by Phase 3)
phase2_best_params.json               chosen hyperparameters per seed
phase2_summary_no_ambiguous.csv       same as above without the ambiguous rows
models/                               saved models (used by Phase 3, no need to send)

phase3_summary.csv                    every model x cost ratio x threshold policy
phase3_infeasible.csv                 combinations where no threshold met the floors
phase3_accuracy_vs_cost_ranking.csv   most accurate vs cheapest model
phase3_roc_operating_points.png       ROC curve with the chosen thresholds
phase3_sensitivity_summary.csv        recall drop after cheap URL changes
phase3_error_analysis.csv             features of missed vs caught phishing sites

phase4_feature_distribution.csv       URL feature values in the original vs Mendeley data
phase4_roc_auc_gap.csv                ROC AUC inside the original data vs on Mendeley
phase4_gap_ratio10.csv                cost, missed phishing, recall, precision per setting
phase4_summary.csv                    everything from Phase 4

Key columns:
  cost_per_1000               (C_FN x FN + C_FP x FP) per 1,000 URLs, lower is better
  missed_phishing_per_1000    phishing sites let through per 1,000 URLs
  fpr                         share of legitimate sites wrongly blocked
  ece                         calibration error, lower means the probabilities are more honest
  feasible_seeds_out_of_5     in how many seeds a threshold meeting the floors existed

======================================================================
CHANGING SETTINGS
======================================================================

All settings are at the top of common.py:
  SEEDS          the 5 random seeds
  COST_RATIOS    [2, 10, 50]
  RECALL_MIN     0.95
  PRECISION_MIN  0.90

If you change anything, rerun from Phase 2 onward.

======================================================================
COMMON PROBLEMS
======================================================================

"No such file or directory: data/clean.csv"  -> run phase1_data.py first
"No such file ... phase2_predictions.csv"    -> run phase2_models.py before phase3
"command not found: python"                  -> the venv is not on; run source venv/bin/activate
"ModuleNotFoundError"                        -> run pip install -r requirements.txt again with the venv on
