# Predicting MLB All-Star Selections with Machine Learning

Group project by **Vincent Rupp and Sean Littmann**, testing a wide range of machine learning methods to predict whether an MLB hitter would be selected as an All-Star the following season, using player season stats from Stathead covering 10 full seasons (2012–2023, excluding 2019 and 2020 because of COVID).

## The question

Can a player's current-season hitting stats predict whether they'll make the All-Star team the *next* season?

## Data

- 4,514 rows (player-seasons) x 31 features
- Traditional stats (AVG, HR, PA), advanced metrics (OPS+), team and position data, and current All-Star status
- Target variable: `AS_next_year` (Yes/No)
- Pitchers and players with fewer than 100 plate appearances were excluded
- All-Star selections are rare (~9% prevalence), so class imbalance was a central challenge throughout

## Approach

- Cleaned the data: converted categoricals to factors, dropped redundant/derived stats (OPS, TB, H, AB, G, Player_ID), and applied min-max normalization since features like OBP (0.2–0.5) and PA (100–800) were on very different scales
- Trained and tuned 11 models via `caret`: KNN, Logistic Regression, Naive Bayes, Decision Tree, Rules (JRip), SVM, Bagging, Boosting, Random Forest, XGBoost, and a Neural Network
- Used 10-fold cross-validation for every model, with a tuning grid per model and threshold tuning (for probability-based models) to optimize F1 score
- Evaluated primarily on **F1 score** rather than accuracy, since accuracy would reward a model that just predicts "No" every time given the class imbalance

## Results

Model rankings by F1 score:

1. XGBoost — 0.4844
2. Random Forest — 0.4832
3. Logistic Regression — 0.4681
4. Boosting — 0.4593
5. Decision Tree — 0.4294
6. Neural Network — 0.4110
7. Bagging — 0.4103
8. Support Vector Machine — 0.3852
9. Naive Bayes — 0.3733
10. K-Nearest Neighbors — 0.3487
11. Rules (JRip) — 0.3360

XGBoost and Random Forest performed best, though none of the models broke an F1 of 0.5 — not strong enough to rely on outside of this project. Logistic Regression offered a good balance of interpretability and consistent sensitivity despite the simpler approach.

## What we'd do differently

- Incorporate WAR (Wins Above Replacement) as a feature — it's widely considered the best single measure of player value and likely would have improved every model
- All-Star selection itself is a messy target: it blends fan voting, manager selections, and player voting, and fan-vote share is skewed by team popularity (a Yankees player with millions of Instagram followers has a real edge over an equally productive player on a smaller-market team) — stats alone can only explain so much of that process

## Files

- `Executive Summary ML Project.pdf` — write-up of methodology, results, and takeaways
- `Portfolio Document ML Project.pdf` — full portfolio document, co-authored with Sean Littmann
- `Code from ML Project.pdf` — full model code and output (the project's R code is provided as a PDF printout; the Stathead data export is not redistributed)

## Authors

Vincent Rupp and Sean Littmann

---

Built by Vincent Rupp. Shared for portfolio and review purposes; please get in touch before reusing it.
