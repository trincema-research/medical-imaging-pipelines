# RSNA 2024 evaluation metric

Official Kaggle score: **sample-weighted log loss**, lower is better. Implementation: [`best_config_runs/kaggle_metric.py`](../best_config_runs/kaggle_metric.py) (`score` matches the competition notebook). Local tables: [`data/best_config_runs/results/kaggle_metric_scores.csv`](../data/best_config_runs/results/kaggle_metric_scores.csv).

This is **not** the unweighted multi-head cross-entropy used in NAS. Local numbers use ground-truth-centered \(64\times64\) crops and the 70/15/15 study split — not the hidden test set.

---

## 1. Row, weights, and per-group loss

Each labelled site is one row: study, condition, lumbar level. The label is one-hot over three severities \(c \in \{\text{normal/mild},\ \text{moderate},\ \text{severe}\}\). Sample weights are \(w \in \{1, 2, 4\}\) in that order (severe counts \(4\times\)).

Let \(\mathbf{y}_i \in \{0,1\}^3\) be the one-hot label, \(\hat{\mathbf{p}}_i\) the predicted simplex, and \(w_i\) the sample weight. For a row group \(G\),

\[
\mathcal{L}_G
=
-\,
\frac{
\sum_{i \in G} w_i \sum_{c} y_{i,c}\,\log \hat{p}_{i,c}
}{
\sum_{i \in G} w_i
}.
\]

That is `sklearn.metrics.log_loss` with `sample_weight`. Three groups are scored separately:

| group | rows |
|---|---|
| spinal | spinal canal stenosis \(\times\) 5 levels |
| foraminal | left + right neural foraminal narrowing \(\times\) 5 levels |
| subarticular | left + right subarticular stenosis \(\times\) 5 levels |

Averaging the three group losses (next section) keeps spinal from being drowned out: it has 5 sites per study, the other two groups have 10 each.

---

## 2. Any-severe spinal

For study \(s\), let \(G_s\) be its spinal rows. The auxiliary binary label and prediction are

\[
z_s = \max_{i \in G_s} y_{i,\text{severe}},
\qquad
\hat{z}_s = \max_{i \in G_s} \hat{p}_{i,\text{severe}},
\qquad
\tilde{w}_s = \max_{i \in G_s} w_i.
\]

\(\mathcal{L}_{\text{any-severe}}\) is the sample-weighted binary log loss of \(\{(z_s,\hat{z}_s,\tilde{w}_s)\}_s\). Official weight \(\lambda = 1\).

---

## 3. Reported score

\[
S
=
\frac{
\mathcal{L}_{\text{spinal}}
+
\mathcal{L}_{\text{foraminal}}
+
\mathcal{L}_{\text{subarticular}}
+
\lambda\,\mathcal{L}_{\text{any-severe}}
}{
3+\lambda
}
\qquad (\lambda=1).
\]

Private leaderboard (1,874 teams): 1st **0.389**, last gold **0.404**.

---

## 4. Suggested Methods wording

> Models were scored with the RSNA 2024 competition metric: sample-weighted log loss with weights \(1/2/4\) for normal/mild, moderate, and severe. The loss was computed separately on spinal, foraminal, and subarticular rows so that the five spinal sites were not dominated by the ten foraminal and ten subarticular sites, then averaged with an auxiliary any-severe spinal term (binary log loss of the per-study max of spinal severe labels and of spinal severe probabilities; weight \(\lambda=1\)). Lower is better. Local scores used ground-truth-centered \(64\times64\) crops on a 70/15/15 study-level split, not the hidden Kaggle test set.

---

## 5. Local NAS-winner scores (test)

Retrain checkpoints, split seed 42, `any_severe_scalar=1`. MaxViT had no `best_model.pt`. ConvNeXt 3D RFN produced non-finite logits; those rows were dropped (test \(n\) 5650 vs 7130).

| architecture | representation | \(S\) (test) |
|---|---|---|
| ConvNeXt | 2D | **0.489** |
| ViT | 2D | 0.598 |
| EfficientNet | 2D | 0.638 |
| ViT | 2.5D-to-3D | 0.795 |
| EfficientNet | 2.5D-to-3D | 0.828 |
| ConvNeXt | 2.5D-to-3D | 0.927 |

```bash
python -m rsna2024_lumbar.best_config_runs kaggle-score
```
