# Team Workflow for a 5-Person Hackathon Group

## Roles

| Role | Responsibility |
|---|---|
| Integrator | Owns final notebook/submission. Merges only validated improvements. |
| Data/debug lead | Checks shapes, labels, min/max ranges, sample plots, scoring format. |
| Baseline lead | Gets the first working model and valid submission as quickly as possible. |
| Experimenter A | Tries architecture/preprocessing changes. |
| Experimenter B | Tries loss, augmentation, regularization, or hyperparameters. |

## Branch/notebook naming

Do not edit one notebook together. Copy a baseline notebook and name it clearly:

```text
01_image_classifier_baseline_uditha_aug_v1.ipynb
01_image_classifier_baseline_alex_lr_test.ipynb
02_denoiser_sam_loss_test.ipynb
```

## Experiment log rule

Every experiment must be logged in `experiments/experiment_log.csv` before it is considered for the final solution.

Minimum fields to fill in:

```text
owner, notebook, problem, idea, preprocessing, model, epochs, val_metric, leaderboard_metric, status, notes
```

## Merge rule

Only merge a change if it improves at least one of:

- validation score,
- leaderboard score,
- training stability,
- runtime,
- reproducibility,
- submission reliability.

Do not merge ideas because they seem more sophisticated.
