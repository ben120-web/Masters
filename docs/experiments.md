# Experiment protocol

1. Version code and parameters in Git.
2. Version datasets and model outputs through DVC.
3. Run `dvc repro` from a clean environment.
4. Inspect learning curves in TensorBoard.
5. Compare parameters, metrics and artifacts in MLflow.
6. Use `dvc metrics diff` and inspect `reports/promotion.json` before accepting
   a candidate. Aggregate and worst-stratum SNR gates are configured in
   `params.yaml`.
7. Reject a candidate when any gate fails; never average away a harmful slice.
8. Validate the candidate on a locked, subject-held-out external dataset.
9. Record model approval and limitations in the model card.

The default local MLflow store is suitable for individual iteration. A shared
deployment should use a database-backed tracking server and access-controlled
artifact store.

The committed [reference result](../reports/reference/README.md) is deliberately
rejected: it improves the aggregate metric while degrading high-SNR inputs.
That result is retained as honest evidence and as a regression target for the
next modelling iteration.
