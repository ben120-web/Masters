# Experiment protocol

1. Version code and parameters in Git.
2. Version datasets and model outputs through DVC.
3. Run `dvc repro` from a clean environment.
4. Inspect learning curves in TensorBoard.
5. Compare parameters, metrics and artifacts in MLflow.
6. Use `dvc metrics diff` before accepting a candidate.
7. Validate the candidate on a locked, subject-held-out external dataset.
8. Record model approval and limitations in the model card.

The default local MLflow store is suitable for individual iteration. A shared
deployment should use a database-backed tracking server and access-controlled
artifact store.
