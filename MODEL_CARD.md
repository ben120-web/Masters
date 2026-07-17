# Model card: ECG motion-artefact denoiser

## Summary

The maintained reference model is a residual 1-D convolutional network that
predicts motion artefact and subtracts it from a single-lead ECG window.

## Intended use

- Research into ECG signal quality and motion-artefact removal.
- Reproducible comparison of denoising architectures and training protocols.
- Educational demonstration of an end-to-end biomedical MLOps workflow.

## Out-of-scope use

- Diagnosis, triage or treatment decisions.
- Unvalidated use on sampling rates, lead configurations, devices or patient
  populations different from the evaluation data.
- Claims of clinical efficacy without prospective external validation.

## Inputs and outputs

The model accepts normalised single-lead ECG segments shaped `(batch, 1, time)`
and returns denoised segments of the same shape. Sampling rate and segment length
are recorded with each checkpoint.

## Evaluation

The pipeline reports RMSE, normalised correlation coefficient and SNR
improvement on a subject-held-out test split. The versioned
[reference run](reports/reference/README.md) records +0.970 dB mean SNR
improvement, but -4.483 dB in the cleanest synthetic stratum. It is correctly
marked **rejected** by the worst-case promotion check. This generated-data run
validates software behaviour only; it is not a clinical benchmark.

## Limitations

The default data generator is synthetic and exists to validate the software
pipeline, not clinical performance. Real-world evaluation must account for
device characteristics, demographics, arrhythmias, electrode placement and
motion types. Denoising may suppress diagnostically relevant morphology.
