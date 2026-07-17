# Reproducible reference run

This snapshot records the default CPU run defined by [`params.yaml`](../../params.yaml)
and the content hashes in [`dvc.lock`](../../dvc.lock). It exists so reviewers can
inspect a result without downloading DVC artifacts or trusting a headline claim.

| Measure | Result |
| --- | ---: |
| Held-out records / subjects | 32 / 4 |
| Input SNR | 4.685 dB |
| Output SNR | 5.655 dB |
| Mean SNR improvement | +0.970 dB |
| Normalised correlation | 0.861 |
| RMSE | 0.0805 |
| Worst SNR-stratum change | -4.483 dB at 24 dB input SNR |
| Promotion decision | **Rejected** |

The aggregate gate passes, but the worst-case gate fails because the network
degrades already-cleaner synthetic inputs. The candidate is therefore not
promoted. This is a software-validation experiment on generated data, not a
clinical-performance result.

Machine-readable evidence: [`data_manifest.json`](data_manifest.json),
[`metrics.json`](metrics.json), [`metrics_by_snr.csv`](metrics_by_snr.csv),
[`training_summary.json`](training_summary.json) and
[`promotion.json`](promotion.json).
