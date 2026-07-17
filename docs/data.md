# Data management

Generated and processed data are controlled by DVC and excluded from Git.

The default `prepare` stage creates deterministic synthetic data grouped by
synthetic subject. Replace this stage with an approved dataset adapter for real
experiments; do not commit patient data or credentials.

Configure a remote explicitly, for example:

```bash
dvc remote add -d storage s3://YOUR-BUCKET/ecg-denoising
dvc remote modify storage region eu-west-2
dvc push
```

Keep authentication in the provider's credential chain or environment, never
in `.dvc/config`. Local secrets can be placed in `.dvc/config.local`, which DVC
does not commit.

All splitting must occur by subject and, for augmented data, by original signal
and noise source. Derived windows from one source must not cross split boundaries.
