# Dataset provenance

This project uses the public IBM Telco Customer Churn sample from the archived
IBM code-pattern repository:

- source repository: `IBM/telco-customer-churn-on-icp4d`
- source file: `data/Telco-Customer-Churn.csv`
- download script: `src/download_data.py`
- rows: 7,043
- expected SHA-256:
  `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`

`data/raw/` is intentionally git-ignored. A fresh clone and GitHub Actions fetch
the source file with `python src/download_data.py`; the script verifies the
checksum before writing the local copy. This keeps the public repository small
and makes the exact input reproducible without silently accepting an upstream
file change.

IBM's source repository is distributed under the Apache License 2.0. A copy of
that upstream licence is retained as `data/IBM_APACHE_LICENSE.txt` for clear
provenance. The portfolio project's own code is licensed separately under the
repository-level MIT `LICENSE`.

The sample is used only as a public demonstration dataset. Nothing in this
repository should be read as evidence of access to a telecommunications
company's private customer records.
