# tools/ — acceptance checker for the frozen configuration

This directory holds the checker used to verify that **freezing the campaign's
CLI options into module constants changed no value**.

## What it checks

Run against a baseline commit (the tree just before the freeze started), it
compares the two revisions:

| Stage | Checks |
|---|---|
| `S0` | every `.py` compiles |
| `S1` | the set of surviving flags equals the whitelist, in both directions |
| `S2value` | each frozen flag has an `UPPER_CASE` constant with the **same value and type** as its old argparse default |
| `S2whitelist` | a whitelisted flag may not disappear from any file |
| `S3orphan` | a frozen flag's `args.<dest>` may not still be used |
| `S4` | every `args.<x>` read is declared somewhere |
| `S5` | a `required=True` flag must survive (it has no default to freeze) |
| `S6` | flags cited in runnable documented commands must exist |
| `S7` | function and class signatures are unchanged |
| `S8` | with `--code-only`, the AST minus docstrings is identical (used to prove comment-only edits) |
| `S9` | the `_config` block of the shipped `summary.json` is still reproducible from the code |

Exit code is the number of failures. `WARN` does not block.

## Usage

```bash
python3 tools/flag_audit.py --base b84926d --work . \
    --whitelist tools/whitelist.txt \
    --deviations tools/deviations.json
```

`b84926d` is the last commit before the first `freeze:` commit. The tag
`paper-submission` (`df1affd`) is the state before the whole cleanup.

To say "only comments may have changed here":

```bash
python3 tools/flag_audit.py --base b84926d --work . \
    --whitelist tools/whitelist.txt --deviations tools/deviations.json \
    --code-only src/dataset/verify_maps.py src/common/domain_scaling.py
```

## Files

| File | Meaning |
|---|---|
| `flag_audit.py` | the checker |
| `whitelist.txt` | the flags that deliberately **survive** as flags (machine-dependent, paths, paper sweeps and sample sizes) |
| `deviations.json` | the two options whose frozen value deliberately differs from its argparse default, with the authorised value |

## The two authorised deviations

```json
{ "--success-only": true, "--descriptor-bins": 16 }
```

* `--success-only` frozen to `True` (argparse said `False`): the six checkpoints
  all record `success_only: true`.
* `--descriptor-bins` frozen to `16`: the argparse default was already
  `OCCUPANCY_BINS = 16`. The `"default 8"` that used to appear in the help text
  was stale. Verified against the campaign's own leakage report
  (`16x16 occupancy`).
