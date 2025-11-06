# store_insights

## Byte-compiling the uploader script

The `python -m compileall` helper expects paths relative to your current
working directory. Run the command from the repository root so the script is
found correctly:

```
python -m compileall scripts/vfp_dbf_to_rdsv2.py
```

If you are already inside the `scripts/` directory, drop the leading folder
and compile the file directly:

```
python -m compileall vfp_dbf_to_rdsv2.py
```

Both invocations point to the same file; the key difference is the relative
path supplied to `compileall`.

