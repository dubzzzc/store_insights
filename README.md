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

## Building a Windows executable

Use [PyInstaller](https://pyinstaller.org) to bundle the uploader script into a
stand-alone `.exe`:

1. Create and activate a fresh virtual environment (recommended).
2. Install the uploader dependencies and PyInstaller:
   ```bash
   pip install -r scripts/requirements-uploader.txt
   pip install pyinstaller
   ```
3. Build the executable with the provided spec file:
   ```bash
   pyinstaller --clean --noconfirm scripts/vfp_dbf_to_rdsv2.spec
   ```

The bundled binary will be written to `dist/VFP_DBF_Uploader/VFP_DBF_Uploader.exe`.
Launch it normally to open the Tk GUI, or pass `--headless`/`--config` arguments
for background and scripted usage.
