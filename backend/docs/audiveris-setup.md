# Audiveris setup

FingerFlow runs Audiveris as a separate command-line program. Audiveris is not
a Python package, so `pip install` does not install it.

## 1. Install Audiveris

Download a current binary release from the official Audiveris project:

- Releases: <https://github.com/Audiveris/audiveris/releases>
- Handbook: <https://audiveris.github.io/audiveris/_pages/handbook/>

Use the packaged release for your operating system where possible. It provides
the launcher and the runtime expected by that release. If you build Audiveris
from source, install the JDK version stated in that release's documentation.

After extracting or installing it, locate the launcher:

- Windows MSI: `C:\Program Files\Audiveris\Audiveris.exe`
- Windows archive/source distribution: `...\bin\Audiveris.bat`
- macOS/Linux: `.../bin/Audiveris`

## 2. Configure FingerFlow

Set `AUDIVERIS_CMD` to the full launcher path.

PowerShell, current terminal:

```powershell
$env:AUDIVERIS_CMD = "C:\Program Files\Audiveris\Audiveris.exe"
```

PowerShell, persist for your user account:

```powershell
[Environment]::SetEnvironmentVariable(
  "AUDIVERIS_CMD",
  "C:\Program Files\Audiveris\Audiveris.exe",
  "User"
)
```

macOS/Linux:

```bash
export AUDIVERIS_CMD="/opt/audiveris/bin/Audiveris"
```

Alternatively, add the launcher's `bin` directory to `PATH`, or pass
`--audiveris "path/to/launcher"` to the test script.

## 3. Verify the installation

PowerShell:

```powershell
& $env:AUDIVERIS_CMD -version
```

macOS/Linux:

```bash
"$AUDIVERIS_CMD" -version
```

The command should print the Audiveris, Java, OS, and OCR-engine versions.

## 4. Run the end-to-end smoke test

From the FingerFlow repository root:

```bash
pip install -r backend/requirements.txt
python scripts/test_omr_pipeline.py "path/to/score.jpg"
```

Or provide the launcher directly:

```bash
python scripts/test_omr_pipeline.py "path/to/score.jpg" \
  --audiveris "C:\Program Files\Audiveris\Audiveris.exe"
```

The test:

1. preprocesses the image with OpenCV;
2. runs Audiveris in non-GUI batch mode;
3. asks Audiveris to export uncompressed MusicXML;
4. parses the result and verifies a MusicXML score root element.

Test artifacts are written under `tmp/omr-tests/`.

## Troubleshooting

### `Audiveris was not found`

Check that `AUDIVERIS_CMD` points to the launcher file, not just the Audiveris
installation folder. Open a new terminal after setting a persistent variable.

### Java/runtime errors

Prefer the packaged Audiveris distribution. For a source build, confirm that
`java -version` matches the requirement for your Audiveris release.

### No MusicXML is produced

Try the image in the Audiveris GUI. OMR can fail when a photo is blurred,
cropped through staves, strongly warped, or contains unsupported notation.
Inspect the preprocessed image in the test output directory and preserve
Audiveris's error text when reporting a failure.

### Processing takes a long time

The first run can be slower while Java and OCR components initialize. The
integration defaults to a five-minute timeout, which can be changed through
the `timeout_seconds` argument of `recognize_score`.
