# Windows setup and validation

These commands assume the extracted folder is:

`D:\work\job\lss-sbi-lab`

## PowerShell

```powershell
cd D:\work\job\lss-sbi-lab

# Python 3.12 is recommended. Replace 3.12 if another supported version is installed.
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m unittest discover -s tests -v
```

Expected final lines:

```text
Ran 8 tests in ...
OK
```

## Command Prompt

```bat
cd /d D:\work\job\lss-sbi-lab
py -3.12 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Optional quick benchmark run

The full baseline/robust runs are computationally larger. For a quick end-to-end check:

```powershell
python sbi_lss_lab.py --n-train 100 --n-validation 30 --n-calibration 20 --n-misspecification 10 --n-posterior-samples 200 --epochs 10 --output-dir results/windows_smoke
```

Do not use the smoke-run numbers in the application; the application results come from the archived final runs described in `RESULTS.md`.
