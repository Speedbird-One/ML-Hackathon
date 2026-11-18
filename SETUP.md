# Setup (for teammates)

Requires **Python 3.11+** (the team venv uses 3.13). CPU only; no GPU needed.

## 1. Clone
```bash
git clone https://github.com/Speedbird-One/ML-Hackathon.git
cd ML-Hackathon
```

## 2. Create a virtual environment and install everything
**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Add the data
The raw data is **not** in git. Copy the organiser-provided `train/` and `student_test/` folders into `data/`
(see `data/README.md`), or point to them with an environment variable:
```powershell
$env:PATHSNAP_DATA = "C:\path\to\folder\containing\train_and_student_test"   # PowerShell
```
```bash
export PATHSNAP_DATA=/path/to/folder/containing/train_and_student_test       # bash
```

## 4. Check it works
```bash
python -c "from pathsnap.data import *; print(len(list_trips('train')), len(list_trips('student_test')))"
```
Expected output: `80 10`

## Team rules
- Never commit data, secrets, or the evaluator trips (`.gitignore` covers `data/`).
- Do not touch evaluator data during development.
- Add any new package to `requirements.txt` (pinned) in the same commit that uses it.
- Work on a branch per task (`git checkout -b feat/<name>`), merge via PR or a quick pull-rebase.

_More instructions will be added below as the project grows._
