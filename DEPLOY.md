# Two-laptop deployment (API on Laptop 1, UI on Laptop 2)

```
 Laptop 2 (UI)                                   Laptop 1 (model + API)
 streamlit run ui/app.py                         uvicorn api.main:app --host 0.0.0.0 --port 8000
 ┌───────────────────────┐   GET  /trips         ┌──────────────────────────────────────────────┐
 │ list of 10 eval trips │ ───────────────────▶  │ artifacts/lgbm.joblib + inference_config.json │
 │ user picks a trip     │   POST /predict       │ data/evaluator/<trip>/  (road network)        │
 │ Folium map + metrics  │ ───────────────────▶  │ candidates → classifier → HMM/Viterbi         │
 │                       │ ◀─────────────────── │ JSON: route geometry + metrics + timings      │
 └───────────────────────┘   JSON response       └──────────────────────────────────────────────┘
```

Laptop 1 does all the ML. Laptop 2 needs only `streamlit`, `requests` and `folium`.

## 0. Network (do this first)
* Put **both laptops on the same phone hotspot**. College Wi-Fi often blocks laptop-to-laptop traffic
  (client isolation), so a hotspot is the reliable choice. The hotspot also gives Laptop 2 internet for the map tiles.
* On Laptop 1, find its IP address:
  * Windows: `ipconfig` → "IPv4 Address" under the Wi-Fi adapter (e.g. `192.168.43.10`)
  * macOS: `ipconfig getifaddr en0` · Linux: `hostname -I`

## 1. Laptop 1: model + API
```powershell
cd ML-Hackathon
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt            # already done if you followed SETUP.md
```
Put the evaluator trips in `data/evaluator/` (same layout as `student_test/`).

Start the API:
```powershell
$env:PATHSNAP_PARTITION = "evaluator"      # use "student_test" for rehearsals
$env:PATHSNAP_PRECOMPUTE = "1"             # decode all trips in the background so the demo is instant
uvicorn api.main:app --host 0.0.0.0 --port 8000
```
```bash
PATHSNAP_PARTITION=evaluator PATHSNAP_PRECOMPUTE=1 uvicorn api.main:app --host 0.0.0.0 --port 8000
```
Check it locally by opening `http://localhost:8000/docs` in a browser.

**Windows firewall:** the first time, Windows asks whether to allow Python on networks. Tick **Private**
(and Public if the hotspot is marked public) and click Allow. If you missed the prompt, run this in an admin PowerShell:
```powershell
New-NetFirewallRule -DisplayName "PathSnap API" -Direction Inbound -Protocol TCP -LocalPort 8000 -Action Allow
```

## 2. Laptop 2: UI (friend's laptop)
```powershell
git clone https://github.com/Speedbird-One/ML-Hackathon.git
cd ML-Hackathon
python -m venv .venv
.venv\Scripts\Activate.ps1                 # macOS/Linux: source .venv/bin/activate
pip install -r ui/requirements.txt
```
Test the connection (use Laptop 1's IP):
```powershell
python ui/api_client.py --api http://10.30.200.193:8000
```
You should see `/health` and the list of 10 trips. Then start the UI:
```powershell
$env:PATHSNAP_API_URL = "http://10.30.200.193:8000"
streamlit run ui/app.py
```
The browser opens at `http://localhost:8501`. Pick a trip and click **Reconstruct route**.

Optional: if Laptop 2 also has the evaluator folder in `data/evaluator/`, the UI ticks
"Send GPS track from this laptop" and uploads the GPS points to the API. Otherwise Laptop 1 uses its own copy.

## 3. Screenshots for report Section 7
* **UI:** the Streamlit page showing the trip list, the metrics and the Folium map.
* **API responses on the command line:** run `python ui/api_client.py --api http://<IP>:8000 --trip <id>`
  on Laptop 2. Also screenshot the uvicorn window on Laptop 1, which prints every request (client IP, timings,
  classifier vs HMM metrics).

## 4. Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | model name, features, HMM params, partition |
| GET | `/trips` | the trips to choose from (id, #points, has reference, cached) |
| POST | `/predict` | `{"trip_id": "00000003", "track": [[lon,lat,t],...]?}` → route + metrics + geometry |
| GET | `/predict/{trip_id}` | same, without geometry (quick browser check) |
| GET | `/map/{trip_id}` | interactive Folium map rendered on Laptop 1 |

## 5. Switching the model (e.g. after Bayesian HPO)
Save the tuned model to `artifacts/` and change `model_file` / `model_name` in
`artifacts/inference_config.json`. Keep the `features` list identical to what the model was trained on.
Restart uvicorn. The HMM parameters and `hmm_top_k` (candidates per point fed to Viterbi) are in the same file.

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| UI: "Cannot reach http://…" | Same hotspot? Correct IP? uvicorn started with `--host 0.0.0.0`? Firewall rule? Try `http://<IP>:8000/health` in Laptop 2's browser. |
| API prints "no trips in data/evaluator" | Copy the evaluator trips there, or set `PATHSNAP_DATA`. It falls back to student_test meanwhile. |
| First request is slow | Expected for long trips (road graph + Viterbi). Use `PATHSNAP_PRECOMPUTE=1` and wait for "precompute finished". |
| Map area is blank but lines show | Laptop 2 has no internet for the map tiles. Use the hotspot's data. |
| IP changed | Hotspots hand out new IPs on reconnect. Re-run `ipconfig` and update the sidebar field. |
