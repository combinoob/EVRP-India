# EVRP India — Time-Optimal EV Routing with Charging Awareness

![EVRP_Bengaluru_Mysuru](image.png)

This repository implements a **time-optimal Electric Vehicle Routing Problem (EVRP)** pipeline for Indian cities, integrating:



- Traffic-aware routing (Google Directions API)

- Physics-based EV energy consumption modeling

- Probabilistic charging station availability

- Charging-aware route planning (single charging stop)

- Interactive map visualization with SOC tracking



> ⚠️ Current scope: **single charging stop only**, suitable for short to medium distance trips.



---

## How to Run



### A. Start the Charging Backend (Terminal 1)



```bash

cd evrp_charge_availability_backend

.venv/bin/activate   # or .venvScriptsactivate on Windows

python scripts/serve.py

Backend runs at:

http://###.0.0.1:8000
```


### B. Run EVRP Planner (Terminal 2)

```bash
cd evrp_india

.venv/bin/activate

python -m src.main

```



## 1. What This Project Does



Given:

- an **origin**

- a **destination**

- an **EV configuration**

- a **departure time**



The system computes:



1. The **fastest route** using real-time traffic

2. The **energy consumption** along that route using elevation and speed

3. Whether the trip is **feasible without charging**

4. If not feasible:

&nbsp;  - proactively chooses a **charging station before SOC reaches reserve**

&nbsp;  - minimizes **expected total travel time**, accounting for:

&nbsp;    - driving time

&nbsp;    - expected waiting time at charger

&nbsp;    - charging time

5. Outputs:

&nbsp;  - total travel time

&nbsp;  - total energy

&nbsp;  - chosen charging stop (if any)

&nbsp;  - interactive map with route, chargers, and SOC hover



---



## 2. Architecture Overview



This project (`evrp_india`) works together with a **separate backend**:



### A. `evrp_india` (this repo)

Responsible for:

- Routing (Google Directions)

- Energy modeling

- Planning logic

- Visualization



### B. Charging Availability Backend

A separate service: https://github.com/suharoy/evrp_charge_availability_backend





It provides:

- `/stations/nearby`

- `/recommend` → probabilistic charger availability (`p_success`)



This backend **must be running** for charging-aware planning.



---



## 3. Project Structure



evrp_india/

├── src/

│ ├── main.py # Entry point

│ ├── config.py # All configuration constants

│ ├── planner/

│ │ └── min_time_with_charging.py

│ ├── routing/

│ │ ├── google_api.py # Directions + elevation

│ │ └── google_energy_profile.py

│ ├── charging/

│ │ └── availability_client.py

│ ├── models/

│ │ └── vehicle.py

│ └── viz/

│ └── map_viz.py # Folium visualization

├── data/ # (empty except cache, ignored)

├── README.md

└── .gitignore



---



## 4. How the Planning Logic Works



### Step 1 — Fastest Route

Uses **Google Directions API** with traffic:

origin → destination



Returns geometry, duration, and per-step info.



---



### Step 2 — Energy Estimation

For each step:

- distance / duration → speed

- elevation difference → slope

- applies EV physics model:

&nbsp; - rolling resistance

&nbsp; - aerodynamic drag

&nbsp; - gravitational component

&nbsp; - drivetrain efficiency



Produces:

- total energy (kWh)

- step-wise energy consumption



---



### Step 3 — Direct Feasibility Check

Let:

- usable energy = `battery_kwh × SOC_START`

- reserve = `battery_kwh × SOC_MIN`



If:

direct_energy ≤ usable − reserve

→ no charging needed.



---



### Step 4 — SOC Critical Point

If not feasible:

- simulate SOC along the route

- find the first point where SOC would hit reserve



This is the **danger point**.



---



### Step 5 — Proactive Charging Search

Instead of waiting until reserve:

- look **backwards** from the critical point

- sample points along the route

- query `/recommend(lat, lon, time)` for chargers



This allows **charging before SOC becomes critical**.



---



### Step 6 — Charger Evaluation (Single Stop)

For each candidate charger `C`:

- route `origin → C`

- route `C → destination`

- compute:

&nbsp; - energy for both legs

&nbsp; - charging time

&nbsp; - expected waiting time using `p_success`


Objective:

minimize expected total time



The best charger is selected.



---

### 5. Extension to Multiple Stations

- We divide the whole long route into parts.
  
- The prigin is now set to Charging Station 1 for the first part.
  
- Starting SOC is set to 100.
  
- This recursion solves the routing problem for long distances, although latency is high.

### 6. Visualization



The output map (`routes.html`) shows:



- Fastest route (purple polyline)

- All charging stations along the route corridor (⚡ icons)

- Chosen charging stop(s) (green ⚡)

- Dense SOC dots along the route

&nbsp; - hover shows SOC percentage at that location



No animation or legends are used — the map is kept minimal and readable.



---


