# EDA report

## Scope

The EDA covers only the labelled `train` partition. It does not read `student_test` or `evaluator` trips. This keeps evaluation data out of development work.

The notebook checks the expected PathSnap trip structure:

- GPS track: longitude, latitude, timestamp
- Road-network nodes and directed arcs
- Ground-truth route arc IDs

It measures trip length, elapsed time, sampling interval, GPS movement, route size, network size, invalid route references, and route connectivity. It also creates distribution plots and a map for one representative training trip.

## How to run

1. Copy the organiser-provided `train/` directory into `data/`, or set `PATHSNAP_DATA` to its parent directory.
2. Create the project environment with the commands in `SETUP.md`.
3. Open and run `notebooks/01_eda.ipynb` from top to bottom.

The notebook creates `results/eda/` locally. Review its CSV summaries and figures before deciding which generated outputs to keep in Git.

## Current status

The notebook ran on all 80 labelled training trips. It found 208,417 GPS points in total. The median trip has 1,529 GPS points, lasts 1,710 seconds, and has a GPS path length of 26.8 km. The median road network has 9,608 nodes and 19,526.5 directed arcs.

Every trip loaded successfully. All route arc IDs are valid and timestamps strictly increase. The median interval is 1 second, but 60 trips contain 7,273 gaps longer than 1.5 seconds. The longest gap is 153,064 seconds. Trip `00000057` does not project into the configured Darmstadt UTM CRS and needs coordinate validation before it is used for distance-based features. There are 13 disconnected transitions across the labelled routes of 10 trips. Inspect these trips before using strict route-continuity assumptions in training or decoding.

The generated outputs are:

- `trip_summary.csv`: one row per training trip
- `dataset_summary.csv`: dataset-level metrics
- `data_quality.csv`: loading and route-validation checks
- `trip_distributions.png`: distributions across trips
- `representative_trip.png`: GPS, road network, and labelled route

## What to review before modelling

Use the generated summaries to decide the next modelling steps:

- Keep the current data-load and route-ID checks in the feature pipeline.
- Validate or exclude trip `00000057` before using projected distance features.
- Split trajectories at large timestamp gaps before calculating distance, speed, candidate transitions, or HMM transitions. The 95th-percentile within-trip speed reaches 53.7 m/s, so use robust distance and speed features.
- Compare trip and network sizes. Large networks or long tracks affect candidate-search and Viterbi runtime.
- Inspect the 10 trips with disconnected route transitions. They may indicate data issues or require careful transition handling.
- Review the representative map before building candidate and emission features.

## Modelling prerequisites

Before trying classifiers or the HMM/Viterbi decoder, the team still needs to define and implement:

1. Candidate-road-arc generation for every GPS point.
2. Labels that mark whether each candidate is on the ground-truth route at that point.
3. Features such as point-to-arc distance, heading agreement, GPS speed, and transition distance.
4. Trip-level train/validation splits to prevent leakage between points from the same trip.
5. Baseline metrics for candidate recall, classifier quality, route accuracy, and decoding runtime.

The EDA outputs are inputs to those decisions, not model results.
