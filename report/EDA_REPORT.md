# EDA report

## Scope

EDA only examines the labelled train subset. It will not process student_test or evaluator drives. Thus evaluation data will not be used during the development.

The notebook verifies the expected structure of PathSnap trip:

- GPS trajectory: longitude, latitude, timestamp
- Graph of road network nodes and edges
- IDs of ground truth route edges

Trip duration, time passed, sampling interval, GPS displacement, route size, network size, invalid route references, and route connectedness are evaluated. Distribution plots and a map of the example training drive are created.

## How to run

1. Place the `train/` directory provided by organisers in `data/`, or set the `PATHSNAP_DATA` variable to the parent directory.
2. Create the project environment by running the commands in `SETUP.md`.
3. Execute `notebooks/01_eda.ipynb` notebook from top to bottom.

The notebook will create the `results/eda/` directory locally. Examine its CSV statistics and visualisations to decide what generated files to commit.

## Status Quo

The notebook was run on all 80 labelled training trips. It identified 208,417 GPS points in total. The median trip contains 1,529 GPS points, takes 1,710 seconds, and has a GPS path length of 24.0 km. The median road network consists of 9,608 nodes and 19,526.5 directed arcs.

All trips were loaded successfully. Route arc IDs are correct, and the timestamp values strictly increase. The median interval value is 1 second; however, 60 trips include 7,273 intervals larger than 1.5 seconds. The maximum interval length is 153,064 seconds. The first run of this notebook was made under the assumption of one Darmstadt UTM CRS; it resulted in an error for trip `00000057` (Ecuador), and distorted lengths for numerous trips outside Europe. As this dataset includes cities from around the world, `Trip.track_xy()` now projects each trip to its own local metric system. This report reflects the rerun made with this projection applied; all projected coordinates are finite. There are 13 disconnected transitions for the labelled routes of 10 trips. Check those trips before making any route-continuity assumptions during training or decoding.

The outputs generated are:

- `trip_summary.csv`: one row per training trip
- `dataset_summary.csv`: dataset-level metrics
- `data_quality.csv`: loading and route-validation checks
- `trip_distributions.png`: distributions across trips
- `representative_trip.png`: GPS, road network, and labelled route

## Pre-model review items

Based on the generated summary, plan further modelling as follows:

- The existing data-load and route-ID filtering must be kept in the feature pipeline.
- Distances and speeds follow per-trip projection. The validity check for trip `00000057` is true.
- For distance, speed, candidate transition, and HMM transition calculation, split the trajectories when large gaps exist in the timestamps. Since 95% percentile speed in the trip is 45.3 m/s, robust distances and speeds should be used.
- Trip and network size comparison is needed since large networks/tracks influence the runtime of candidate search and Viterbi.
- Examine the 10 trips where the route transitions are disconnected. They may need special treatment either due to data or transition issues.
- Review the representative map before creating candidate and emission features.

## Prerequisites for modelling

Prior to exploring classification models and HMM/Viterbi decoding, the team will need to specify and code:

1. Generation of candidate road arcs for every GPS point.
2. Labelling of candidates whether they belong to the true route at this point.
3. Features like point-to-arc distance, heading similarity, GPS speed, and transition distance.
4. Splitting of trips into training and validation to avoid leakage between points of the same trip.
5. Metrics for candidate recall, classifier performance, route accuracy, and decoding time.

EDA output will be used in these decisions, not vice versa.
