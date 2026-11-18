| Model              |   CV macro F1 |   CV top-1 |   CV log loss |   Student-test top-1 |   Student-test macro F1 |   Student-test log loss |   Inference ms/pt |   Final fit s |
|:-------------------|--------------:|-----------:|--------------:|---------------------:|------------------------:|------------------------:|------------------:|--------------:|
| LightGBM (untuned) |       0.92078 |    0.86911 |       0.08745 |              0.87644 |                 0.92656 |                 0.07650 |           0.04078 |      11.08319 |
| LightGBM (tuned)   |       0.92232 |    0.87033 |       0.08501 |              0.87497 |                 0.92603 |                 0.07615 |           0.02445 |       6.93818 |

Search cost: 30 trials (11 pruned) in 16.3 min (objective: mean CV log_loss, minimize); best trial #15.
