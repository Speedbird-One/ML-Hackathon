| Partition                           | Decoder         |   Top-1 |   Top-1 moving |   Top-1 slow (<1 m/s) |   Route continuity |   Route recall |   Route precision |   Route F1 |
|:------------------------------------|:----------------|--------:|---------------:|----------------------:|-------------------:|---------------:|------------------:|-----------:|
| train out-of-fold (used for tuning) | classifier only |  0.8715 |         0.8895 |                0.4702 |             0.8974 |         0.8901 |            0.8417 |     0.8589 |
| train out-of-fold (used for tuning) | HMM default     |  0.9018 |         0.9149 |                0.6197 |             0.9562 |         0.9027 |            0.9372 |     0.9162 |
| train out-of-fold (used for tuning) | HMM tuned       |  0.9324 |         0.9412 |                0.7591 |             0.9543 |         0.9091 |            0.9393 |     0.9212 |
| student-test (check only)           | classifier only |  0.8746 |         0.8878 |                0.4609 |             0.9239 |         0.8892 |            0.8608 |     0.8720 |
| student-test (check only)           | HMM default     |  0.9116 |         0.9220 |                0.5913 |             0.9779 |         0.9023 |            0.9552 |     0.9260 |
| student-test (check only)           | HMM tuned       |  0.9257 |         0.9341 |                0.6678 |             0.9767 |         0.9116 |            0.9616 |     0.9344 |

Search: 29 trials in 10.5 min on 8 worker(s); objective = train out-of-fold top-1 after HMM; default parameters were trial 0 (top-1 0.9018); best trial #22 (top-1 0.9324).
