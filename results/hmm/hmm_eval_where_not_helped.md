# Where the HMM did not help (student-test)

## HMM, default parameters
- top-1 change per trip (points): min +0.90, median +4.03, max +8.21
- trips where the HMM was worse than classifier-only: 0 of 10
- slow points (<1 m/s): classifier-only 0.461 -> HMM 0.591 (575 points)
- continuity: classifier-only 0.924 -> HMM 0.978

## HMM, tuned parameters
- top-1 change per trip (points): min +2.10, median +5.23, max +8.79
- trips where the HMM was worse than classifier-only: 0 of 10
- slow points (<1 m/s): classifier-only 0.461 -> HMM 0.668 (575 points)
- continuity: classifier-only 0.924 -> HMM 0.977
