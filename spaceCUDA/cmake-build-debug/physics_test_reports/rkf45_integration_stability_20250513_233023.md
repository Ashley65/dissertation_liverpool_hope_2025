# Physics Engine Test Report
Generated: 2025-05-13 23:30:23

## System Configuration
- Number of bodies: 9
- Number of ships: 0

## Initial Bodies State
| Index | Body | Position (x,y,z) | Velocity (vx,vy,vz) | Mass |
|-------|------|-----------------|-------------------|------|
| 0 | Body 0 | (0, 0, 0)  | (0, 0, 0)  | 1.989e+30 |
| 1 | Body 1 | (5.791e+10, 0, 0)  | (0, 47362, 0)  | 3.3011e+23 |
| 2 | Body 2 | (1.0821e+11, 0, 0)  | (0, 35022, 0)  | 4.8675e+24 |
| 3 | Body 3 | (1.496e+11, 0, 0)  | (0, 29783, 0)  | 5.972e+24 |
| 4 | Body 4 | (2.2794e+11, 0, 0)  | (0, 24077, 0)  | 6.4171e+23 |
| 5 | Body 5 | (7.7857e+11, 0, 0)  | (0, 13072, 0)  | 1.8982e+27 |
| 6 | Body 6 | (1.4335e+12, 0, 0)  | (0, 9652.8, 0)  | 5.6834e+26 |
| 7 | Body 7 | (2.8725e+12, 0, 0)  | (0, 6835.2, 0)  | 8.681e+25 |
| 8 | Body 8 | (4.4951e+12, 0, 0)  | (0, 5477.8, 0)  | 1.0243e+26 |

## RKF45 Integration Stability Test

### Initial Conditions
Initial Earth-Sun distance: 1.496e+11 m
Initial Earth-Sun orbital energy: -2.65076e+33 J

### Simulation Configuration
- Barnes-Hut algorithm: Disabled
- Time step: 8640 s (1/10 day)
- Simulation duration: 10 days
- Total steps: 100

### Simulation Progress
- Step 0 (Day 0): Earth-Sun distance = 1.496e+11 m
- Step 20 (Day 2): Earth-Sun distance = 1.496e+11 m
- Step 40 (Day 4): Earth-Sun distance = 1.496e+11 m
- Step 60 (Day 6): Earth-Sun distance = 1.496e+11 m
- Step 80 (Day 8): Earth-Sun distance = 1.496e+11 m

Simulation completed in 9 ms

### Results
Final Earth-Sun distance: 1.496e+11 m
Distance error ratio: 0%
Final Earth-Sun orbital energy: -2.65076e+33 J
Energy error ratio: 0%

### Test Results
- Distance Preservation Test: PASSED (threshold: 5%)
- Energy Conservation Test: PASSED (threshold: 5%)
