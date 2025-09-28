# Physics Engine Test Report
Generated: 2025-05-13 23:36:01

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

## Newtonian Gravity Accuracy Test

Initial Earth position: (1.496e+11, 0)
Initial Sun-Earth orbital energy: -2.65076e+33 J

Barnes-Hut algorithm: Disabled
Earth orbital period: 3.15542e+07 s
Simulation time step: 86449.8 s
Simulation steps: 365

### Simulation Progress
- Step 0: Earth position = (1.496e+11, 0, 0)
- Step 73: Earth position = (1.496e+11, 0, 0)
- Step 146: Earth position = (1.496e+11, 0, 0)
- Step 219: Earth position = (1.496e+11, 0, 0)
- Step 292: Earth position = (1.496e+11, 0, 0)

Simulation completed in 21 ms

### Results
Final Earth position: (1.496e+11, 0)
Position error: 0 m (0% of orbital radius)
Final Sun-Earth orbital energy: -2.65076e+33 J
Energy error: 0%

### Test Results
- Position Test: PASSED (threshold: 2% of orbital radius)
- Energy Conservation Test: PASSED (threshold: 0.1%)
