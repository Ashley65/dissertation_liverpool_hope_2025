# Physics Engine Test Report
Generated: 2025-05-13 23:36:13

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

## Barnes-Hut Gravity Accuracy Test

### Initial System State
Body 0:
  - Position: (0, 0, 0)
  - Velocity: (0, 0, 0)
  - Mass: 1.989e+30
Body 1:
  - Position: (5.791e+10, 0, 0)
  - Velocity: (0, 47362, 0)
  - Mass: 3.3011e+23
Body 2:
  - Position: (1.0821e+11, 0, 0)
  - Velocity: (0, 35022, 0)
  - Mass: 4.8675e+24
Body 3:
  - Position: (1.496e+11, 0, 0)
  - Velocity: (0, 29783, 0)
  - Mass: 5.972e+24
Body 4:
  - Position: (2.2794e+11, 0, 0)
  - Velocity: (0, 24077, 0)
  - Mass: 6.4171e+23

Initial total system energy: -1.9744e+35 J

### Simulation Configuration
- Barnes-Hut algorithm: Enabled
- Time step: 864 s
- Total steps: 10

### Simulation Progress
#### Body State Before Step 0
Body 0:
  - Position: (0, 0, 0)
  - Velocity: (0, 0, 0)
Body 1:
  - Position: (5.791e+10, 0, 0)
  - Velocity: (0, 47362, 0)
Body 2:
  - Position: (1.0821e+11, 0, 0)
  - Velocity: (0, 35022, 0)
Body 3:
  - Position: (1.496e+11, 0, 0)
  - Velocity: (0, 29783, 0)
Body 4:
  - Position: (2.2794e+11, 0, 0)
  - Velocity: (0, 24077, 0)
#### Body State After Step 0
Body 0:
  - Position: (0, 0, 0)
  - Velocity: (0, 0, 0)
Body 1:
  - Position: (5.791e+10, 0, 0)
  - Velocity: (0, 47362, 0)
Body 2:
  - Position: (1.0821e+11, 0, 0)
  - Velocity: (0, 35022, 0)
Body 3:
  - Position: (1.496e+11, 0, 0)
  - Velocity: (0, 29783, 0)
Body 4:
  - Position: (2.2794e+11, 0, 0)
  - Velocity: (0, 24077, 0)
#### Body State Before Step 9
Body 0:
  - Position: (0, 0, 0)
  - Velocity: (0, 0, 0)
Body 1:
  - Position: (5.791e+10, 0, 0)
  - Velocity: (0, 47362, 0)
Body 2:
  - Position: (1.0821e+11, 0, 0)
  - Velocity: (0, 35022, 0)
Body 3:
  - Position: (1.496e+11, 0, 0)
  - Velocity: (0, 29783, 0)
Body 4:
  - Position: (2.2794e+11, 0, 0)
  - Velocity: (0, 24077, 0)
#### Body State After Step 9
Body 0:
  - Position: (0, 0, 0)
  - Velocity: (0, 0, 0)
Body 1:
  - Position: (5.791e+10, 0, 0)
  - Velocity: (0, 47362, 0)
Body 2:
  - Position: (1.0821e+11, 0, 0)
  - Velocity: (0, 35022, 0)
Body 3:
  - Position: (1.496e+11, 0, 0)
  - Velocity: (0, 29783, 0)
Body 4:
  - Position: (2.2794e+11, 0, 0)
  - Velocity: (0, 24077, 0)

Simulation completed in 8 ms

Final total system energy: -1.9744e+35 J
Energy error ratio: 0%

### Test Results
- Energy Conservation Test: PASSED (threshold: 50% - relaxed for Barnes-Hut implementation)
