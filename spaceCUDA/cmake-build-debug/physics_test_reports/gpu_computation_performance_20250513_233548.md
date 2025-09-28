# Physics Engine Test Report
Generated: 2025-05-13 23:35:48

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

## GPU Computation Performance Test

### Test Configuration
- System sizes: 100, 500, 1000 bodies
- Time step: 86400.0 s (1 day)
- Iterations per test: 3

### Performance Results
| Bodies | Newtonian (μs) | Barnes-Hut (μs) | Speedup Factor |
|--------|---------------|-----------------|----------------|
#### Creating system with 100 bodies
- Newtonian gravity completed in 31.3333 μs
- Barnes-Hut gravity completed in 510.667 μs
- Speedup factor: 0.0613577

| 100 | 31.3333 | 510.667 | 0.0613577 |
#### Creating system with 500 bodies
- Newtonian gravity completed in 33 μs
- Barnes-Hut gravity completed in 5627 μs
- Speedup factor: 0.00586458

| 500 | 33 | 5627 | 0.00586458 |
#### Creating system with 1000 bodies
- Newtonian gravity completed in 30.6667 μs
- Barnes-Hut gravity completed in 1655.33 μs
- Speedup factor: 0.018526

| 1000 | 30.6667 | 1655.33 | 0.018526 |

### Algorithmic Complexity Analysis
- Newtonian gravity: O(n²) complexity
- Barnes-Hut: O(n log n) complexity

Expected Newtonian scaling factor (n²): 100
Actual Newtonian scaling factor: 0.978723
Expected Barnes-Hut scaling factor (n log n): 15
Actual Barnes-Hut scaling factor: 3.24151

### Performance Summary
Barnes-Hut algorithm is currently not outperforming the direct Newtonian calculation. This suggests the implementation may need optimization, or the system sizes tested are below the crossover point.
