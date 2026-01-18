# Taylor-Green Vortex (TGV) Workflow

This directory contains the scripts and output files for the 3D Taylor-Green Vortex simulation using `spectralDNS`.

## 1. Run the Simulation
Run the solver using the `TG.py` script. This will perform the computations and save the flow data to `NS9_w.h5`.

```bash
# Activate the environment
conda activate spectralDNS

# Run the simulation (adjust parameters as needed)
# --nu: Viscosity (Re = 1/nu)
# --M: Mesh size (2^M in each direction, e.g., 6 -> 64^3)
# --T: End time
# --write_result: Save data every N steps (Crucial for visualization)
mpirun -np 4 python TG.py --nu 0.0025 --M 6 6 6 --T 1.0 --write_result 10 --plot_step 0 NS
```

**Note:** You might see a `Regression check failed` warning at the end. This is expected when you run with custom parameters (like a different `nu`) and can be ignored.

## 2. Generate Visualization Wrapper
The simulation outputs raw HDF5 data (`NS9_w.h5`). To view this in ParaView or VisIt, you need an XDMF wrapper file that describes the grid and time steps.

Run the provided helper script:

```bash
python create_xdmf.py
```

This will generate a file named **`NS9_w.xdmf`**.

## 3. Visualize the Data
Open the generated `.xdmf` file in your visualization tool.

**Using ParaView:**
```bash
paraview NS9_w.xdmf
```
*   When prompted for a reader, select **XDMF Reader**.
*   Click **Apply**.
*   Use the **Play** button to watch the simulation.

**Using VisIt:**
```bash
visit -o NS9_w.xdmf
```
*   Go to **Plots -> Add -> Pseudocolor -> U_X** (or U_Y, U_Z, P, etc.).
*   Click **Draw**.
*   **To see vectors:** Create a Vector expression `{U_X, U_Y, U_Z}`.

## File Descriptions
*   **`TG.py`**: The main simulation script (copy of `demo/TG.py`).
*   **`create_xdmf.py`**: Helper script to generate the correct XDMF wrapper for visualization.
*   **`NS9_w.h5`**: The heavy data file containing velocity, pressure, and curl fields.
*   **`NS9_w.xdmf`**: The "map" file used by ParaView/VisIt to read the .h5 file.
*   **`NS9_c.h5`**: Checkpoint/Mesh file (contains grid coordinates).
*   **`diagnostics.csv`**: CSV file containing time history of Cycle, Kinetic Energy, Enstrophy, and Divergence.