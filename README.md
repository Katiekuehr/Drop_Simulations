# Drop rebound at low Weber number
This repository provides a python version of the MATLAB scripts from [this repository](https://github.com/harrislab-brown/LowWeberDropRebound?tab=readme-ov-file) for simulating the dynamics of liquid droplets impacting solid substrates. The code systematically explores droplet deformation, spreading, rebound behaviors, and other key physical phenomena, capturing detailed dynamics through numerical simulations.

## Overview 
The project models the complex behavior of liquid droplets when they impact solid surfaces, capturing phenomena like maximum spreading, bounce optimization, energy transfer, and shape deformation using a numerical approach based on spherical harmonics and Newton-Raphson methods. It includes simulation sweeps for parameter exploration such as Weber number (`We`), Bond number (`Bo`), and Ohnesorge number (`Oh`).

# Python Code
Simulations can be run in bulk sweeps or individually. 

## For Simulation Sweeps
To perform simulation sweeps, use the [drop_simulations_sweep.py](https://github.com/Katiekuehr/Drop_Simulations/blob/main/drop_simulations_sweep.py) script. 

The function `plot_and_save()` can be called with either **dimensional parameters** (`rho_in_CGS`= fluid density, `sigma_in_CGS` = surface tension, `nu_in_GCS` = kinematic viscosity, `V_in_CGS` = initial velocity) or **non-dimensional parameters** (`Bond`, `Web`, `Ohn`) along with `R_in_CGS`.

- `plot_and_save()` runs a single simulation for the given input parameters and saves the resulting data and plots in the following directory structure:
  - **Main directory:** `desktop_path = 'Mode_Convergence_R={R_in_CGS}_modes={n_thetas}_Web={We}'`
  - **Subdirectory:** `folder_name = 'simulation_We={We}_Oh={Oh}_Bo={Bo}_modes={n_thetas}'`
  
Both `desktop_path` and `folder_name` are automatically created in this structure, with the `folder_name` placed inside `desktop_path`.

**Customizing Folder Locations:**
To save results to a different location or rename the folder, modify the `desktop_path` and `folder_name` variables to point to the desired folder or path.

**Sweeping Across Parameters:**
- When performing a sweep over the Weber number (`We`): Update the `desktop_path` so that it is **independent of `We`**. This ensures that all results from the Weber-sweep are stored in a single folder.
- To sweep over a range of values for `We`, `Bo`, and `Oh`: Call `plot_and_save()` for each value in the sweep, adjusting the parameters accordingly.


## Running Single Simulations

For individual simulations, use the [drop_simulations_singular.ipynb](https://github.com/Katiekuehr/Drop_Simulations/blob/main/drop_simulations_singular.ipynb). The simulations will be run in a Jupyter Notebook. Plots and videos will not be automatically saved but require manual downloading. 

In the "user input" section, adjust the physical constants as desired:
- **$\rho$** = fluid density
- **$\sigma$** = surface tension
- **$g$** = gravitational acceleration
- **$\nu$** = kinematic viscosity
- **$R$** = radius
- **$V$** = initial velocity
- **$T_{end}$** = non-dimensional time of simulation end
- **$n_{thetas}$** = number of Legendre Polynomials used for approximation
- **`n_sampling_time_L_mode`**: limits the size of the dimensionless time-step. The larger `n_sampling_time_L_mode`, the smaller the timestep.

To run video animations in `drop_simulation()` and `pressure_simulation()`, it is necessary to have **FFmpeg** installed. See below for a guide on how to install FFmpeg.


### Installing FFmpeg

This project requires **FFmpeg** to save or render movie simulations.  
Please install FFmpeg according to your operating system:

---

**Windows**
1. Download FFmpeg from the [official website](https://ffmpeg.org/download.html).
   - Under "Windows", choose a build from sites like Gyan.dev or BtbN.
2. Extract the downloaded `.zip` file.
3. Add the `bin` folder (inside the extracted folder) to your **System PATH**:
   - Search for "Environment Variables" → Edit the **PATH** variable → Add the path to the `bin` folder.
4. Open a new Command Prompt and verify the installation:
   ```bash
   ffmpeg -version
   ```

**macOS**

If you have Homebrew installed, run:
```bash
   brew install ffmpeg
```
Then verify: 
```bash
ffmpeg -version
```
If you don't have Homebrew installed, you can install it first by running:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Linux (Ubuntu/Debian)**

Update your package list and install FFmpeg:
```bash
sudo apt update
sudo apt install ffmpeg
```
Then verify:
```bash
ffmpeg -version
```


