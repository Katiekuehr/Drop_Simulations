# Drop rebound at low Weber number
This repository provides a python version of the MATLAB scripts from [this repository](https://github.com/harrislab-brown/LowWeberDropRebound?tab=readme-ov-file) for simulating the dynamics of liquid droplets impacting solid substrates. The code systematically explores droplet deformation, spreading, rebound behaviors, and other key physical phenomena, capturing detailed dynamics through numerical simulations.

## Overview 
The project models the complex behavior of liquid droplets when they impact solid surfaces, capturing phenomena like maximum spreading, bounce optimization, energy transfer, and shape deformation using a numerical approach based on spherical harmonics and Newton-Raphson methods. It includes simulation sweeps for parameter exploration such as Weber number (`We`), Bond number (`Bo`), and Ohnesorge number (`Oh`).

## Python Code
### Running Simulations
You can run simulations in bulk or individually. For individual simulations, use [drop_simulations_singular.ipynb](https://github.com/Katiekuehr/Drop_Simulations/blob/main/drop_simulations_singular.ipynb). In "user input" adjust physical constants as desired. $\rho$ = density, $\sigma$ = surface tesnion, $g$ = gravitational acceleration, $nu$ = kinematic viscosity; $R$ = radius, $V$ = initial velocity, $T_{end}$ = non-dimensional time of simulation end, $n_{thetas}$ = number of Legendre Polynomials used for approximation, n\_sampling\_time\_L\_mode: limits size of dimensionless time-step - the larger n\_sampling\_time\_L\_mode, the smaller the timestep.

To run video animations in `drop_simulation()` and `pressure_simulation()`, it is necessary to gave installed FFmpeg. See below for a guide on how to install FFmpeg. Video simulations will not be saved automatically but require manual downloading for saving.



## Installing FFmpeg

This project requires **FFmpeg** to save or render movie simulations.  
Please install FFmpeg according to your operating system:

---

### Windows
1. Download FFmpeg from the [official website](https://ffmpeg.org/download.html).
   - Under "Windows", choose a build from sites like Gyan.dev or BtbN.
2. Extract the downloaded `.zip` file.
3. Add the `bin` folder (inside the extracted folder) to your **System PATH**:
   - Search for "Environment Variables" → Edit the **PATH** variable → Add the path to the `bin` folder.
4. Open a new Command Prompt and verify the installation:
   ```bash
   ffmpeg -version
   ```

### macO
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

### Linux (Ubuntu/Debian)
Update your package list and install FFmpeg:
```bash
sudo apt update
sudo apt install ffmpeg
```
Then verify:
```bash
ffmpeg -version
```


