# Ants

A collection of notebooks and scripts for analyzing ant-driven pendulum experiments, including video processing, coordinate tracking, simulation, and data analysis.

## Authors

| [![Partner](https://img.shields.io/badge/Avshalom_Mezer_Keydar-black?style=for-the-badge&logo=github)](https://github.com/notagoodprogramer) | [![Nitzan](https://img.shields.io/badge/Nitzan_Bar_Eitan-black?style=for-the-badge&logo=github)](https://github.com/NitzanBarEitan) |
| :---: | :---: |

## Repository Structure

```
Ants/
├── .idea/                           # IDE project settings
├── .gitignore                       # Files to ignore in version control
├── data_analysis.ipynb              # Jupyter notebook for data analysis
├── object_tracking_coordinates.csv  # CSV of extracted pendulum coordinates
├── pendulum_simulation.ipynb        # Jupyter notebook for pendulum simulation
├── pendulum_vid_track_and_graph.ipynb  # Notebook for video tracking & plotting
├── pendulumtracking.py              # Python script for video-based tracking
└── video_processing.ipynb           # Notebook for preprocessing videos
```

## Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/NitzanBarEitan/Ants.git
   cd Ants
   ```
2. **Create a Python environment** (optional but recommended)

   ```bash
   python3 -m venv env
   source env/bin/activate   # On Windows: env\\Scripts\\activate
   ```
3. **Install dependencies**

   ```bash
   pip install jupyter numpy pandas matplotlib opencv-python scipy
   ```

## Usage

### Launch Jupyter Notebooks

Start JupyterLab or Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Open any `.ipynb` file to explore.

## Notebooks Overview

* dataanalysis.ipynb: Load and analyze extracted coordinates (created with video\_processing.py; plots the relevant graphs and performs statistical analysis.
* **video\_processing.ipynb**: Tracks the ant-driven pendulums and outputs a csv file with the coordinates of each pendulum.
* **pendulum\_vid\_track\_and\_graph.ipynb**: Tracks the physical system (physically coupled pendulums) and plots the relevant graphs.
* **pendulum\_simulation.ipynb**: All the simulations used for the project.
