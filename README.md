## Setup and Usage

### 1. Data Acquisition
Download the **MOLA Mars Map** from the link below and place the files in the root directory of the project:
* [MOLA Mars Map Download](https://drive.google.com/drive/folders/1GkDenVWffuh1Tn4eem-cdBP_hcvb8O1M?usp=sharing)

### 2. Running Simulations
You can run the following scripts depending on the type of test required:

* **Random Map Test (2D):**
  Run this script for a mission test on a randomly generated 2D map.
  ```bash
  python test_ppo_visual.py
  ```

* **Actual Mars Map Test (2D & 3D):**
  Run this script to use the actual MOLA Mars map data for both 2D and 3D mission tests.
  ```bash
  python visual_mission_3d.py
  ```
