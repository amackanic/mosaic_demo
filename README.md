# mosaic_demo

# Catheter Navigation Simulation Demo

This project simulates a catheter navigating through a 3D vessel using a control algorithm based on simulated IVUS (intravascular ultrasound) and impedance signals. The simulation visualizes the vessel, catheter, IVUS cross-section, and impedance signal in real time.

## Features
- 3D vessel with variable radius, branch, and optional plaque/stenosis
- Catheter automatically centers itself using simulated IVUS and impedance signals
- Realistic IVUS image and forward-looking impedance plot
- Visualization of vessel, catheter, IVUS, and impedance in a single window
- Support for vessel branches, stenosis, and wall thickening (plaque)
- Interactive 3D view (via matplotlib)

## Installation
1. **Clone the repository or copy the files to your local machine.**
2. **Create a Python virtual environment (optional but recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install the required packages:**
   ```bash
   pip install numpy matplotlib scipy
   ```

## Usage
Run the main demo script:
```bash
python demo_step1.py
```

This will open a window with three panels:
- **Left:** 3D vessel and catheter
- **Middle:** Simulated IVUS cross-section
- **Right:** Forward-looking impedance plot

The catheter will automatically advance through the vessel, using the simulated signals to stay centered and avoid plaque/stenosis.

## Customization
- You can adjust vessel geometry, plaque, and control parameters at the top of `demo_step1.py`.
- To simulate different vessel shapes, modify the centerline and radius arrays.
- To add more realism (e.g., inertia, noise, eccentric plaque), see the comments in the code.

## Requirements
- Python 3.7+
- numpy
- matplotlib
- scipy

## License
This project is for demonstration and research purposes only.
