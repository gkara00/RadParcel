# RadParcel: A Dashboard for Radionuclides Dispersion Simulations
**Version:** v0.0-beta

**RadParcel** is an interactive dashboard designed to track the dispersion of radionuclides in the ocean. Built with the power of the [**OceanParcels**](https://oceanparcels.org/) Langrangian framework, Python, and Plotly Dash, this project enables visualization and analysis of radionuclides trajectories and activity decay influenced by ocean currents.

---

## Features

- **Radionuclide Dispersion Simulation**: Simulate the movement and decay of radionuclides in realistic ocean currents.
- **User-Friendly Interface**: Users can specify parameters such as release location, simulation duration, activity levels, and radionuclide half-life.
- **Realistic Ocean Data**: Utilizes Copernicus Marine Service (CMEMS) datasets for ocean currents.
- **Dynamic Visualizations**: Interactive and animated trajectory maps powered by Plotly.
- **Export Results**: Saves particle trajectory data to Zarr format for post-processing and analysis.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/gkara00/RadParcel.git
   cd RadParcel
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   See also [here](https://docs.oceanparcels.org/en/latest/installation.html)

3. Run the application:
   ```bash
   python app.py
   ```

4. Open the app in your browser:
   ```
   http://127.0.0.1:8050/

---

## Dependencies

- **Python 3.7+**
- [Parcels](https://oceanparcels.org/)
- Dash and Plotly  
- NumPy  
- Xarray 
- Zarr
- Netcdf4 

---

## **How to Use**  

1. **Set Parameters**:
   - Input release longitude and latitude.
   - Select simulation duration and radionuclide properties.

2. **Run Simulation**:
   - Click **Simulate** to model radionuclide dispersion.

3. **Visualize Results**:
   - View interactive maps with particle trajectories and activity decay.

---

## **Hackathon Context**

This project was developed as part of [RAMONES Hackathon](https://sites.google.com/view/ramones-hackathon-2024/home) held on December 4-5 at the Physics Department of National and Kapodistrian University of Athens (NKUA). Our goal is to provide a scalable solution for tracking and visualizing radionuclide dispersion in oceans, aiding in environmental monitoring and risk assessment.

---

## **Contributors**

Team **Radiohead** (Ocean Physics and Modelling Group | NKUA, Athens):

- **John Karagiorgos** ([gkara00](https://github.com/gkara00))
- **Roushit Dallenga** ([rousit55](https://github.com/rousit55))

---

## Acknowledgments

- [OceanParcels](https://oceanparcels.org/) particle tracking framework
- [Copernicus Marine Service](https://marine.copernicus.eu/) for ocean current datasets
- [Dash](https://dash.plotly.com/) for interactive visualization
- [RAMONES](https://ramones-project.eu/) EU H2020 FET project

---

## **Future Enhancements**

- TODO  

---
