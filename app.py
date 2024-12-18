import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np
import xarray as xr
from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, Variable
from datetime import timedelta, datetime
from operator import attrgetter
import os
import sys

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "RadParcel"

# Dash Layout
app.layout = html.Div(
    [
        html.H1("RadParcel v0.0-beta | A Dashboard for Radionuclides Dispersion Simulations", style={"textAlign": "center"}),
        html.Div(
            [
                html.Label("Select Release Location:"),
                dcc.Input(id="input-lon", type="number", placeholder="Longitude", step=0.0001, style={"marginRight": "10px"}),
                dcc.Input(id="input-lat", type="number", placeholder="Latitude", step=0.0001, style={"marginRight": "10px"}),
                html.Label("Select Start Date:"),
                dcc.DatePickerSingle(
                    id="date-picker",
                    min_date_allowed=datetime(2024, 11, 19),
                    max_date_allowed=datetime(2024, 12, 11),
                    initial_visible_month=datetime(2024, 1, 1),
                    date=datetime(2024, 1, 1),
                ),
    html.Div([
        html.Label("Enter Simulation Duration (in days):"),
        dcc.Input(id='duration-input', type='number', value=7, min=1, max=30, step=1),  # User-defined input
        html.Div(id='duration-error', style={'color': 'red', 'fontSize': 14}),  # Error message if input is invalid
    ]),
                html.Label("Enter Initial Activity (in Bq):"),
                dcc.Input(id='activity-input', type='number', value=500, min=0, step=100),
                html.Label("Enter Half-life (in hours):"),
                dcc.Input(id='half-life-input', type='number', value=1, min=0, step=0.01),
                html.Button("Simulate", id="simulate-button", style={"marginLeft": "10px"}),
            ],
            style={"margin": "20px"},
        ),
        dcc.Graph(id="trajectory-map"),

    # Add the logos at the bottom
    html.Div([
        html.Img(
            src='/assets/Ramones-Logo.jpeg',
            style={
                'height': '250px',  # Adjust the size of the logo (increase as needed)
                'width': 'auto',  # Maintain aspect ratio
                'display': 'block',  # Ensures proper positioning
                'marginLeft': '10px',  # Adds some space from the edge
            }
        ),
    ], style={
        'position': 'fixed',
        'bottom': '10px',  # Distance from the bottom
        'left': '10px',  # Align to the left
    }),

    html.Div([
        html.Img(
            src='/assets/schematic.png',
            style={
                'height': '250px',
                'width': 'auto',
                'display': 'block',
                'marginLeft': '1300px',
            }
        ),
    ], style={
        'position': 'fixed',
        'bottom': '10px',
        'left': '10px',
    }),

    html.Div([
        html.Img(
            src='/assets/opam.png',
            style={
                'height': '70px',
                'width': 'auto',
                'display': 'block',
                'marginLeft': '550px',
            }
        ),
    ], style={
        'position': 'fixed',
        'bottom': '10px',
        'left': '10px',
    })

    ]
)

def load_currents():

    # Load realistic ocean current data (replace with your netCDF file path)
    # Example using Copernicus dataset for the Mediterranean (https://data.marine.copernicus.eu/product/MEDSEA_ANALYSISFORECAST_PHY_006_013/description)

    dataset_path = "./data/cmems_mod_med_phy-cur_anfc_4.2km_P1D-m_1733300227925.nc"  # Replace with actual path
    # Check if the file exists
    try:
        os.stat(dataset_path)

    except FileNotFoundError:
        print("The ocean current data file does not exist.")
#        sys.exit(1)

    ds = xr.open_dataset(dataset_path)
    ds = ds.isel(depth=0)

    variables = {'U': 'uo', 'V': 'vo'}

    dimensions = {'U': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'},\
                  'V': {'lon': 'longitude', 'lat': 'latitude', 'time': 'time', 'depth': 'depth'} }

    fieldset = FieldSet.from_xarray_dataset(ds, variables, dimensions, mesh="spherical", allow_time_extrapolation=True)

    return fieldset

class PParticle(JITParticle):
    distance = Variable('distance', initial=0., dtype=np.float32) # the distance travelled
    prev_lon = Variable('prev_lon', dtype=np.float32, to_write=False,
                        initial=attrgetter('lon')) # the previous longitude
    prev_lat = Variable('prev_lat', dtype=np.float32, to_write=False,
                        initial=attrgetter('lat')) # the previous longitude
    initial_activity = Variable('initial_activity', dtype=np.float32, initial=500.0)  # Initial activity (Bq)
    activity = Variable('activity', dtype=np.float32, initial=500.0)  # Activity (Bq)
    half_life = Variable('half_life', dtype=np.float32, initial=1.0)  # Half-life initialized

# Keeping track of the total distance travelled by a particle:
def TotalDistance(particle, fieldset, time):
    """Calculate the distance in latitudinal direction
    (using 1.11e2 kilometer per degree latitude)"""
    lat_dist = (particle.lat - particle.prev_lat) * 1.11e2
    lon_dist = (
        (particle.lon - particle.prev_lon)
        * 1.11e2
        * math.cos(particle.lat * math.pi / 180)
    )
    # Calculate the total Euclidean distance travelled by the particle
    particle.distance += math.sqrt(math.pow(lon_dist, 2) + math.pow(lat_dist, 2))

    # Set the stored values for next iteration
    particle.prev_lon = particle.lon
    particle.prev_lat = particle.lat

def update_activity(particle, fieldset, time):
    """Decay activity based on initial activity, half-life, and elapsed time."""
    # Calculate the elapsed time since the particle was created (in seconds)
    elapsed_time = time - particle.time

    # Convert the half-life from hours to seconds (1 hour = 3600 seconds)
    half_life_seconds = particle.half_life * 3600

    # Calculate the decay constant (lambda) using the half-life
    decay_constant = math.log(2) / half_life_seconds  # Decay constant in 1/seconds

    # Calculate the new activity using the exponential decay formula
    particle.activity = particle.initial_activity * math.exp(decay_constant * elapsed_time) # back in time
    # particle.activity = particle.initial_activity * math.exp(-decay_constant * elapsed_time) # forward in time

def DeleteParticle(particle, fieldset, time):
    particle.delete()

# Function to simulate and save data to Zarr
def simulate_and_save_to_zarr(lon, lat, start_time, duration_input, zarr_file, activity, half_life):

    # load ocean currents dataset
    fieldset = load_currents()

    # Initialize ParticleSet
    pset = ParticleSet.from_list(
        fieldset=fieldset,
        pclass=PParticle,
        lon=[lon],
        lat=[lat],
        time=start_time,
	    initial_activity=activity,
	    half_life=half_life,
    )

    output_file = pset.ParticleFile(
         name=zarr_file,  # the file name
          outputdt = timedelta(hours=6),  # the time step of the outputs
	)
    # Simulate particle trajectories
    # In this example, particles running in backward time (dt<0)
    # To run forward in time, just set dt>0.

    pset.execute(
         pset.Kernel(AdvectionRK4) + pset.Kernel(TotalDistance) + pset.Kernel(update_activity),
        runtime=timedelta(days=duration_input),  # Simulate for 30 days
        dt=-timedelta(hours=1),
	output_file=output_file,
        recovery={ErrorCode.ErrorOutOfBounds: DeleteParticle}
    )

# Callback to simulate and create animation from Zarr
@app.callback(
    Output("trajectory-map", "figure"),
    Input("simulate-button", "n_clicks"),
    State("input-lon", "value"),
    State("input-lat", "value"),
    State("date-picker", "date"),
    State('duration-input', 'value'),
    State('activity-input', 'value'),
    State('half-life-input', 'value')
)
def update_trajectory(n_clicks, release_lon, release_lat, start_date, duration, activity, half_life):
    if n_clicks is None or release_lon is None or release_lat is None or start_date is None or duration is None:
        # Default empty map
        fig = go.Figure(go.Scattergeo())
        fig.update_layout(
            title="Radionuclides Trajectories in Backward Time",
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="white",
                countrycolor="gray",
                coastlinecolor="black",
                resolution=50,  # For higher resolution, use values like 50 or 110
                lonaxis=dict(
                    showgrid=True,
                    range=[-6, 36.5],  # Longitude range for the Mediterranean
                ),
                lataxis=dict(
                    showgrid=True,
                    range=[30, 46],  # Latitude range for the Mediterranean
                ),
                center=dict(
                    lon=15, # Approximate central longitude of the Mediterranean
                    lat=38,  # Approximate central latitude of the Mediterranean
                ),
	),
        )

        return fig

    # Ensure that inputs are valid floating-point numbers
    try:
        release_lon = float(release_lon)
        release_lat = float(release_lat)
        activity = float(activity)
        half_life = float(half_life)
    except ValueError:
        return go.Figure(go.Scattergeo())  # Return empty map if invalid inputs

    # Validate longitude and latitude ranges
    if not (-180 <= release_lon <= 180) or not (-90 <= release_lat <= 90):
        return go.Figure(go.Scattergeo())  # Return empty map if out-of-bounds

    # Parse start date
    start_date = datetime.fromisoformat(start_date)

    # Zarr file path for storing results
    zarr_file = "particle_trajectories.zarr"

    # Simulate and save to Zarr
    simulate_and_save_to_zarr(release_lon, release_lat, start_date, duration, zarr_file, activity, half_life)

    # Load data from Zarr
    ds = xr.open_zarr(zarr_file)
    times = ds.time.values
    lons = ds.lon.values
    lats = ds.lat.values
    activities = ds.activity.values

    # Create frames for animation
    frames = []
    for i, t in enumerate(times[0,:]):
        frames.append(
            go.Frame(
                data=[
                    go.Scattergeo(
                        lon=list(lons[:,:i].flatten()),
                        lat=list(lats[:,:i].flatten()),
                        mode="markers+lines",
#                        marker=dict(size=4, color="red"),
                        marker=dict(size=4, color=activities[:,i], colorscale='Viridis', cmin=0, cmax=1000, colorbar=dict(title='Activity (Bq)')),
                    )
                ] ,
                name=str(t),
            layout=go.Layout(
                annotations=[
                    dict(
                        text=f"Time: {str(t).split('.')[0]}",
                        x=0.5,  # Position on the plot (centered horizontally)
                        y=1.1,  # Above the plot area
                        xref="paper",
                        yref="paper",
                        showarrow=False,
                        font=dict(size=16, color="black"),
                    )
                ]
            ),
            )
        )

    # Initial positions
    fig = go.Figure(
        data=[
            go.Scattergeo(
                lon=lons[:, 0],
                lat=lats[:, 0],
                mode="markers",
                marker=dict(size=4, color="red"),
            )
        ],
        layout=go.Layout(
            title="Radionuclides Trajectories in Backward Time Animation",
            geo=dict(
                projection_type="natural earth",
                showland=True,
                landcolor="white",
                countrycolor="gray",
                coastlinecolor="black",
                resolution=50,
		lonaxis=dict(
                    showgrid=True,
                    range=[-6, 36.5],  # Longitude range for the Mediterranean
                ),
                lataxis=dict(
                    showgrid=True,
                    range=[30, 46],  # Latitude range for the Mediterranean
                ),
                center=dict(
                    lon=15,  # Approximate central longitude of the Mediterranean
                    lat=38,  # Approximate central latitude of the Mediterranean
		),
            ),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                            "label": "Play",
                            "method": "animate",
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                            "label": "Pause",
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 87},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 1.8,
                    "yanchor": "top",
                }
            ],
        ),
        frames=frames,
    )

    return fig

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True, host='127.0.0.1', port=8050)
