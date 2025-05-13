import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# define helper functions ------
# BEWARE! sliders require same type for steps/max etc.


def einsteinRelationReturnD(u, T):
    """
    Calculate the mobility coefficient using Einstein relation.
    D = (μ * k * T ) / q
    where:
    - D is the diffusion coefficient
    - μ is the mobility
    - k is Boltzmann's constant (1.38e-23 J/K)
    - T is temperature in Kelvin
    - q is charge of an electron (1.6e-19 C)
    """
    k = 1.38e-23  # Boltzmann's constant in J/K
    q = 1.6e-19  # Charge of an electron in C
    return (u * k * T) / (q)


# Set page configuration
st.set_page_config(page_title="Wow this deserves an A+", layout="wide")

st.title("Evolution of Charge Carriers Through Space and Time")
st.subheader("or: plotting n(x,t) and p(x,t)")

# Sidebar for parameters
st.sidebar.header("Parameters")

# actual stuff we'll use -----

# Add semiconductor physics parameters
st.sidebar.header("Semiconductor Parameters")

st.sidebar.write(
    "This app uses the following parameters for calculations. "
    "You can adjust them in the sidebar."
    "Default values are based on Silicon."
)
un = st.sidebar.number_input(
    "Electron mobility (μ), note e-4 for m^2/(V·s)",
    value=1200e-4,
    help="Electron mobility in m^2/(V·s)",
)
up = st.sidebar.number_input(
    "Hole mobility (μ), note e-4 for m^2/(V·s)",
    value=400e-4,
    help="Hole mobility in m^2/(V·s)",
)

# Temperature for Einstein relation calculations
T = st.sidebar.slider(
    "Temperature (T)",
    value=300.0,
    min_value=100.0,
    max_value=500.0,
    step=10.0,
    help="Temperature in Kelvin for Einstein relation calculations",
)

hvUnconverted = st.sidebar.slider(
    "Photon energy (hv)",
    value=2.0,
    min_value=0.1,
    max_value=10.0,
    step=0.1,
    help="Photon energy in eV",
)

hv = hvUnconverted * 1.6e-19  # Convert eV to Joules

n = st.sidebar.slider(
    "Efficiency (n)",
    value=1.0,
    min_value=0.0,
    max_value=1.0,
    step=0.01,
)

Po = st.sidebar.slider(
    "Power (Po)",
    min_value=0.1,
    max_value=11.0,
    value=3.0,
    step=1.0,
    help="Units in this app: mW",
)

V = st.sidebar.slider(
    "Voltage (V)",
    min_value=0.1,
    max_value=11.0,
    value=2.5,
    step=0.5,
    help="Units in this app: V",
)

L = st.sidebar.slider(
    "Length (L) in m, where 1m = 1e6 μm (appears as 0)",
    min_value=0.1e-6,
    max_value=20e-6,
    value=10e-6,
    step=1e-6,
    help="Units in this app: m",
)

st.sidebar.write(f"Length (L): {L:.9f} m")


E = V / L

st.sidebar.write(f"Electric field (E): {E:.2f} V/m from V / L")

x0 = st.sidebar.slider(
    "Initial position (x0)",
    min_value=0.0,
    max_value=1.0,
    value=0.2,
    step=0.05,
    help="Offset in percent terms of L (0 to L)",
)


# Defining dependents
D_n = einsteinRelationReturnD(un, T)
D_p = einsteinRelationReturnD(up, T)
tau_n = L**2 / (D_n)
tau_p = L**2 / (D_p)
vn = un * E
vp = up * E

st.sidebar.write("Values calculated from the above parameters:")
st.sidebar.write(f"Diffusion coefficient (D_n): {D_n:.2e} m^2/s")
st.sidebar.write(f"Diffusion coefficient (D_p): {D_p:.2e} m^2/s")
st.sidebar.write(f"Diffusion time (tau_n): {tau_n:.2e} s")
st.sidebar.write(f"Diffusion time (tau_p): {tau_p:.2e} s")
st.sidebar.write(f"Drift velocity (vn): {vn:.2e} m/s")
st.sidebar.write(f"Drift velocity (vp): {vp:.2e} m/s")

######################################################
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Define the crucial functions
##########################################################

# def for electrons


def n_carrierPropagation(x, t):
    chunk1 = (Po * n) / (hv)
    chunk2 = np.exp((-1 * t) / tau_n)
    chunk3 = np.sqrt(1 / (4 * np.pi * D_n * t))
    chunk4top = -1 * ((x - x0) + vn * t) ** 2
    chunk4bottom = 4 * D_n * t
    chunk4 = np.exp(chunk4top / chunk4bottom)
    soln = chunk1 * chunk2 * chunk3 * chunk4
    return soln


# Define for holes


def p_carrierPropagation(x, t):
    chunk1 = (Po * n) / (hv)
    chunk2 = np.exp((-1 * t) / tau_p)
    chunk3 = np.sqrt(1 / (4 * np.pi * D_p * t))
    chunk4top = -1 * ((x - x0) + vp * t) ** 2
    chunk4bottom = 4 * D_p * t
    chunk4 = np.exp(chunk4top / chunk4bottom)
    soln = chunk1 * chunk2 * chunk3 * chunk4
    return soln


# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXxx
# DEFINE GRAPHING BOUJNDS
# Define the time and space arrays
x_values = np.geomspace(1.98e-23, 2e-23, 10000)
t_values = np.geomspace(1e-9, 1e-4, 10000)  # Time values from 0 to 1 microsecond

figElectrons = px.line(
    x=x_values,
    y=n_carrierPropagation(x_values, t_values),
    title="Electron Carrier Propagation",
    labels={"x": "Distance (m)", "y": "Carrier Concentration (m^-3)"},
)
figElectrons.update_traces(line_color="red")

figHoles = px.line(
    x=x_values,
    y=p_carrierPropagation(x_values, t_values),
    title="Hole Carrier Propagation",
    labels={"x": "Distance (m)", "y": "Carrier Concentration (m^-3)"},
)
figHoles.update_traces(line_color="cyan")


# now, actually plot together.
# --------------------------------------------

from plotly.subplots import make_subplots


# Create a subplot figure
figTurbo = make_subplots(rows=1, cols=1)


# Add traces from figElectrons
for trace in figElectrons.data:
    figTurbo.add_trace(trace)

# Add traces from figHoles
for trace in figHoles.data:
    figTurbo.add_trace(trace)

# Update layout
figTurbo.update_layout(
    title="Carrier Propagation in Semiconductor",
    xaxis_title="Distance (m)",
    yaxis_title="Carrier Concentration (m^-3)",
    #    legend_title_text='Carrier Type',
    #    legend=dict(
    #        itemsizing='constant',
    #        orientation='h',
    #        yanchor='bottom',
    #        y=1.02,
    #        xanchor='right',
    #        x=1
    #    )
)

# Show the combined figure
st.plotly_chart(figTurbo, use_container_width=True)

st.write("Where electrons are shown in RED and holes in BLUE.")
