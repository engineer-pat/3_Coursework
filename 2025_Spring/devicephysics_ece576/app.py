import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp


def einsteinRelationReturnMu(D, T):
    """
    Calculate the mobility coefficient using Einstein relation.
    μ = (D * q) / (k * T)
    where:
    - D is the diffusion coefficient
    - μ is the mobility
    - k is Boltzmann's constant (1.38e-23 J/K)
    - T is temperature in Kelvin
    - q is charge of an electron (1.6e-19 C)
    """
    k = 1.38e-23  # Boltzmann's constant in J/K
    q = 1.6e-19  # Charge of an electron in C
    return (D * q) / (k * T)


def einsteinRelationReturnD(mu, T):
    """
    Calculate the diffusion coefficient using Einstein relation.
    D = (μ * k * T) / q
    where:
    - D is the diffusion coefficient
    - μ is the mobility
    - k is Boltzmann's constant (1.38e-23 J/K)
    - T is temperature in Kelvin
    - q is charge of an electron (1.6e-19 C)
    """
    k = 1.38e-23  # Boltzmann's constant in J/K
    q = 1.6e-19  # Charge of an electron in C
    return (mu * k * T) / q


# Set page configuration
st.set_page_config(page_title="Wow this deserves an A+", layout="wide")

st.title("Evolution of Charge Carriers Through Space and Time")
st.subheader("or: plotting n(x,t) and p(x,t)")

# Sidebar for parameters
st.sidebar.header("Parameters")

# actual stuff we'll use -----

# Add semiconductor physics parameters
st.sidebar.header("Semiconductor Parameters")

# Temperature for Einstein relation calculations
temperature = st.sidebar.number_input(
    "Temperature (T)",
    value=300.0,
    min_value=100.0,
    max_value=500.0,
    step=1.0,
    help="Temperature in Kelvin for Einstein relation calculations",
)

# Material selector for diffusion coefficients
material = st.sidebar.selectbox(
    "Select Semiconductor Material",
    [
        "Silicon",
        "Gallium Arsenide",
        "Germanium",
        "Gallium Nitride",
        "Silicon Carbide",
        "Custom",
    ],
    help="Choose a semiconductor material to use predefined diffusion coefficients",
)

# Define mobility values for different materials
material_params = {
    "Silicon": {"μ_n": 1200e-4, "μ_p": 400e-4},  # FIXME!!! Maybe meters or cm
    "Gallium Arsenide": {"μ_n": 8000, "μ_p": 400},
    "Germanium": {"μ_n": 3900, "μ_p": 1900},
    "Gallium Nitride": {"μ_n": 1000, "μ_p": 200},
    "Silicon Carbide": {"μ_n": 800, "μ_p": 120},
    "Custom": {"μ_n": 1000, "μ_p": 400},
}

# Set diffusion coefficients based on material selection
if material == "Custom":
    μ_n = st.sidebar.number_input(
        "Electron mobility (μ_n)",
        value=1000.0,
        min_value=1.0,
        step=10.0,
        help="Typical units: cm²/Vs",
    )

    μ_p = st.sidebar.number_input(
        "Hole mobility (μ_p)",
        value=400.0,
        min_value=1.0,
        step=10.0,
        help="Typical units: cm²/Vs",
    )

    # Calculate diffusion coefficients using Einstein relation
    D_n = einsteinRelationReturnD(μ_n, temperature)
    D_p = einsteinRelationReturnD(μ_p, temperature)

    st.sidebar.info(f"D_n = {D_n:.2f} cm²/s (calculated from mobility)")
    st.sidebar.info(f"D_p = {D_p:.2f} cm²/s (calculated from mobility)")
else:
    # Get mobility values from the material parameters
    μ_n = material_params[material]["μ_n"]
    μ_p = material_params[material]["μ_p"]

    # Calculate diffusion coefficients using Einstein relation
    D_n = einsteinRelationReturnD(μ_n, temperature)
    D_p = einsteinRelationReturnD(μ_p, temperature)

    st.sidebar.info(f"Using material properties for {material}:")
    st.sidebar.info(f"μ_n = {μ_n} cm²/Vs, μ_p = {μ_p} cm²/Vs")
    st.sidebar.info(
        f"D_n = {D_n:.2f} cm²/s, D_p = {D_p:.2f} cm²/s (calculated via Einstein relation)"
    )

# Add light/potential parameters
st.sidebar.header("Light and Device Parameters")

# Power Po
P_0 = st.sidebar.slider(
    "Power (Po)",
    min_value=0.1,
    max_value=11.0,
    value=3.0,
    step=1.0,
    help="Units in this app: mW",
)

n = st.sidebar.slider(
    "Efficiency (n)",
    min_value=0.1,
    max_value=1.0,
    value=1.0,
    step=0.05,
    help="For this app, we'll use 1.0 as default. This is the efficiency of the light source.",
)

hv = 3.2044e-19  # J
# st.sidebar.slider(
#     "Photon energy (hv)",
#     min_value=0.1e-19,
#     max_value=10e-19,
#     value=3.2044e-19,
#     step=1e-19,
#     help="Units in this app: J",
# )

V = st.sidebar.slider(
    "Voltage (V)",
    min_value=0.1,
    max_value=10.0,
    value=2.5,
    step=0.1,
    help="Units in this app: V",
)

L = 5e-6  # 10 μm
# st.sidebar.slider(
#     "Length (L)",
#     min_value=0.1,
#     max_value=100.0,
#     value=10.0,
#     step=1.0,
#     help="Units in this app: μm",
# )

# Calculate the electric field
E = V / L
st.sidebar.info(f"E = {E:.2f} V/μm (calculated from voltage and length)")

# Calculate the carrier velocities v = μ * E
carrierVelocity_n = μ_n * E
carrierVelocity_p = μ_p * E
st.sidebar.info(f"Carrier velocity (n) = {carrierVelocity_n:.2f} cm/s")
st.sidebar.info(f"Carrier velocity (p) = {carrierVelocity_p:.2f} cm/s")


# Add other parameters
st.sidebar.header("Other Parameters")

# Initial x_o
x_0 = 0
# x_0 = st.sidebar.slider(
#     "Initial position (x₀)", min_value=0.1, max_value=L, value=0.0, step=0.1
# )

# t_0 = st.sidebar.slider(
#     "Initial time (t₀)", min_value=0.0, max_value=20.0, value=0.0, step=0.1
# )


# crap to demonstrate how this app can work -----
# ---------------------

# # Initial condition
# y0 = st.sidebar.number_input("Initial value (y₀)", value=10.0, step=0.1)

# # Decay constant
# k = st.sidebar.number_input(
#     "Decay constant (k)", value=0.5, min_value=0.01, max_value=10.0, step=0.1
# )

# Time range
# t_max = st.sidebar.slider(
#     "Maximum time", min_value=1.0, max_value=20.0, value=10.0, step=0.5
# )

# Number of points
# st.sidebar.slider(
#     "Number of points", min_value=50, max_value=1000, value=200, step=50
# )

# Carrier lifetimes
tau_n = 32.05e-9  # ns
tau_p = 32.05e-9  # ns

# st.sidebar.number_input(
#     "Electron lifetime (τ_n)",
#     value=1.0,
#     min_value=0.01,
#     step=0.01,
#     help="Typical units: μs",
# )

# tau_p = st.sidebar.number_input(
#     "Hole lifetime (τ_p)",
#     value=0.5,
#     min_value=0.01,
#     step=0.01,
#     help="Typical units: μs",
# )


# Position and time references
# st.sidebar.header("Reference Points")
# x_0 = st.sidebar.number_input(
#     "Reference position (x₀)",
#     value=0.0,\
#     step=0.1,
#     min_value=0.0,
#     max_value=L,
#     help="Typical units: μm",
# )

# Unneeded in this implementation, but keeping for reference
# t_0 = st.sidebar.number_input(
#     "Reference time (t₀)", value=0.0, step=0.1, help="Typical units: ns"
# )


##################################################################
# The EQUATIONS!!!!!!!!
#################################################################
def n_carrierPropagation(x, t):
    chunk1 = (P_0 * n) / (hv)
    chunk2 = np.exp((-1 * t) / tau_n)
    chunk3 = np.sqrt(1 / (4 * np.pi * D_n * t))
    chunk4top = -1 * ((x - x_0) + carrierVelocity_n * t) ** 2
    chunk4bottom = 4 * D_n * t
    chunk4 = np.exp(chunk4top / chunk4bottom)
    soln = chunk1 * chunk2 * chunk3 * chunk4
    return soln


def p_carrierPropagation(x, t):
    chunk1 = (P_0 * n) / (hv)
    chunk2 = np.exp((-1 * t) / tau_p)
    chunk3 = np.sqrt(1 / (4 * np.pi * D_p * t))
    chunk4top = -1 * ((x - x_0) + carrierVelocity_p * t) ** 2
    chunk4bottom = 4 * D_p * t
    chunk4 = np.exp(chunk4top / chunk4bottom)
    solp = chunk1 * chunk2 * chunk3 * chunk4
    return solp


# Add time slider for visualization
# t_selected = st.sidebar.slider(
#     "Time for visualization (t)",
#     min_value=1e-9,
#     max_value=1e-6,
#     value=5e-8,
#     format="%.2e",
#     help="Time at which to visualize carrier distribution (s)",
# )

# Use geometrically spaced x values to better visualize the carrier distribution
# This gives more points near x_0 where the distribution changes quickly
x_values = np.geomspace(L / 100, L)  # Start near zero but not at zero

# For time-specific visualization
t_values = np.geomspace(1e-9, 1e-6)

# Create plots
# col1, col2 = st.columns(2)

# For n(x,t)
# with col1:
st.subheader("n(x,t) Electron Carrier Propagation")
fig_n = px.line(
    x=x_values * 1e6,  # Convert to μm for display
    y=n_carrierPropagation(x_values, t_values),
    labels={"x": "Position (μm)", "y": "n(x,t)"},
    # title=f"n(x,t) at t = {t_values*1e9:.2f} ns",
)
fig_n.update_traces(line=dict(color="blue", width=2))
fig_n.update_layout(
    xaxis_title="Position (μm)",
    yaxis_title="Electron Concentration n(x,t)",
    xaxis=dict(
        range=[0, min(10, L * 1e6)]
    ),  # Limit x-axis range to see the peak better
)
st.plotly_chart(fig_n, use_container_width=True)

# For p(x,t)
st.subheader("p(x,t) Hole Carrier Propagation")
fig_p = px.line(
    x=x_values * 1e6,  # Convert to μm for display
    y=p_carrierPropagation(x_values, t_values),
    labels={"x": "Position (μm)", "y": "p(x,t)"},
    # title=f"p(x,t) at t = {t_values*1e9:.2f} ns",
)
fig_p.update_traces(line=dict(color="red", width=2))
fig_p.update_layout(
    xaxis_title="Position (μm)",
    yaxis_title="Hole Concentration p(x,t)",
    xaxis=dict(
        range=[0, min(10, L * 1e6)]
    ),  # Limit x-axis range to see the peak better
)
st.plotly_chart(fig_p, use_container_width=True)

# # Combined plot
# st.subheader("Comparison of Solutions")
# fig_combined = go.Figure()
# fig_combined.add_trace(
#     go.Scatter(
#         x=solution.t,
#         y=solution.y[0],
#         mode="lines",
#         name="Numerical",
#         line=dict(color="blue", width=2),
#     )
# )
# fig_combined.add_trace(
#     go.Scatter(
#         x=t_analytical,
#         y=y_analytical,
#         mode="lines",
#         name="Analytical",
#         line=dict(color="red", width=2, dash="dash"),
#     )
# )
# fig_combined.update_layout(
#     title="Comparison of Numerical and Analytical Solutions",
#     xaxis_title="Time",
#     yaxis_title="y(t)",
#     legend_title="Solution Type",
# )
# st.plotly_chart(fig_combined, use_container_width=True)


# # Display material properties table
# st.subheader("Semiconductor Material Properties")
# st.markdown(
#     "The table below shows typical mobility values for common semiconductor materials at room temperature. Diffusion coefficients are calculated using the Einstein relation."
# )

# # Calculate diffusion coefficients for the table using Einstein relation
# D_n_values = []
# D_p_values = []
# mobilities_n = [1450, 8000, 3900, 1000, 800]  # μ_n values
# mobilities_p = [500, 400, 1900, 200, 120]  # μ_p values

# for μ_n, μ_p in zip(mobilities_n, mobilities_p):
#     D_n_values.append(round(einsteinRelationReturnD(μ_n, 300.0), 2))  # at 300K
#     D_p_values.append(round(einsteinRelationReturnD(μ_p, 300.0), 2))  # at 300K

# # Create a DataFrame for the table
# material_data = {
#     "Material": [
#         "Silicon",
#         "Gallium Arsenide",
#         "Germanium",
#         "Gallium Nitride",
#         "Silicon Carbide",
#     ],
#     "Electron Mobility (μ₍ₙ₎) [cm²/Vs]": mobilities_n,
#     "Hole Mobility (μ₍ₚ₎) [cm²/Vs]": mobilities_p,
#     "Electron Diffusion Coefficient (D₍ₙ₎) [cm²/s]": D_n_values,
#     "Hole Diffusion Coefficient (D₍ₚ₎) [cm²/s]": D_p_values,
# }

# # Highlight the currently selected material
# if material != "Custom":
#     st.markdown(f"**Currently using: {material}**")
#     st.markdown(f"**Temperature: {temperature} K**")

# # Display the table
# st.table(material_data)

# st.markdown(
#     """
# *Note: Diffusion coefficients are calculated using the Einstein relation at the specified temperature. Actual values may vary based on doping concentration, temperature, and other factors.*
# """
# )
