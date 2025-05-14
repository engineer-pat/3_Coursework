import streamlit as st
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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

x0_unconverted = st.sidebar.slider(
    "Initial position (x0)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="Offset in percent terms of L (0 to L)",
)

x0 = x0_unconverted * L

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


# 1) pick grids
x = np.linspace(0, L, 200)
t_vals = np.linspace(
    1e-20, 1e-9, 1000
)  # 1000 points from 1e-20 to 1e-9, beyond nanosecond phenomena are not intersting

# 2) assemble
rows = []
for t in t_vals:
    yn = n_carrierPropagation(x, t)
    yp = p_carrierPropagation(x, t)
    for xi, yi in zip(x, yn):
        rows.append({"x": xi, "conc": yi, "carrier": "electrons", "time": t})
    for xi, yi in zip(x, yp):
        rows.append({"x": xi, "conc": yi, "carrier": "holes", "time": t})

df = pd.DataFrame(rows)


# 3) animate
fig = px.line(
    df,
    x="x",
    y="conc",
    color="carrier",
    animation_frame="time",
    range_x=[0, L],
    range_y=[0, df.conc.max() * 1.1],  # <— this locks your y-axis
    labels={"x": "Distance (m)", "conc": "Conc. (m⁻³)"},
)


def J_current(x, t):
    """
    Calculate the current density using the Einstein relation.
    J = q * (n * vn + p * vp)
    where:
    - J is the current density
    - q is charge of an electron (1.6e-19 C)
    - n is electron concentration
    - p is hole concentration
    """
    n_conc = n_carrierPropagation(x, t)
    p_conc = p_carrierPropagation(x, t)
    JChunk1 = 1.6e-19 * (n_conc * vn + p_conc * vp) * E
    derivnChunk1 = (
        Po * n / (hv) * np.exp((-1 * t) / tau_n) * np.sqrt(1 / (4 * np.pi * D_n * t))
    )
    derivnChunk2 = ((x - x0) + vn * t) / (2 * D_n * t)
    derivnChunk3 = np.exp(-1 * ((x - x0) + vn * t) ** 2 / (4 * D_n * t))
    derivn = derivnChunk1 * derivnChunk2 * derivnChunk3
    derivpChunk1 = (
        Po * n / (hv) * np.exp((-1 * t) / tau_p) * np.sqrt(1 / (4 * np.pi * D_p * t))
    )
    derivpChunk2 = ((x - x0) - vp * t) / (2 * D_p * t)
    derivpChunk3 = np.exp(-1 * ((x - x0) - vp * t) ** 2 / (4 * D_p * t))
    derivp = derivpChunk1 * derivpChunk2 * derivpChunk3
    JChunk2 = 1.6e-19 * D_n * derivn
    JChunk3 = 1.6e-19 * D_p * derivp

    J = JChunk1 + JChunk2 - JChunk3

    return J


# --- build dfJ, one row per (x,t) ---
rowsJ = []
for t in t_vals:
    J_vals = J_current(x, t)
    for xi, Ji in zip(x, J_vals):
        rowsJ.append({"x": xi, "J": Ji, "time": t})
dfJ = pd.DataFrame(rowsJ)


# --- compute a y-range that will holds peaks ---
ymax = dfJ.J.abs().max() * 1.1

# --- animate J(x,t) just like n/p ---
figJ = px.line(
    dfJ,
    x="x",
    y="J",
    animation_frame="time",
    range_x=[0, L],
    range_y=[-ymax, ymax],  # or [0, ymax] if J never goes negative
    labels={"x": "Distance (m)", "J": "Current (A/m²)"},
)

st.title("J(x,t) Animation")
st.plotly_chart(figJ, use_container_width=True)
# TRANSIENT TIME RESPONSE SIGNAL

# fig
st.title("n(x,t) and p(x,t) Animation")
st.plotly_chart(fig, use_container_width=True)
# st.plotly_chart(figJ, use_container_width=True)

# 4) Add dataframe to enjoy the data
st.title("n(x,t) and p(x,t) DataFrame")
st.dataframe(df, use_container_width=True)
st.title("J(x,t) DataFrame")
st.dataframe(dfJ, use_container_width=True)
# st.table(df)
