import numpy as np
import pandas as pd
import plotly.express as px
from scipy.special import erfc

# Function definitions


def fn_D_T(Do, Ea, T):
    k = 8.617e-5  # Boltzmann constant, eV/K
    return Do * np.exp(-Ea / (k * T))


def fn_diffLength(D, t):
    return np.sqrt(D * t)


def fn_C_predep(x, t, Cs, D):
    # Pre-deposition concentration profile
    return Cs * erfc(x / (2 * np.sqrt(D * t)))


def fn_Qt_Qo(D, t, Cs):
    # Total dopant content at time t
    return (2 * Cs) * np.sqrt(D * t) / np.sqrt(np.pi)


def fn_C_drivein(x, Qo, D, t):
    # Drive-in concentration profile
    outside = Qo / np.sqrt(np.pi * D * t)
    inside = -(x**2) / (4 * D * t)
    return outside * np.exp(inside)


# Problem 1: Pre-deposition
T1 = 975 + 273.15  # K
Do_P = 3.85  # cm^2/s
Ea = 3.66  # eV
D1 = fn_D_T(Do_P, Ea, T1)
t1 = 30 * 60  # seconds
Cs = 1e21  # surface concentration, cm^-3
Csub = 1e17  # substrate concentration, cm^-3

# Compute pre-deposition profile and figure
x_um1 = np.linspace(0, 1, 100)
x_cm1 = x_um1 * 1e-4
C_pre = fn_C_predep(x_cm1, t1, Cs, D1)
df1 = pd.DataFrame({"x (µm)": x_um1, "C (predep)": C_pre})
fig1 = px.scatter(
    df1,
    x="x (µm)",
    y="C (predep)",
    title="Pre-deposition Concentration Profile",
    labels={"x (µm)": "Position (µm)", "C (predep)": "Concentration"},
)
fig1.update_traces(mode="lines+markers")

# Problem 2: Drive-in
T2 = 1100 + 273.15  # K
t2 = 20 * 60  # seconds
D2 = fn_D_T(Do_P, Ea, T2)
Qt = fn_Qt_Qo(D1, t1, Cs)

# Compute drive-in profile and figure
x_um2 = x_um1  # reuse same positions
x_cm2 = x_um2 * 1e-4
C_drive = fn_C_drivein(x_cm2, Qt, D2, t2)
df2 = pd.DataFrame({"x (µm)": x_um2, "C (drivein)": C_drive})
fig2 = px.scatter(
    df2,
    x="x (µm)",
    y="C (drivein)",
    title="Drive-in Concentration Profile",
    labels={"x (µm)": "Position (µm)", "C (drivein)": "Concentration"},
)
fig2.update_traces(mode="lines+markers")

# Merge figures into fig3
fig3 = fig1
for trace in fig2.data:
    fig3.add_trace(trace)

fig3.update_layout(
    title="Concentration Profiles: Pre-deposition and Drive-in",
    legend=dict(title="Profiles"),
    xaxis_title="Position (µm)",
    yaxis_title="Concentration",
)

# Render only fig3
fig3.show()
