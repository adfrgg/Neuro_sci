import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
# Global Variables (Initial Values)
C_m = 1.0  # membrane capacitance, uF/cm^2
g_Na = 120.0  # Sodium conductance, mS/cm^2
g_K = 36.0  # Potassium conductance, mS/cm^2
g_L = 0.3  # Leak conductance, mS/cm^2
E_Na = 50.0  # Sodium reversal potential, mV
E_K = -77.0  # Potassium reversal potential, mV
E_L = -54.387  # Leak reversal potential, mV
V_init = -80.0  # Initial membrane potential, mV

# Define gating variable equations
def alpha_n(V):
    return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)

def alpha_m(V):
    return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

def beta_m(V):
    return 4.0 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1 / (1 + np.exp(-(V + 35) / 10))

# Function to compute membrane potential update
def compute_membrane_potential(V, I_ext, g_Na, g_K, g_L, E_Na, E_K, E_L, m, h, n, C_m):
    I_Na = g_Na * (m**3) * h * (V - E_Na)
    I_K = g_K * (n**4) * (V - E_K)
    I_L = g_L * (V - E_L)
    
    dVdt = (I_ext - I_Na - I_K - I_L) / C_m
    return dVdt #membrane potential over time 

# Function to update gating variables
def update_gating_variables(V, m, h, n, dt):
    dm_dt = alpha_m(V) * (1 - m) - beta_m(V) * m
    dn_dt = alpha_n(V) * (1 - n) - beta_n(V) * n
    dh_dt = alpha_h(V) * (1 - h) - beta_h(V) * h

    m += dm_dt * dt
    n += dn_dt * dt
    h += dh_dt * dt
    
    return m, n, h #gating variables m-Na activate, n-K activate, h-Na inactivate (Probability)

# Function to simulate the Hodgkin-Huxley model
def simulate_hodgkin_huxley(C_m, g_Na, g_K, g_L, E_Na, E_K, E_L, V_init, I_stim_amp, stim_start, stim_duration):
    # Time variables
    dt = 0.01  # time step, ms
    time = np.arange(0, 50, dt)  # time array
    
    # External stimulus
    I_ext = np.zeros(len(time))
    I_ext[int(stim_start/dt):int((stim_start + stim_duration)/dt)] = I_stim_amp  # Apply current stimulus
    
    # Initial conditions
    V = V_init  # Initial membrane potential
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))  # Initial m-gate value
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))  # Initial n-gate value
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))  # Initial h-gate value
    
    # Store results
    V_trace = np.zeros(len(time))
    m_trace = np.zeros(len(time))
    n_trace = np.zeros(len(time))
    h_trace = np.zeros(len(time))

    # Simulation loop using Euler's method
    for i in range(len(time)):
        dVdt = compute_membrane_potential(V, I_ext[i], g_Na, g_K, g_L, E_Na, E_K, E_L, m, h, n, C_m)
        V += dVdt * dt
        m, n, h = update_gating_variables(V, m, h, n, dt)

        # Store values
        V_trace[i] = V
        m_trace[i] = m
        n_trace[i] = n
        h_trace[i] = h

    return time, V_trace, m_trace, n_trace, h_trace #membrane potentail and gating variables over time 

# Function to plot the results
def plot_potential_gating_var(time, V_trace, m_trace, n_trace, h_trace):
    plt.figure(figsize=(12, 8))

    # Plot Membrane Potential
    plt.subplot(4, 1, 1)
    plt.plot(time, V_trace, label='Membrane potential (V)')
    plt.ylabel('V (mV)')
    plt.title('Hodgkin-Huxley Model: Membrane Potential and Gating Variables')
    plt.legend()

    # Plot m-gate
    plt.subplot(4, 1, 2)
    plt.plot(time, m_trace, label='m-gate (Na activation)', color='r')
    plt.ylabel('m')
    plt.legend()

    # Plot n-gate
    plt.subplot(4, 1, 3)
    plt.plot(time, n_trace, label='n-gate (K activation)', color='g')
    plt.ylabel('n')
    plt.legend()

    # Plot h-gate
    plt.subplot(4, 1, 4)
    plt.plot(time, h_trace, label='h-gate (Na inactivation)', color='b')
    plt.xlabel('Time (ms)')
    plt.ylabel('h')
    plt.legend()

    plt.tight_layout()
    plt.show()