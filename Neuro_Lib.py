import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import Neuro_Lib as NL
# Global parameters
params = {
    'C_m': 1.0,
    'g_Na': 120.0,
    'g_K': 36.0,
    'g_L': 0.3,
    'E_Na': 50.0,
    'E_K': -77.0,
    'E_L': -54.0,
    'V_init': -80.0
}

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
# Function to update global parameters
def update_params(change):
    params[change.owner.description] = change.new

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
def simulate_hodgkin_huxley():
    

    # Time variables
    dt = 0.01  # time step, ms
    time = np.arange(0, 50, dt)  # time array
    
    # Initial conditions from global params
    V = params['V_init']
    m = alpha_m(V) / (alpha_m(V) + beta_m(V))
    n = alpha_n(V) / (alpha_n(V) + beta_n(V))
    h = alpha_h(V) / (alpha_h(V) + beta_h(V))
    
    # Arrays for simulation data
    V_trace, m_trace, n_trace, h_trace = np.zeros_like(time), np.zeros_like(time), np.zeros_like(time), np.zeros_like(time)
    
    for i, t in enumerate(time):
        I_Na = params['g_Na'] * m**3 * h * (V - params['E_Na'])
        I_K = params['g_K'] * n**4 * (V - params['E_K'])
        I_L = params['g_L'] * (V - params['E_L'])
        dVdt = (- I_Na - I_K - I_L) / params['C_m']
        V += dVdt * dt
        m += (alpha_m(V) * (1 - m) - beta_m(V) * m) * dt
        n += (alpha_n(V) * (1 - n) - beta_n(V) * n) * dt
        h += (alpha_h(V) * (1 - h) - beta_h(V) * h) * dt
        V_trace[i], m_trace[i], n_trace[i], h_trace[i] = V, m, n, h

    return time, V_trace, m_trace, n_trace, h_trace

# Function to create the sliders and update global variables
# Create interactive controls
def create_controls():
    layout = widgets.Layout(width='450px')
    sliders = {
        'C_m': widgets.FloatSlider(value=params['C_m'], min=1., max=10., step=1., description='C_m', layout=layout),
        'g_Na': widgets.FloatSlider(value=params['g_Na'], min=0., max=150., step=5., description='g_Na', layout=layout),
        'g_K': widgets.FloatSlider(value=params['g_K'], min=0., max=50., step=5., description='g_K', layout=layout),
        'g_L': widgets.FloatSlider(value=params['g_L'], min=0., max=1., step=0.1, description='g_L', layout=layout),
        'E_Na': widgets.FloatSlider(value=params['E_Na'], min=0., max=100., step=5., description='E_Na', layout=layout),
        'E_K': widgets.FloatSlider(value=params['E_K'], min=-100., max=0., step=5., description='E_K', layout=layout),
        'E_L': widgets.FloatSlider(value=params['E_L'], min=-100., max=100., step=5., description='E_L', layout=layout),
        'V_init': widgets.FloatSlider(value=params['V_init'], min=-100., max=50., step=5., description='V_init', layout=layout)
    }
    for slider in sliders.values():
        slider.observe(update_params, names='value')
    return widgets.VBox(list(sliders.values()))