import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
import Neuro_Lib as NL
class Gating_var:
  def alpha_m(V):
      return 0.1 * (V + 40) / (1 - np.exp(-(V + 40) / 10))

  def beta_m(V):
      return 4.0 * np.exp(-(V + 65) / 18)

  def alpha_n(V):
      return 0.01 * (V + 55) / (1 - np.exp(-(V + 55) / 10))
    
  def beta_n(V):
      return 0.125 * np.exp(-(V + 65) / 80)

  def alpha_h(V):
      return 0.07 * np.exp(-(V + 65) / 20)

  def beta_h(V):
      return 1 / (1 + np.exp(-(V + 35) / 10))
  
  
def Hodgkin_Huxley(C_m,g_Na,g_K,g_L,E_Na,E_K,E_L,V):
  # Time variables
  dt = 0.01  # time step, ms
  time = np.arange(0, 50, dt)  # time array

  # Initial conditions
 # Initial membrane potential, in mV
  m = NL.alpha_m(V) / (NL.alpha_m(V) + NL.beta_m(V))  # Initial m-gate value
  n = NL.alpha_n(V) / (NL.alpha_n(V) + NL.beta_n(V))  # Initial n-gate value
  h = NL.alpha_h(V) / (NL.alpha_h(V) + NL.beta_h(V))  # Initial h-gate value

  # External stimulus
  I_ext = np.zeros(len(time))
  I_ext[2000:5000] = 10  # Apply current stimulus from 5 to 15 ms

  # Store results
  V_trace = np.zeros(len(time))
  m_trace = np.zeros(len(time))
  n_trace = np.zeros(len(time))
  h_trace = np.zeros(len(time))

  # Arrays to store the dynamics
  m_values = np.zeros(len(time))
  n_values = np.zeros(len(time))
  h_values = np.zeros(len(time))

  # Simulation loop using Euler's method
  for i in range(len(time)):
      # Calculate ionic currents
      I_Na = g_Na * (m**3) * h * (V - E_Na)
      I_K = g_K * (n**4) * (V - E_K)
      I_L = g_L * (V - E_L)
      
      # Update membrane potential
      dVdt = (I_ext[i] - I_Na - I_K - I_L) / C_m
      V += dVdt * dt
      
      # Update gating variables
      dm_dt = NL.alpha_m(V) * (1 - m) - NL.beta_m(V) * m
      dn_dt = NL.alpha_n(V) * (1 - n) - NL.beta_n(V) * n
      dh_dt = NL.alpha_h(V) * (1 - h) - NL.beta_h(V) * h
      
      m += dm_dt * dt
      n += dn_dt * dt
      h += dh_dt * dt
      
      # Store values for plotting
      m_values[i] = m
      n_values[i] = n
      h_values[i] = h
      
      # Store values
      V_trace[i] = V
      m_trace[i] = m
      n_trace[i] = n
      h_trace[i] = h
  return time,V_trace,m_trace,n_trace,h_trace