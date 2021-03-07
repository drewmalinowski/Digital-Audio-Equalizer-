# =============================================================================
# Drew Malinowski
# Audio EQ Program
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.fftpack import fft, fftshift
from scipy.optimize import fsolve
Pi = np.pi
plt.rcParams.update({'font.size': 10})


# =============================================================================
# CHOOSE EQ PARAMATERS:
# =============================================================================

# Choose a sample from a song
SONG = 'Garcon.csv'

# Select Cut-off frequencies in Hz:
f1 = 200
f2 = 300

# Select gain ratio:
Output_Ratio = 1

# time step (use Sample_Rate_Finder.py to find this value)
dt = 2.2675736961451248e-05


# =============================================================================
# Load Signal Data
# =============================================================================

df = pd.read_csv(SONG) 
Signal_L = df['L'].values
Signal_R = df['R'].values
t = np.arange(0,dt*len(Signal_L),dt)

# =============================================================================
# Note: Sampling frequency (fs) is equal to number of samples per second,
# hence the amount of samples (len(sensor_sig)) divided by the time span:
# =============================================================================

fs = len(Signal_L)/(t.max()-t.min())
x = np.zeros(len(Signal_L))

# =============================================================================
# Use Fourier Transform to decompose the noisy signal. Use this information
# to create a bandpass filter that attenuates the unwanted noise. 
# =============================================================================

def Fast_Fourier_Transform(x,fs):
    N = len(x)
    X_fft = fft(x)
    X_fft_shifted = fftshift(X_fft)
    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N
    X_phi = np.angle(X_fft_shifted)
    for i in range(len(X_phi)):
        if np.abs(X_mag[i])<.05:
            X_phi[i] = 0
    return X_mag, X_phi, freq

MagL, PhaseL, FreqL = Fast_Fourier_Transform(Signal_L,fs)
MagR, PhaseR, FreqR = Fast_Fourier_Transform(Signal_R,fs)

# =============================================================================
# Select 3 dB cutoff frequencies and generate RLC Bandpass filter
# =============================================================================

def Passband_Selector(Filter_Info):
    
    # Bandwidth and center frequency
    Beta = Filter_Info[0]
    Omega0 = Filter_Info[1]
    
    # Cut-off frequencies
    Omega1 = f1*(2*Pi)
    Omega2 = f2*(2*Pi)
    
    # Cut-off frequency equations:
    Eq = np.zeros(2)
    Eq[0] = -0.5*Beta + np.sqrt((0.5*Beta)**2 + Omega0**2) - Omega1
    Eq[1] = 0.5*Beta + np.sqrt((0.5*Beta)**2 + Omega0**2) - Omega2
    
    return Eq

# =============================================================================
# Use built in Scipy function (fsolve) to solve system of equations:
# =============================================================================

Filter_Info = fsolve(Passband_Selector, (f1*2*Pi, f2*2*Pi))

Bandwidth = Filter_Info[0] # Rad/s
Omega0 = Filter_Info[1] # Rad/s

# =============================================================================
# Given desired bandwidth, calculate component parameters:
# =============================================================================

C = 500e-9 # Select capacitor value

R = 1/(C*Bandwidth)
L = 1/(C*Omega0**2)

# =============================================================================
# Transfer functions
# =============================================================================

delta = 100
f = np.arange(1,10e6+delta,delta)

def H(f):
    w = f*2*Pi
    num = w/(R*C)
    den = np.sqrt(w**4+((1/(R*C))**2 - (2/(L*C)))*(w**2)+(1/(L*C))**2)
    y = num/den
    return y

def phase(f):
    w = f*2*Pi
    x = ((np.pi)/2) - np.arctan((w/(R*C))/(-w**2+(1/(L*C))))
    for i in range (len(w)):
        if (1/(L*C)-w[i]**2)<0:
            x[i] -= np.pi
    return x

num = [1/(R*C),0]
den = [1, 1/(R*C), 1/(L*C)]
num2,den2 = sig.bilinear(num, den, fs)

filtered_signalL = sig.lfilter(num2,den2,Signal_L)
filtered_signalR = sig.lfilter(num2,den2,Signal_R)

MagL2, PhaseL2, FreqL2 = Fast_Fourier_Transform(filtered_signalL,fs)
MagR2, PhaseR2, FreqR2 = Fast_Fourier_Transform(filtered_signalR,fs)

Filtered_Signal = np.column_stack([filtered_signalL, filtered_signalR])

# =============================================================================
# Gain:
# =============================================================================

Output = Output_Ratio * Filtered_Signal
Output_L = Output_Ratio * filtered_signalL
Output_R = Output_Ratio * filtered_signalR

# =============================================================================
# Export filtered signals:
# =============================================================================

# =============================================================================
# df2 = pd.DataFrame(Output)
# np.savetxt(r'C:\Users\drewm\Documents\Audio_EQ_Program\OUTPUT\Output.txt',\
#            df2.values)
# =============================================================================

dfL2 = pd.DataFrame(Output_L)
np.savetxt(r'C:\Users\drewm\Documents\Audio_EQ_Program\OUTPUT\Output_L.txt',\
           dfL2.values)

dfR2 = pd.DataFrame(Output_R)
np.savetxt(r'C:\Users\drewm\Documents\Audio_EQ_Program\OUTPUT\Output_R.txt',\
           dfR2.values)

# =============================================================================
# Check results:
# =============================================================================

print("")
print("Sampling Frequency = ", fs)
print("")
print("Filter Parameters:")
print("R = ", R, "Ω")
print("L = ", L, "H")
print("C = ", C, "F")
print("")

# =============================================================================
# Plots.
# Note: Make_stem function is a workaround to avoid stem plotting with
# large sampling frequency. 
# =============================================================================

def make_stem(ax,x,y,color='tab:blue',style='solid',label='',linewidths=2.5,\
              **kwargs):
    ax.axhline(linewidth=2,color='tab:red')
    ax.vlines(x,0,y,color=color,linestyles=style,label=label,\
              linewidths=linewidths)
    ax.set_ylim([1.05*y.min(),1.05*y.max()])

    
plt.figure(figsize=(15,5))
plt.plot(t,Signal_L)
plt.grid()
plt.title('Input Signal (L)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(t,Signal_R)
plt.grid()
plt.title('Input Signal (R)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

fig , (ax1) = plt.subplots(1,1,figsize=(15,5))
plt.subplot(ax1)
make_stem(ax1,FreqL,MagL)
plt.grid(True)
plt.title('Input Signal (L)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0,600])
plt.show()

fig , (ax1) = plt.subplots(1,1,figsize=(15,5))
plt.subplot(ax1)
make_stem(ax1,FreqL,MagL)
plt.grid(True)
plt.title('Input Signal (L)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0,600])
plt.show()

plt.figure(figsize=(15,5))
plt.semilogx(f, 20*np.log10(H(f)))
plt.grid(True)
plt.title("Frequency Response of Band-Pass Filter")
plt.ylabel("Gain [dB]")
plt.xlabel("Frequency [Hz]")
plt.show

plt.figure(figsize=(15,5))
plt.plot(t,Output_L)
plt.grid(True)
plt.title('Output (L)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

plt.figure(figsize=(15,5))
plt.plot(t,Output_R)
plt.grid(True)
plt.title('Output Signal (R)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

fig , (ax1) = plt.subplots(1,1,figsize=(15,5))
plt.subplot(ax1)
make_stem(ax1,FreqL2,MagL2)
plt.grid(True)
plt.title('Output Signal (L)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0,600])
plt.show()

fig , (ax1) = plt.subplots(1,1,figsize=(15,5))
plt.subplot(ax1)
make_stem(ax1,FreqR2,MagR2)
plt.grid(True)
plt.title('Output Signal (R)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.xlim([0,600])
plt.show()