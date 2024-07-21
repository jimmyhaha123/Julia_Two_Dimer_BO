import numpy as np
import matplotlib.pyplot as plt

# Parameters
duration = 1.0  # seconds
sample_rate = 1000  # Hz
frequency = 10  # Hz

# Generate time array
t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

# Generate sine signal
signal = np.sin(2 * np.pi * frequency * t)

# Perform FFT
fft_result = np.fft.fft(signal)
fft_freq = np.fft.fftfreq(len(t), 1 / sample_rate)

# Calculate power spectrum
power_spectrum = np.abs(fft_result) ** 2

# Plot the results
plt.figure(figsize=(12, 8))

# Plot original signal
plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Sine Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot power spectrum
plt.subplot(2, 1, 2)
plt.plot(fft_freq[:len(fft_freq)//2], power_spectrum[:len(power_spectrum)//2])
plt.title('Power Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.xlim(0, 50)  # Limit x-axis to focus on relevant frequencies

plt.tight_layout()
plt.show()