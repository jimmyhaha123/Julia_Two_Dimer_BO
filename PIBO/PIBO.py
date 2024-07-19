import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import fsolve
from scipy.linalg import eig
from autograd import jacobian
import autograd.numpy as anp

replication = 100
def augmented_fft(x):
    # Replicate the signal 100 times
    x_extended = np.tile(x, replication)
    
    # Perform FFT on the extended signal
    X = np.fft.fft(x_extended)
    
    # Return the full FFT result
    return X


def highest_peak_deviation(freqs, vals, num=40):
    if len(vals) < num + 1:
        print("Not enough peaks.")
        return num - 1
    elif vals[0] < 10**(-3):  # First peak too small
        print("First peak too small. No oscillations.")
        return num - 1
    else:
        sub_peaks = vals[1:num+1]
        sub_peaks = sub_peaks / sub_peaks[0]  # Normalization with respect to first peak
        cost = np.sum(np.abs(sub_peaks - 1))
        print(f"Successful. Loss: {cost}")
        return cost


def single_dimer_sys(t, u, p):
    n11, n10, n20, w2 = p

    w1 = w2

    def N1(a1):
        return n11 * (abs(a1)) ** 2 + n10
    
    def N2(a2):
        return n20
    
    a1, a2 = u

    du = np.zeros(2, dtype=complex)
    du[0] = 1j*w1*a1 + N1(a1)*a1 + 1j*a2
    du[1] = 1j*w2*a2 + N2(a2)*a2 + 1j*a1

    return du


def solve_and_interpolate_sys(u0, p, t_span):

    rtol = 1e-8
    atol = 1e-10

    # Solve the differential equations
    solution = solve_ivp(single_dimer_sys, t_span, u0, args=(p,), method='RK45', rtol=rtol, atol=atol)

    # Calculate the mean of the time steps
    mean_dt = np.mean(np.diff(solution.t))

    # Create an array of new time points with equal time steps within the bounds of the original time points
    t_new = np.arange(t_span[0], solution.t[-1], mean_dt)

    # Interpolate the solution to these new time points
    a1_interp = interp1d(solution.t, solution.y[0], kind='cubic', fill_value="extrapolate")
    a2_interp = interp1d(solution.t, solution.y[1], kind='cubic', fill_value="extrapolate")

    u_new = np.vstack((a1_interp(t_new), a2_interp(t_new)))

    return t_new, u_new


def isclose(a, b, epsilon):
    return np.all(np.abs(np.array(a) - np.array(b)) < epsilon)


def jacobian_eig(p):
    n11, n10, n20, w2 = p
    
    # Define the time derivatives
    def time_derivatives(vars):
        I1, I2, theta = vars
        dI1dt = 2*n11*(I1**2) + 2*n10*I1 + 2*anp.sqrt(I1*I2)*anp.sin(theta)
        dI2dt = 2*n20*I2 - 2*anp.sqrt(I1*I2)*anp.sin(theta)
        dthetadt = (anp.sqrt(I2/I1) - anp.sqrt(I1/I2)) * anp.cos(theta)
        return anp.array([dI1dt, dI2dt, dthetadt])
    
    # Analytical fixed points
    I1 = I2 = -(n10 + n20) / n11
    theta = np.arcsin(n20)
    
    # Ensure I1 and I2 are positive and real
    if I1 <= 0 or I2 <= 0 or not np.isreal(theta):
        return []

    # Convert fixed points to numpy arrays explicitly
    fixed_point = np.array([I1, I2, theta], dtype=float)  # Explicit conversion to float
    fixed_points = [fixed_point]
    jacobian_eigenvalues = []
    
    # Compute the Jacobian matrix at the fixed point using autograd
    J_func = jacobian(time_derivatives)
    J = J_func(fixed_point)
    
    # Calculate the eigenvalues of the Jacobian matrix
    eigenvalues = eig(J)[0]
    jacobian_eigenvalues.append(eigenvalues)
    
    return list(zip(fixed_points, jacobian_eigenvalues))


def loss(p):
    u0 = [0.5 + 0j, 0.5 + 0j]
    t_span = [0, 50000]

    # n11 = -0.5
    # n10 = 2.88
    # n20 = 0.2
    # w2 = 1
    # p = [n11, n10, n20, w2]

    t_new, u_new = solve_and_interpolate_sys(u0, p, t_span)
    start_index = int(0.7 * len(t_new))
    t_new = t_new[start_index:]
    u_new = u_new[:, start_index:]

    repetition, repeat_index = repetition_check(u_new, t_new, dimensionality=2)

    if repetition:
        t_new = t_new[repeat_index:]
        u_new = u_new[:, repeat_index:]
    else:
        return 39
    
    
    mean_time_step = t_new[1] - t_new[0]
    num_res = len(u_new)
    x = []
    transforms = []
    peak_frequencies = []
    sorted_vals = []
    losses = []
    for i in range(num_res):
        x.append(u_new[i].real)
        temp = augmented_fft(u_new[i].real)
        temp = temp / (len(x[-1]) * replication)  # Normalize by the length of the extended signal
        transforms.append(temp)
        peaks, vals, freqs = extract_peaks(mean_time_step, x[-1], transforms[-1])
        peak_frequencies.append(peaks)
        sorted_vals.append(vals)
        losses.append(highest_peak_deviation(peak_frequencies[-1], sorted_vals[-1]))
    
    return losses, freqs, transforms



def extract_peaks(mean_time_step, x_sol, transform):
    N = len(x_sol) * replication
    dt = mean_time_step
    freqs = np.fft.fftfreq(N, d=dt)
    half_N = int(np.ceil(N / 2))
    freqs = freqs[:half_N]
    transform = transform[:half_N]
    mag_transform = np.abs(transform) ** 2
    
    peaks, _ = find_peaks(mag_transform)
    vals = mag_transform[peaks]
    sorted_indices = np.argsort(vals)[::-1]
    sorted_pks = peaks[sorted_indices]
    sorted_vals = vals[sorted_indices]  # Magnitude of peaks
    peak_frequencies = freqs[sorted_pks]  # Frequency of peaks

    return peak_frequencies, sorted_vals, freqs


def repetition_check(x, t_interp, dimensionality=8):
    first_data_point = [x[j][0] for j in range(dimensionality)]
    epsilon = 0.01
    repeating_times = []
    repeating_indices = []

    for i in range(len(t_interp)):
        current_point = [x[j][i] for j in range(dimensionality)]
        if isclose(current_point, first_data_point, epsilon):
            repeating_times.append(t_interp[i])
            repeating_indices.append(i)

    repeat_index = -1
    repetition = False

    if repeating_indices and repeating_indices[-1] > 5000:
        repeat_index = repeating_indices[-1]
        print("Repetition found.")
        repetition = True

    # Checking for valid repetition
    if repeat_index == -1:
        repeat_index = len(x[0])  # Corrected to use length of the first vector in x
        print("No repeating index.")
        repetition = False

    return repetition, repeat_index

p = [-0.5, 2.88, 0.2, 1]
fixed_points_with_eigenvalues = jacobian_eig(p)
fp, eigs = fixed_points_with_eigenvalues[0]
print("Max eigenvalue: " + str(max(eigs.real)))

l, freqs, transforms = loss(p)
transform = transforms[0]
transform = abs(transform) ** 2
transform = transform[:int(np.ceil(len(transform) / 2))]
plt.plot(freqs, transform)
plt.yscale('log')
plt.show()



n20_range = np.linspace(0.1, 0.35, 20)
losses = []
eiganvalues = []
counter = 0

# for n20 in n20_range:
#     counter += 1
#     print("Iteration: " + str(counter))
#     p[2] = n20
#     l, _, _ = loss(p)
#     losses.append(l[0])
#     fixed_points_with_eigenvalues = jacobian_eig(p)
#     fp, eigs = fixed_points_with_eigenvalues[0]
#     print(max(eigs.real))
#     eiganvalues.append(max(eigs.real))







 

    