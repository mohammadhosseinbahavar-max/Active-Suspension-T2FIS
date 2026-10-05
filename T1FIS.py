import numpy as npp
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl
import time
np.random.seed(42)

_trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
ms = 500
mus = 40
ks = 10000
kt = 150000
cs = 1000
ks_nl = 0.3 * ks

e_max = 0.025
de_max = 3.5
f_max = 900

error = ctrl.Antecedent(np.linspace(-1, 1, 500), 'error')
delta_error = ctrl.Antecedent(np.linspace(-1, 1, 500), 'delta_error')


def A1(x1):
    if -1.5 <= x1 <= -1:
        return 1
    elif -1 <= x1 <= -0.5:
        return (-0.5 - x1) / (-0.5 + 1)
    else:
        return 0


def A2(x1):
    if x1 <= -1:
        return 0
    elif -1 <= x1 <= -0.5:
        return (x1 + 1) / (-0.5 + 1)
    elif -0.5 <= x1 <= 0:
        return (0 - x1) / (0 + 0.5)
    else:
        return 0


def A3(x1):
    if x1 <= -0.5:
        return 0
    elif -0.5 <= x1 <= 0:
        return (x1 + 0.5) / 0.5
    elif 0 <= x1 <= 0.5:
        return (0.5 - x1) / 0.5
    else:
        return 0


def A4(x1):
    if x1 <= 0:
        return 0
    elif 0 <= x1 <= 0.5:
        return x1 / 0.5
    elif 0.5 <= x1 <= 1:
        return (1 - x1) / 0.5
    else:
        return 0


def A5(x1):
    if x1 <= 0.5:
        return 0
    elif 0.5 <= x1 <= 1:
        return (x1 - 0.5) / 0.5
    elif 1 <= x1 <= 1.5:
        return 1
    else:
        return 0


def safe(func, universe):
    vals = np.array([func(x) for x in universe])
    return np.clip(vals, 0, 1)


labels = ['NB', 'NS', 'ZE', 'PS', 'PB']

for var in [error, delta_error]:
    var['NB'] = safe(A1, var.universe)
    var['NS'] = safe(A2, var.universe)
    var['ZE'] = safe(A3, var.universe)
    var['PS'] = safe(A4, var.universe)
    var['PB'] = safe(A5, var.universe)

mf_dict = {
    'NB': A1,
    'NS': A2,
    'ZE': A3,
    'PS': A4,
    'PB': A5
}

singleton_output = {
    'NB': -1.0,
    'NS': -0.5,
    'ZE':  0.0,
    'PS':  0.5,
    'PB':  1.0
}

reduced_labels = ['NB', 'ZE', 'PB']
reduced_rule_table = [
    ['PB', 'PS', 'ZE'],
    ['PS', 'ZE', 'NS'],
    ['ZE', 'NS', 'NB']
]
def singleton_fuzzy_inference(e_val, de_val):
    num = 0.0
    den = 0.0

    for i, e_lbl in enumerate(reduced_labels):
        for j, de_lbl in enumerate(reduced_labels):
            out_lbl = reduced_rule_table[i][j]
            mu_e = float(np.clip(mf_dict[e_lbl](e_val), 0, 1))
            mu_de = float(np.clip(mf_dict[de_lbl](de_val), 0, 1))
            w = min(mu_e, mu_de)
            c = singleton_output[out_lbl]
            num += w * c
            den += w

    return 0.0 if den < 1e-12 else num / den

def road_input(t):
    if 1.0 <= t <= 1.2:
        return 0.05 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.0 <= t <= 5.5:
        return -0.01 * np.sin(np.pi * (t - 1.0) / 0.2)
    elif 5.5 <= t <= 6.5:
        return -0.07 * np.sin(np.pi * (t - 0.7) / 0.1)
    return 0

def spring_force(dx):
    return ks * dx + ks_nl * dx ** 3


def dynamics(t, state, fa):
    xs, dxs, xus, dxus = state
    xr = road_input(t)
    ddxs = (spring_force(xus - xs) + cs * (dxus - dxs) + fa) / ms
    ddxus = (spring_force(xs - xus) + cs * (dxs - dxus) + kt * (xr - xus) - fa) / mus

    return np.array([dxs, ddxs, dxus, ddxus])

dt = 0.01
T = 20
timeax = np.arange(0, T, dt)

state = np.zeros(4)
history = []
force_hist = []
acc_hist = []

start_time = time.perf_counter()
for t in timeax:
    e = state[0] / e_max
    de = (state[1] - state[3]) / de_max

    e_noisy = np.clip(e + np.random.normal(0, 0.08),-1, 1)
    de_noisy = np.clip(de + np.random.normal(0, 0.07),-1, 1)
    u = singleton_fuzzy_inference(e_noisy, de_noisy)
    fa = u * f_max
    k1 = dynamics(t, state, fa)
    k2 = dynamics(t + dt / 2, state + dt * k1 / 2, fa)
    k3 = dynamics(t + dt / 2, state + dt * k2 / 2, fa)
    k4 = dynamics(t + dt, state + dt * k3, fa)

    state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    history.append(state.copy())
    force_hist.append(fa)
    acc_hist.append(k1[1])

history = np.array(history)
end_time = time.perf_counter()

total_simulation_time = end_time - start_time

print(f"\nSimulation Time: "f"{total_simulation_time:.2f} seconds")

def calculate_metrics_t2fs(acc_data, xs_data, force_data, t_axis):
    acc_arr = np.array(acc_data)
    xs_arr = np.array(xs_data)
    force_arr = np.array(force_data)

    rms_acc = np.sqrt(np.mean(np.square(acc_arr)))
    peak_acc = np.max(np.abs(acc_arr))
    iae_acc = _trapz(np.abs(acc_arr), t_axis)
    itae_acc = _trapz(t_axis * np.abs(acc_arr), t_axis)

    rms_disp = np.sqrt(np.mean(np.square(xs_arr)))
    peak_disp = np.max(np.abs(xs_arr))
    iae_disp = _trapz(np.abs(xs_arr), t_axis)
    itae_disp = _trapz(t_axis * np.abs(xs_arr), t_axis)

    rms_force = np.sqrt(np.mean(np.square(force_arr)))
    peak_force = np.max(np.abs(force_arr))

    return {'Acc': [rms_acc, peak_acc, iae_acc, itae_acc],'Disp': [rms_disp, peak_disp, iae_disp, itae_disp],
            'Force': [rms_force, peak_force]}

metrics_t2fs = calculate_metrics_t2fs(acc_hist, history[:, 0], force_hist, timeax)

print("T1FISS")
print("!" * 45)
print(f"Sprung Acc (RMS):    {metrics_t2fs['Acc'][0]:.4f} m/s²")
print(f"Sprung Acc (Peak):   {metrics_t2fs['Acc'][1]:.4f} m/s²")
print(f"Sprung Acc (IAE):    {metrics_t2fs['Acc'][2]:.4f}")
print(f"Sprung Acc (ITAE):   {metrics_t2fs['Acc'][3]:.4f}")
print("-" * 45)
print(f"Body Disp (RMS):     {metrics_t2fs['Disp'][0]:.4f} m")
print(f"Body Disp (Peak):    {metrics_t2fs['Disp'][1]:.4f} m")
print(f"Body Disp (IAE):     {metrics_t2fs['Disp'][2]:.4f}")
print(f"Body Disp (ITAE):    {metrics_t2fs['Disp'][3]:.4f}")
print("-" * 45)
print(f"Control Force (RMS):  {metrics_t2fs['Force'][0]:.2f} N")
print(f"Control Force (Peak): {metrics_t2fs['Force'][1]:.2f} N")
print("!" * 45)
