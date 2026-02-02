import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

params_Gi = {
    "k_on": 1e6,
    "k_off": 1e-2,
    "R_tot": 1.0,

    "k_Gi": 1.0,
    "k_Gi_off": 0.5,
    "G_tot": 1.0,

    "k_inh": 5.0,
    "k_deg": 0.2
}


def model_Gi(t, y, L, p):
    R_star, Gi_star, cAMP = y

    dRdt = p["k_on"] * L * (p["R_tot"] - R_star) - p["k_off"] * R_star

    dGidt = p["k_Gi"] * R_star * (p["G_tot"] - Gi_star) - p["k_Gi_off"] * Gi_star

    AC_star = 1 / (1 + p["k_inh"] * Gi_star)

    dcAMPdt = AC_star - p["k_deg"] * cAMP

    return [dRdt, dGidt, dcAMPdt]

L_values = np.logspace(-12, -6, 30)

cAMP_ss = []

for L in L_values:
    y0 = [0.0, 0.0, 1.0]  

    sol = solve_ivp(
        model_Gi,
        t_span=(0, 500),
        y0=y0,
        args=(L, params_Gi),
        method="LSODA"
    )

    cAMP_ss.append(sol.y[2, -1])

plt.figure()
plt.semilogx(L_values, cAMP_ss, marker='o')
plt.xlabel("Концентрация лиганда (М)")
plt.ylabel("Стационарное состояние цАМФ")
plt.title("Dose-response кривая")
plt.grid(True)
plt.show()