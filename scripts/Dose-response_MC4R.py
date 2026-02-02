import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

params = {
    "k_on": 1e6,      # 1/(M*s) связывание лиганда
    "k_off": 1e-2,    # 1/s диссоциация
    "R_tot": 1.0,     # общее число рецепторов

    "k_Gs": 1.0,      # активация Gs
    "k_Gs_off": 0.5,  # деактивация Gs
    "G_tot": 1.0, # общее кол-во Gs

    "k_AC": 1.0,      # активация аденилатциклазы
    "k_deg": 0.2    # деградация цАМФ
}

def model(t, y, L, p): # создание функции, которая будет состоять из системы диффуров
    R_star, Gs_star, cAMP = y

    # Изменение количества активного рецептора по времени
    dRdt = p["k_on"] * L * (p["R_tot"] - R_star) - p["k_off"] * R_star

    # Изменение количества активного Gs по времени
    dGsdt = p["k_Gs"] * R_star * (p["G_tot"] - Gs_star) - p["k_Gs_off"] * Gs_star

    # Аденилатциклаза (Gi = 0)
    AC_star = (p["k_AC"] * Gs_star) / (1 + p["k_AC"] * Gs_star)

    # цАМФ
    dcAMPdt = AC_star - p["k_deg"] * cAMP

    return [dRdt, dGsdt, dcAMPdt]

L_values = np.logspace(-12, -6, 30) # точки концентраций лиганда

cAMP_ss = []

for L in L_values:
    y0 = [0.0, 0.0, 0.0]   # начальные условия

    sol = solve_ivp(
        model,
        t_span=(0, 2000),
        y0=y0,
        args=(L, params),
        method="LSODA"
    )

    cAMP_ss.append(sol.y[2, -1])  # стационарный cAMP

plt.figure()
plt.semilogx(L_values, cAMP_ss, marker='o')
plt.xlabel("Концентрация лиганда (M)")
plt.ylabel("Стационарное состояние цАМФ")
plt.title("Dose-response кривая")
plt.grid(True)

plt.show()