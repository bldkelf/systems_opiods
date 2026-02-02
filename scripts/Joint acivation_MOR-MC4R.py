import numpy as np

def model_joint(t, y, Ls, Li, p):
    R_s, R_i, G_s, G_i, AC, cAMP = y

    dR_s = p["k_on_s"] * Ls * (p["R_s_tot"] - R_s) - p["k_off_s"] * R_s
    dR_i = p["k_on_i"] * Li * (p["R_i_tot"] - R_i) - p["k_off_i"] * R_i

    dG_s = p["k_Gs"] * R_s * (1 - G_s) - p["k_Gs_deact"] * G_s
    dG_i = p["k_Gi"] * R_i * (1 - G_i) - p["k_Gi_deact"] * G_i

    dAC = (
        p["k_basal"] * (1 - AC)
        + p["k_act"] * G_s * (1 - AC)
        - p["k_inh"] * G_i * AC
    )

    dcAMP = p["k_cAMP"] * AC - p["k_deg"] * cAMP

    return [dR_s, dR_i, dG_s, dG_i, dAC, dcAMP]

import numpy as np

params = {
    "k_on_s": 1e6,
    "k_off_s": 0.1,
    "k_on_i": 1e6,
    "k_off_i": 0.1,

    "R_s_tot": 1.0,
    "R_i_tot": 1.0,

    "k_Gs": 1.0,
    "k_Gs_deact": 0.2,
    "k_Gi": 1.0,
    "k_Gi_deact": 0.2,

    "k_basal": 0.2,
    "k_act": 1.0,
    "k_inh": 1.2,

    "k_cAMP": 1.0,
    "k_deg": 0.3
}

from scipy.integrate import solve_ivp

Ls_range = np.logspace(-12, -6, 20)  
Li_range = np.logspace(-12, -6, 20)  

cAMP_ss = np.zeros((len(Li_range), len(Ls_range)))

for i, Li in enumerate(Li_range):
    for j, Ls in enumerate(Ls_range):

        y0 = [
            0.0,  
            0.0,  
            0.0,  
            0.0, 
            0.2,  
            1.0   
        ]

        sol = solve_ivp(
            model_joint,
            t_span=(0, 500),
            y0=y0,
            args=(Ls, Li, params),
            method="LSODA"
        )

        cAMP_ss[i, j] = sol.y[5, -1]

import matplotlib.pyplot as plt

plt.figure(figsize=(7, 6))
plt.contourf(
    np.log10(Ls_range),
    np.log10(Li_range),
    cAMP_ss,
    levels=20
)
plt.colorbar(label="Стационар цАМФ")
plt.xlabel("log10 [α-МСГ]")
plt.ylabel("log10 [Морфин]")
plt.title("Совместная Gs / Gi активация: баланс цАМФ")
plt.show()

