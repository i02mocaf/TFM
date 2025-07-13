# %%
import pandas as pd
import numpy as np

pd.options.display.max_rows = 10
pd.options.display.max_columns = 30

# %%
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.preprocessing import FunctionTransformer

# %%
colonColectomia = pd.read_csv("RefinadoColon/ColectomiaBase.csv", delimiter=",")

# %%
colonColectomia

# %%
colonColectomia.describe()

# %%
# División e X, Y
colonColectomiaX = colonColectomia[["SEX", "UC_EXTENSION_DX1COLONO", "EIMS_Dx", "EIMS_TYPE", "TTO_IS", "NUM_ADVANCED", "LnEdaddias"]] # With Smoke 0.79210 Without Smoke 0.79484
colonColectomiaXnumerica = colonColectomia[["PLAQ", "VSG", "ALFA1", "ALBU", "PCR"]]
colonColectomiaY = colonColectomia[["COLECTOMY_FUP", "Colectomia"]]

# %%
# Escalar variables con la función ln

def logScaler(X, shift=1.01):
    X_log = np.round(np.log(X + shift), 6)
    return pd.DataFrame(X_log, index=X.index, columns=X.columns)

#fTransformerRobust = RobustScaler()
fTransformerLog = FunctionTransformer(func=logScaler)

colonColectomiaXnumerica = pd.DataFrame(fTransformerLog.fit_transform(colonColectomiaXnumerica), columns=colonColectomiaXnumerica.columns)

colonColectomiaX = pd.concat([colonColectomiaX, colonColectomiaXnumerica], axis=1)


colonColectomia = pd.concat([colonColectomiaX, colonColectomiaY], axis=1)

# %%
colonColectomia

# %%
colonColectomia.describe()

# %%
"""
# Regresión de Cox - SIN brotes
"""

# %%
"""
## Modelo
"""

# %%
print("CONSTRUCCION DEL MODELO:")

cph = CoxPHFitter()
kmf = KaplanMeierFitter()
variablesEtiquetas = colonColectomiaY.columns.values
variablesCoxSignificancia = colonColectomiaX.columns.values
cph.fit(colonColectomia[np.concatenate((variablesEtiquetas, variablesCoxSignificancia), axis=0)], duration_col = variablesEtiquetas[1], event_col = variablesEtiquetas[0])
significancia = cph._compute_p_values()
while max(significancia) >= 0.05:
    variableEliminar = np.where(significancia == max(significancia))[0][0]
    variablesCoxSignificancia = np.delete(variablesCoxSignificancia, variableEliminar, axis=0)
    cph.fit(colonColectomia[np.concatenate((variablesEtiquetas, variablesCoxSignificancia), axis=0)], duration_col = variablesEtiquetas[1], event_col = variablesEtiquetas[0])
    significancia = cph._compute_p_values()
kmf.fit(colonColectomiaY[variablesEtiquetas[1]], colonColectomiaY[variablesEtiquetas[0]])

# %%
print("RESUMEN MODELO DE COX")
cph.print_summary(decimals=5)
print("KAPLAN-MEIER")
fig, ax = plt.subplots(figsize=(15, 5))
kmf.plot(ax=ax)
plt.xticks(np.arange(0, 19000 + 1, step=1000), rotation=-45)
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.grid(True)

# %%
"""
# Regresión de Cox - CON brotes
"""

# %%
colonColectomia = pd.read_csv("RefinadoColon/ColectomiaBase.csv", delimiter=",")
colonColectomia["Fecha"] = pd.to_datetime(colonColectomia["Fecha"], format="%Y-%m-%d")

# %%
"""
## Creación de variables (brotes)
"""

# %%
# Creación de las nuevas columnas relacionadas con los brotes

colonColectomia.insert(loc=(colonColectomia.columns.get_loc("PLAQ") + 1), column="BrotePLAQ", value=[np.nan] * len(colonColectomia))
colonColectomia.insert(loc=(colonColectomia.columns.get_loc("BrotePLAQ") + 1), column="TiempoAccBrotePLAQ", value=[np.nan] * len(colonColectomia))

colonColectomia.insert(loc=(colonColectomia.columns.get_loc("VSG") + 1), column="BroteVSG", value=[np.nan] * len(colonColectomia))
colonColectomia.insert(loc=(colonColectomia.columns.get_loc("BroteVSG") + 1), column="TiempoAccBroteVSG", value=[np.nan] * len(colonColectomia))

colonColectomia.insert(loc=(colonColectomia.columns.get_loc("ALFA1") + 1), column="BroteALFA1", value=[np.nan] * len(colonColectomia))
colonColectomia.insert(loc=(colonColectomia.columns.get_loc("BroteALFA1") + 1), column="TiempoAccBroteALFA1", value=[np.nan] * len(colonColectomia))

colonColectomia.insert(loc=(colonColectomia.columns.get_loc("ALBU") + 1), column="BroteALBU", value=[np.nan] * len(colonColectomia))
colonColectomia.insert(loc=(colonColectomia.columns.get_loc("BroteALBU") + 1), column="TiempoAccBroteALBU", value=[np.nan] * len(colonColectomia))

colonColectomia.insert(loc=(colonColectomia.columns.get_loc("PCR") + 1), column="BrotePCR", value=[np.nan] * len(colonColectomia))
colonColectomia.insert(loc=(colonColectomia.columns.get_loc("BrotePCR") + 1), column="TiempoAccBrotePCR", value=[np.nan] * len(colonColectomia))

# %%
colonColectomia

# %%
# Umbrales inferior (130) y superior (450) de PLAQ
# Umbrales inferior (0) y superior (20) de VSG
# Umbrales inferior (58) y superior (155) de ALFA1
# Umbrales inferior (58) y superior (155) de ALBU
# Umbrales inferior (0) y superior (10) de PCR

# Calcular BroteX - Ha experimentado (o no) un brote de X
for paciente in colonColectomia.groupby("NUM"):
    #print(f"Paciente: {paciente[0]}")
    for patron in paciente[1].itertuples():
        #print(f"    Fecha: {patron.Fecha}, PLAQ: {patron.PLAQ}")
        if(patron.PLAQ < 130 or patron.PLAQ > 450):
            colonColectomia.loc[patron.Index, "BrotePLAQ"] = 1
        else:
            colonColectomia.loc[patron.Index, "BrotePLAQ"] = 0

        if(patron.VSG < 0 or patron.VSG > 20):
            colonColectomia.loc[patron.Index, "BroteVSG"] = 1
        else:
            colonColectomia.loc[patron.Index, "BroteVSG"] = 0
        
        if(patron.ALFA1 < 58 or patron.ALFA1 > 155):
            colonColectomia.loc[patron.Index, "BroteALFA1"] = 1
        else:
            colonColectomia.loc[patron.Index, "BroteALFA1"] = 0

        if(patron.ALBU < 3.2):
            colonColectomia.loc[patron.Index, "BroteALBU"] = 1
        else:
            colonColectomia.loc[patron.Index, "BroteALBU"] = 0

        if(patron.PCR < 0 or patron.PCR > 10):
            colonColectomia.loc[patron.Index, "BrotePCR"] = 1
        else:
            colonColectomia.loc[patron.Index, "BrotePCR"] = 0

# Calcular TiempoAccBroteX - Tiempo acumulado de persistencia de los brotes de X
for paciente in colonColectomia.groupby("NUM"):
    #print(f"Paciente: {paciente[0]}")
    sumTiempoPLAQ = 0
    sumTiempoVSG = 0
    sumTiempoALFA1 = 0
    sumTiempoALBU = 0
    sumTiempoPCR = 0
    for i in range(len(paciente[1]) - 1):
        #print(f"    Fecha: {patron.Fecha}, PLAQ: {patron.PLAQ}, TiempoAccBrotePLAQ: {patron.TiempoAccBrotePLAQ}")
        if(paciente[1].loc[paciente[1].index[i], "BrotePLAQ"] == 1):
            sumTiempoPLAQ = sumTiempoPLAQ + ((paciente[1].loc[paciente[1].index[i + 1], "Fecha"]) - (paciente[1].loc[paciente[1].index[i], "Fecha"])).days
        if(paciente[1].loc[paciente[1].index[i], "BroteVSG"] == 1):
            sumTiempoVSG = sumTiempoVSG + ((paciente[1].loc[paciente[1].index[i + 1], "Fecha"]) - (paciente[1].loc[paciente[1].index[i], "Fecha"])).days
        if(paciente[1].loc[paciente[1].index[i], "BroteALFA1"] == 1):
            sumTiempoALFA1 = sumTiempoALFA1 + ((paciente[1].loc[paciente[1].index[i + 1], "Fecha"]) - (paciente[1].loc[paciente[1].index[i], "Fecha"])).days
        if(paciente[1].loc[paciente[1].index[i], "BroteALBU"] == 1):
            sumTiempoALBU = sumTiempoALBU + ((paciente[1].loc[paciente[1].index[i + 1], "Fecha"]) - (paciente[1].loc[paciente[1].index[i], "Fecha"])).days
        if(paciente[1].loc[paciente[1].index[i], "BrotePCR"] == 1):
            sumTiempoPCR = sumTiempoPCR + ((paciente[1].loc[paciente[1].index[i + 1], "Fecha"]) - (paciente[1].loc[paciente[1].index[i], "Fecha"])).days

        colonColectomia.loc[paciente[1].index[i], "TiempoAccBrotePLAQ"] = sumTiempoPLAQ
        colonColectomia.loc[paciente[1].index[i], "TiempoAccBroteVSG"] = sumTiempoVSG
        colonColectomia.loc[paciente[1].index[i], "TiempoAccBroteALFA1"] = sumTiempoALFA1
        colonColectomia.loc[paciente[1].index[i], "TiempoAccBroteALBU"] = sumTiempoALBU
        colonColectomia.loc[paciente[1].index[i], "TiempoAccBrotePCR"] = sumTiempoPCR
    colonColectomia.loc[paciente[1].index[-1], "TiempoAccBrotePLAQ"] = sumTiempoPLAQ # Último registro no tiene siguiente, así que se deja el acumulado final
    colonColectomia.loc[paciente[1].index[-1], "TiempoAccBroteVSG"] = sumTiempoVSG
    colonColectomia.loc[paciente[1].index[-1], "TiempoAccBroteALFA1"] = sumTiempoALFA1
    colonColectomia.loc[paciente[1].index[-1], "TiempoAccBroteALBU"] = sumTiempoALBU
    colonColectomia.loc[paciente[1].index[-1], "TiempoAccBrotePCR"] = sumTiempoPCR

# %%
for patron in colonColectomia.itertuples():
    print(f"Paciente: {patron.NUM}, Fecha: {patron.Fecha}, ALBU: {patron.ALBU}, BrotePLAQ: {patron.BrotePLAQ}, TiempoAccBrotePLAQ: {patron.TiempoAccBrotePLAQ}")

# %%
for patron in colonColectomia.itertuples():
    print(f"Paciente: {patron.NUM}, Fecha: {patron.Fecha}, ALBU: {patron.ALBU}, BroteVSG: {patron.BroteVSG}, TiempoAccBroteVSG: {patron.TiempoAccBroteVSG}")

# %%
for patron in colonColectomia.itertuples():
    print(f"Paciente: {patron.NUM}, Fecha: {patron.Fecha}, BroteALFA1: {patron.BroteALFA1}, TiempoAccBroteALFA1: {patron.TiempoAccBroteALFA1}")

# %%
for patron in colonColectomia.itertuples():
    print(f"Paciente: {patron.NUM}, Fecha: {patron.Fecha}, ALBU: {patron.ALBU}, BroteALBU: {patron.BroteALBU}, TiempoAccBroteALBU: {patron.TiempoAccBroteALBU}")

# %%
for patron in colonColectomia.itertuples():
    print(f"Paciente: {patron.NUM}, Fecha: {patron.Fecha}, PCR: {patron.PCR}, BrotePCR: {patron.BrotePCR}, TiempoAccBrotePCR: {patron.TiempoAccBrotePCR}")

# %%
colonColectomia

# %%
colonColectomia.to_csv("RefinadoColon/colectomiaBaseALL.csv", index=False)

# %%
colonColectomia = pd.read_csv("RefinadoColon/colectomiaBaseALL.csv", delimiter=",")

# %%
colonColectomiaX = colonColectomia[["SEX", "UC_EXTENSION_DX1COLONO", "EIMS_Dx", "EIMS_TYPE", "TTO_IS", "NUM_ADVANCED", "LnEdaddias", "BrotePLAQ", "BroteVSG", "BroteALFA1", "BroteALBU", "BrotePCR"]]  # With Smoke 0.85737 Without Smoke 0.85807
colonColectomiaXnumerica = colonColectomia[["PLAQ", "VSG", "ALFA1", "ALBU", "PCR"]]
colonColectomiaXtiempo = colonColectomia[["TiempoAccBrotePLAQ", "TiempoAccBroteVSG", "TiempoAccBroteALFA1", "TiempoAccBroteALBU", "TiempoAccBrotePCR"]]
colonColectomiaY = colonColectomia[["COLECTOMY_FUP", "Colectomia"]]

# %%
# Escalar variables con la función ln

def logScaler(X, shift=1.01):
    X_log = np.round(np.log(X + shift), 6)
    return pd.DataFrame(X_log, index=X.index, columns=X.columns)

fTransformerLog = FunctionTransformer(func=logScaler)

colonColectomiaXnumerica = pd.DataFrame(fTransformerLog.fit_transform(colonColectomiaXnumerica), columns=colonColectomiaXnumerica.columns)
colonColectomiaXtiempo = pd.DataFrame(fTransformerLog.fit_transform(colonColectomiaXtiempo), columns=colonColectomiaXtiempo.columns)
colonColectomiaX = pd.concat([colonColectomiaX, colonColectomiaXtiempo, colonColectomiaXnumerica], axis=1)

colonColectomia = pd.concat([colonColectomiaX, colonColectomiaY], axis=1)

# %%
colonColectomia.to_csv("RefinadoColon/colectomiaBaseALLTransformadas.csv", index=False)

# %%
colonColectomia

# %%
colonColectomia.describe()

# %%
"""
## Modelo
"""

# %%
print("CONSTRUCCION DEL MODELO:")

cph = CoxPHFitter()
kmf = KaplanMeierFitter()
variablesEtiquetas = colonColectomiaY.columns.values
variablesCoxSignificancia = colonColectomiaX.columns.values
cph.fit(colonColectomia[np.concatenate((variablesEtiquetas, variablesCoxSignificancia), axis=0)], duration_col = variablesEtiquetas[1], event_col = variablesEtiquetas[0])
significancia = cph._compute_p_values()
while max(significancia) >= 0.05:
    variableEliminar = np.where(significancia == max(significancia))[0][0]
    variablesCoxSignificancia = np.delete(variablesCoxSignificancia, variableEliminar, axis=0)
    cph.fit(colonColectomia[np.concatenate((variablesEtiquetas, variablesCoxSignificancia), axis=0)], duration_col = variablesEtiquetas[1], event_col = variablesEtiquetas[0])
    significancia = cph._compute_p_values()
kmf.fit(colonColectomiaY[variablesEtiquetas[1]], colonColectomiaY[variablesEtiquetas[0]])

# %%
print("RESUMEN MODELO DE COX")
cph.print_summary(decimals=5)
print("KAPLAN-MEIER")
fig, ax = plt.subplots(figsize=(15, 5))
kmf.plot(ax=ax)
plt.xticks(np.arange(0, 19000 + 1, step=1000), rotation=-45)
plt.xlabel("Time")
plt.ylabel("Survival probability")
plt.grid(True)