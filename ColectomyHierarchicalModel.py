# %%
import pandas as pd
import numpy as np

pd.options.display.max_rows = 10
pd.options.display.max_columns = 30

# %%
from mord import LogisticAT

from orca_python import metrics
from orca_python.classifiers.REDSVM import REDSVM
from orca_python.classifiers import SVOREX

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, balanced_accuracy_score, mean_absolute_error, mean_squared_error, accuracy_score, recall_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, RandomizedSearchCV, RandomizedSearchCV
from sklearn.preprocessing import FunctionTransformer

# %%
colonColectomia = pd.read_csv("/mnt/datasets/SurvAnalysis/BeatrizColon/RefinadoColon/colectomiaBaseALL.csv", delimiter=",")

# %%
colonColectomiaX = colonColectomia[["NUM", "SEX", "UC_EXTENSION_DX1COLONO", "EIMS_Dx", "EIMS_TYPE", "Smoke", "TTO_IS", "NUM_ADVANCED", "LnEdaddias", "BrotePLAQ", "BroteVSG", "BroteALFA1", "BroteALBU", "BrotePCR"]]
colonColectomiaXnumerica = colonColectomia[["PLAQ", "VSG", "ALFA1", "ALBU", "PCR"]]
colonColectomiaXtiempo = colonColectomia[["TiempoAccBrotePLAQ", "TiempoAccBroteVSG", "TiempoAccBroteALFA1", "TiempoAccBroteALBU", "TiempoAccBrotePCR"]]
colonColectomiaY = colonColectomia[["COLECTOMY_FUP", "Colectomia"]]

# %%
colonColectomia

# %%
"""
# Procesamiento
"""

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
# Discretización de la variable dependiente tiempo

colonColectomiaOrdinal = colonColectomia.copy()
colonColectomiaOrdinal.loc[colonColectomiaOrdinal["Colectomia"] >= 7200, "Colectomia"] = -4
colonColectomiaOrdinal.loc[colonColectomiaOrdinal["Colectomia"] >= 4320, "Colectomia"] = -3
colonColectomiaOrdinal.loc[colonColectomiaOrdinal["Colectomia"] >= 1800, "Colectomia"] = -2
colonColectomiaOrdinal.loc[colonColectomiaOrdinal["Colectomia"] >= 0, "Colectomia"] = -1
colonColectomiaOrdinal = colonColectomiaOrdinal.replace({-1 : 1, -2 : 2, -3 : 3, -4 : 4})

# %%
colonColectomiaOrdinal

maeTrainLR = []
ccrTrainLR = []
sensitivitiesTrainLR = []
f1TrainLR = []
maeTestLR = []
ccrTestLR = []
sensitivitiesTestLR = []
f1TestLR = []

maeTrainSVM = []
ccrTrainSVM = []
sensitivitiesTrainSVM = []
f1TrainSVM = []
maeTestSVM = []
ccrTestSVM = []
sensitivitiesTestSVM = []
f1TestSVM = []

maeTrainRF = []
ccrTrainRF = []
sensitivitiesTrainRF = []
f1TrainRF = []
maeTestRF = []
ccrTestRF = []
sensitivitiesTestRF = []
f1TestRF = []

for i in range(10):
    print(f"Ejecución nº {i + 1} (Clasificador Binario)")
    # %%
    """
    # Clasificador jerárquico
    """

    # %%
    """
    ## 1º - Predicción del evento (Clasificador Binario)
    """

    # %%
    # Agrupar por pacientes
    pacientes = []  # Lista para guardar (NUM, grupo) de cada paciente
    etiquetasEvent = []  # Lista con 0 o 1 indicando si el paciente tuvo colectomía
    estratos = []
    pacienteIdx = 0
    # Construir X, Y y etiquetas por paciente
    for num, grupo in colonColectomia.groupby("NUM"):
        X = []
        Y = []
        estrato = []
        for paciente in grupo.itertuples():
            X.append(np.array([
                paciente.SEX, paciente.UC_EXTENSION_DX1COLONO, paciente.EIMS_Dx, paciente.EIMS_TYPE,
                paciente.Smoke, paciente.TTO_IS, paciente.NUM_ADVANCED, paciente.LnEdaddias,
                paciente.PLAQ, paciente.VSG, paciente.ALFA1, paciente.ALBU, paciente.PCR,
                paciente.BrotePLAQ, paciente.BroteVSG, paciente.BroteALFA1, paciente.BroteALBU, paciente.BrotePCR,
                paciente.TiempoAccBrotePLAQ, paciente.TiempoAccBroteVSG, paciente.TiempoAccBroteALFA1,
                paciente.TiempoAccBroteALBU, paciente.TiempoAccBrotePCR
            ]))
            Y.append(np.array(paciente.COLECTOMY_FUP))
            estrato.append(pacienteIdx)
            pacienteIdx = pacienteIdx + 1
        pacientes.append(X)
        etiquetasEvent.append(Y)  # Almacenar evento 0 / 1 (mismo evento para las X analíticas de un paciente)
        estratos.append(estrato)


    # Estratificación por paciente (NO puede haber un mismo paciente en train y test)
    trainIdx, testIdx = train_test_split(estratos, test_size=0.2, stratify=[e[0] for e in etiquetasEvent])

    trainIdx = [x for xs in trainIdx for x in xs]
    testIdx = [x for xs in testIdx for x in xs]
    pacientes = [x for xs in pacientes for x in xs]
    etiquetasEvent = [x for xs in etiquetasEvent for x in xs]

    # Separar en conjuntos train y test
    XtrainJerarquico1 = [pacientes[i] for i in trainIdx]
    YtrainJerarquico1 = [etiquetasEvent[i] for i in trainIdx]
    XtestJerarquico1 = [pacientes[i] for i in testIdx]
    YtestJerarquico1 = [etiquetasEvent[i] for i in testIdx]

    # %%
    def binary(y_true, y_pred):
        bAcc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return (0.5 * bAcc + 0.5 * f1)

    # %%
    # CrossValidation  para LogisticRegression
    param_distributions_LogisticRegression = {
        "tol": np.logspace(-3, 0, 4),
        "C": np.logspace(-3, 3, 7),
        "solver": ["liblinear"],
        "max_iter": [100, 200],
    }

    # Scorer personalizado
    scorer = make_scorer(binary)

    # RandomizedSearchCV
    LogisticRegressionModel = RandomizedSearchCV(
        estimator=LogisticRegression(class_weight="balanced"),
        param_distributions=param_distributions_LogisticRegression,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        verbose=0,
        scoring=scorer,
        n_jobs=-1,
        return_train_score=True,
        refit=True
    )

    # %%
    # CrossValidation  para SVM
    param_distributions_SVM = {
        "kernel": ["linear", "rbf", "sigmoid"],
        "gamma": np.logspace(-3, 3, 7),
        "C": np.logspace(-3, 3, 7),
        "coef0": np.linspace(-150, 150, 6)
    }

    # Scorer personalizado
    scorer = make_scorer(binary)

    # RandomizedSearchCV
    SVMModel = RandomizedSearchCV(
        estimator=svm.SVC(class_weight="balanced"),
        param_distributions=param_distributions_SVM,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        verbose=0,
        scoring=scorer,
        n_jobs=-1,
        return_train_score=True,
        refit=True
    )

    # %%
    # CrossValidation  para RandomForest
    param_distributions_RandomForest = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 3],
    }

    # Scorer personalizado
    scorer = make_scorer(binary)

    # RandomizedSearchCV
    RandomForestModel = RandomizedSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced"),
        param_distributions=param_distributions_RandomForest,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        scoring=scorer,
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
        refit=True
    )

    # %%
    # Submuestreo aleatorio
    from imblearn.under_sampling import RandomUnderSampler  

    underSampling = RandomUnderSampler(sampling_strategy={0:500}) #sampling_strategy={0:X, 1:X}
    XtrainUnder, YtrainUnder = underSampling.fit_resample(XtrainJerarquico1, YtrainJerarquico1)

    #XtrainUnder, YtrainUnder = XtrainJerarquico1, YtrainJerarquico1
    from sklearn.utils import shuffle
    XtrainUnder, YtrainUnder = shuffle(XtrainUnder, YtrainUnder)


    # %%
    # Método determinista

    for i in range(10):
        print(f"  Ejecución nº {i + 1} LR")
        # Se entrena el modelo LogisticRegression con los datos de entrenamiento
        #LogisticRegressionModel = LogisticRegression(class_weight="balanced").fit(XtrainUnder, YtrainUnder)
        LogisticRegressionModel.fit(XtrainUnder, YtrainUnder)

        # Se predicen las etiquetas de train
        LogisticRegressionPredictionsTrain = LogisticRegressionModel.predict(XtrainUnder)

        # Se predicen las etiquetas de test
        LogisticRegressionPredictions = LogisticRegressionModel.predict(XtestJerarquico1)

        # Resultados train
        maeTrainLR.append(mean_absolute_error(YtrainUnder, LogisticRegressionPredictionsTrain))
        ccrTrainLR.append(accuracy_score(YtrainUnder, LogisticRegressionPredictionsTrain))
        sensitivitiesTrainLR.append(recall_score(YtrainUnder, LogisticRegressionPredictionsTrain, average=None))
        f1TrainLR.append(f1_score(YtrainUnder, LogisticRegressionPredictionsTrain, average=None))
        #print("Resultados obtenidos en train")
        #print("  MAE = {}".format(maeTrainLR[i]))
        #print("  CCR = {}".format(ccrTrainLR[i]))
        #print(f"  Sensibilidades =  {sensitivitiesTrainLR[i]}")
        #print(f"  F1 =  {f1TrainLR[i]}")
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtrainUnder, LogisticRegressionPredictionsTrain)}")

        # Resultados test
        maeTestLR.append(mean_absolute_error(YtestJerarquico1, LogisticRegressionPredictions))
        ccrTestLR.append(accuracy_score(YtestJerarquico1, LogisticRegressionPredictions))
        sensitivitiesTestLR.append(recall_score(YtestJerarquico1, LogisticRegressionPredictions, average=None))
        f1TestLR.append(f1_score(YtestJerarquico1, LogisticRegressionPredictions, average=None))
        #print("Resultados obtenidos en test")
        #print("  MAE = {}".format(maeTestLR[i]))
        #print("  CCR = {}".format(ccrTestLR[i]))
        #print(f"  Sensibilidades = {sensitivitiesTestLR[i]}")
        #print(f"  F1 =  {f1TestLR[i]}")
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtestJerarquico1, LogisticRegressionPredictions)}")
        print(f"    Mejor LogisticRegressionModel: {LogisticRegressionModel.best_params_}")

    # %%
    # Método determinista

    for i in range(10):
        print(f"  Ejecución nº {i + 1} SVM")
        # Se entrena el modelo SVM con los datos de entrenamiento
        #SVMModel = svm.SVC(class_weight="balanced").fit(XtrainUnder, YtrainUnder)
        SVMModel.fit(XtrainUnder, YtrainUnder)

        # Se predicen las etiquetas de train
        SVMPredictionsTrain = SVMModel.predict(XtrainUnder)

        # Se predicen las etiquetas de test
        SVMPredictions = SVMModel.predict(XtestJerarquico1)

        # Resultados train
        maeTrainSVM.append(mean_absolute_error(YtrainUnder, SVMPredictionsTrain))
        ccrTrainSVM.append(accuracy_score(YtrainUnder, SVMPredictionsTrain))
        sensitivitiesTrainSVM.append(recall_score(YtrainUnder, SVMPredictionsTrain, average=None))
        f1TrainSVM.append(f1_score(YtrainUnder, SVMPredictionsTrain, average=None))
        #print("Resultados obtenidos en train")
        #print("  MAE = {}".format(maeTrainSVM[i]))
        #print("  CCR = {}".format(ccrTrainSVM[i]))
        #print(f"  Sensibilidades = {sensitivitiesTrainSVM[i]}")
        #print(f"  F1 =  {f1TrainSVM[i]}")
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtrainUnder, SVMPredictionsTrain)}")

        # Resultados test
        maeTestSVM.append(mean_absolute_error(YtestJerarquico1, SVMPredictions))
        ccrTestSVM.append(accuracy_score(YtestJerarquico1, SVMPredictions))
        sensitivitiesTestSVM.append(recall_score(YtestJerarquico1, SVMPredictions, average=None))
        f1TestSVM.append(f1_score(YtestJerarquico1, SVMPredictions, average=None))
        #print("Resultados obtenidos en test")
        #print("  MAE = {}".format(maeTestSVM[i]))
        #print("  CCR = {}".format(ccrTestSVM[i]))
        #print(f"  Sensibilidades = {sensitivitiesTestSVM[i]}")
        #print(f"  F1 =  {f1TestSVM[i]}")
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtestJerarquico1, SVMPredictions)}")
        print(f"    Mejor SVMModel: {SVMModel.best_params_}")

    # %%
    # Método estocástico

    for i in range(10):
        print(f"  Ejecución nº {i + 1} RF")
        # Se entrena el modelo RandomForest con los datos de entrenamiento
        #RandomForestModel = RandomForestClassifier(class_weight="balanced").fit(XtrainUnder, YtrainUnder)
        RandomForestModel.fit(XtrainUnder, YtrainUnder)

        # Se predicen las etiquetas de train
        RandomForestPredictionsTrain = RandomForestModel.predict(XtrainUnder)

        # Se predicen las etiquetas de test
        RandomForestPredictions = RandomForestModel.predict(XtestJerarquico1)

        # Resultados train
        maeTrainRF.append(mean_absolute_error(YtrainUnder, RandomForestPredictionsTrain))
        ccrTrainRF.append(accuracy_score(YtrainUnder, RandomForestPredictionsTrain))
        sensitivitiesTrainRF.append(recall_score(YtrainUnder, RandomForestPredictionsTrain, average=None))
        f1TrainRF.append(f1_score(YtrainUnder, RandomForestPredictionsTrain, average=None))
        #print("  Resultados obtenidos en train")
        #print("    MAE = {}".format(maeTrainRF[i]))
        #print("    CCR = {}".format(ccrTrainRF[i]))
        #print(f"    Sensibilidades = {sensitivitiesTrainRF[i]}")
        #print(f"    F1 =  {f1TrainRF[i]}")
        #print(f"    Confusion Matrix = \n{confusion_matrix(YtrainUnder, RandomForestPredictionsTrain)}")

        # Resultados test
        maeTestRF.append(mean_absolute_error(YtestJerarquico1, RandomForestPredictions))
        ccrTestRF.append(accuracy_score(YtestJerarquico1, RandomForestPredictions))
        sensitivitiesTestRF.append(recall_score(YtestJerarquico1, RandomForestPredictions, average=None))
        f1TestRF.append(f1_score(YtestJerarquico1, RandomForestPredictions, average=None))
        #print("  Resultados obtenidos en test")
        #print("    MAE = {}".format(maeTestRF[i]))
        #print("    CCR = {}".format(ccrTestRF[i]))
        #print(f"    Sensibilidades = {sensitivitiesTestRF[i]}")
        #print(f"    F1 =  {f1TestRF[i]}")
        #print(f"    Confusion Matrix = \n{confusion_matrix(YtestJerarquico1, RandomForestPredictions)}")
        print(f"    Mejor RandomForestModel: {RandomForestModel.best_params_}")

    print("* * * * * * * * * * * * * *")

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train LR - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainLR), np.std(maeTrainLR)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainLR), np.std(ccrTrainLR)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTrainLR]), np.mean([v[1] for v in sensitivitiesTrainLR])}, std = {np.std([v[0] for v in sensitivitiesTrainLR]), np.std([v[1] for v in sensitivitiesTrainLR])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TrainLR]), np.mean([v[1] for v in f1TrainLR])}, std = {np.std([v[0] for v in f1TrainLR]), np.std([v[1] for v in f1TrainLR])}")

print("Resultados obtenidos en test LR - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestLR), np.std(maeTestLR)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestLR), np.std(ccrTestLR)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTestLR]), np.mean([v[1] for v in sensitivitiesTestLR])}, std = {np.std([v[0] for v in sensitivitiesTestLR]), np.std([v[1] for v in sensitivitiesTestLR])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TestLR]), np.mean([v[1] for v in f1TestLR])}, std = {np.std([v[0] for v in f1TestLR]), np.std([v[1] for v in f1TestLR])}")

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train SVM - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainSVM), np.std(maeTrainSVM)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainSVM), np.std(ccrTrainSVM)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTrainSVM]), np.mean([v[1] for v in sensitivitiesTrainSVM])}, std = {np.std([v[0] for v in sensitivitiesTrainSVM]), np.std([v[1] for v in sensitivitiesTrainSVM])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TrainSVM]), np.mean([v[1] for v in f1TrainSVM])}, std = {np.std([v[0] for v in f1TrainSVM]), np.std([v[1] for v in f1TrainSVM])}")

print("Resultados obtenidos en test SVM - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestSVM), np.std(maeTestSVM)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestSVM), np.std(ccrTestSVM)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTestSVM]), np.mean([v[1] for v in sensitivitiesTestSVM])}, std = {np.std([v[0] for v in sensitivitiesTestSVM]), np.std([v[1] for v in sensitivitiesTestSVM])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TestSVM]), np.mean([v[1] for v in f1TestSVM])}, std = {np.std([v[0] for v in f1TestSVM]), np.std([v[1] for v in f1TestSVM])}")

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train RF - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainRF), np.std(maeTrainRF)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainRF), np.std(ccrTrainRF)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTrainRF]), np.mean([v[1] for v in sensitivitiesTrainRF])}, std = {np.std([v[0] for v in sensitivitiesTrainRF]), np.std([v[1] for v in sensitivitiesTrainRF])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TrainRF]), np.mean([v[1] for v in f1TrainRF])}, std = {np.std([v[0] for v in f1TrainRF]), np.std([v[1] for v in f1TrainRF])}")

print("Resultados obtenidos en test RF - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestRF), np.std(maeTestRF)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestRF), np.std(ccrTestRF)))
print(f"  Sensibilidades = {np.mean([v[0] for v in sensitivitiesTestRF]), np.mean([v[1] for v in sensitivitiesTestRF])}, std = {np.std([v[0] for v in sensitivitiesTestRF]), np.std([v[1] for v in sensitivitiesTestRF])}")
print(f"  F1 =  {np.mean([v[0] for v in f1TestRF]), np.mean([v[1] for v in f1TestRF])}, std = {np.std([v[0] for v in f1TestRF]), np.std([v[1] for v in f1TestRF])}")

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

# %%
# Se selecciona el mejor método utilizando la sensibilidad de clases
rankingF1Score = [((np.mean([v[0] for v in f1TestLR]) + np.mean([v[1] for v in f1TestLR])) / 2), ((np.mean([v[0] for v in f1TestSVM]) + np.mean([v[1] for v in f1TestSVM])) / 2), ((np.mean([v[0] for v in f1TestRF]) + np.mean([v[1] for v in f1TestRF]))/ 2)]
mejorMetodo = rankingF1Score.index(max(rankingF1Score))

predictionsEventos = []
# Se predicen los eventos
if(mejorMetodo == 0):
    print("El mejor método es Logistic Regression")
    predictionsEventos = LogisticRegressionModel.predict(pacientes)
elif(mejorMetodo == 1):
    print("El mejor método es SVM")
    predictionsEventos = SVMModel.predict(pacientes)
elif(mejorMetodo == 2):
    print("El mejor método es Random Forest")
    predictionsEventos = RandomForestModel.predict(pacientes)

print(f"Predición de {len(predictionsEventos[predictionsEventos == 1])} eventos como eventos de ocurrencia")

maeTrainREDSVM = []
ccrTrainREDSVM = []
amaeTrainREDSVM = []
wkappaTrainREDSVM = []
maeTestREDSVM = []
ccrTestREDSVM = []
amaeTestREDSVM = []
wkappaTestREDSVM = []

maeTrainSVOREX = []
ccrTrainSVOREX = []
amaeTrainSVOREX = []
wkappaTrainSVOREX = []
maeTestSVOREX = []
ccrTestSVOREX = []
amaeTestSVOREX = []
wkappaTestSVOREX = []

maeTrainLAT = []
ccrTrainLAT = []
amaeTrainLAT = []
wkappaTrainLAT = []
maeTestLAT = []
ccrTestLAT = []
amaeTestLAT = []
wkappaTestLAT = []
for i in range(10):
    print(f"Ejecución nº {i + 1} (Clasificador Ordinal)")
    # %%
    """
    ## 2º - Predicción del tiempo (Clasificador Ordinal)
    """

    # %%
    # Agrupar por pacientes
    pacientes = []
    etiquetasTime = []
    estratos = []
    pacienteIdx = 0
    EvIdx = 0  # Índice para avanzar sobre lista de eventos

    for _, grupo in colonColectomia.groupby("NUM"):
        X = []
        Y = []
        estrato = []
        for paciente in grupo.itertuples():
            if predictionsEventos[EvIdx] == 1:
                X.append(np.array([
                    paciente.SEX, paciente.UC_EXTENSION_DX1COLONO, paciente.EIMS_Dx, paciente.EIMS_TYPE,
                    paciente.Smoke, paciente.TTO_IS, paciente.NUM_ADVANCED, paciente.LnEdaddias,
                    paciente.PLAQ, paciente.VSG, paciente.ALFA1, paciente.ALBU, paciente.PCR,
                    paciente.BrotePLAQ, paciente.BroteVSG, paciente.BroteALFA1, paciente.BroteALBU, paciente.BrotePCR,
                    paciente.TiempoAccBrotePLAQ, paciente.TiempoAccBroteVSG, paciente.TiempoAccBroteALFA1,
                    paciente.TiempoAccBroteALBU, paciente.TiempoAccBrotePCR
                ]))

                if paciente.Colectomia >= 7200:
                    Y.append(4)
                elif paciente.Colectomia >= 4320:
                    Y.append(3)
                elif paciente.Colectomia >= 1800:
                    Y.append(2)
                else:
                    Y.append(1)
                estrato.append(pacienteIdx)
                pacienteIdx = pacienteIdx + 1
            EvIdx += 1

        if X:  # Solo si el paciente tiene muestras válidas tras el filtro
            pacientes.append(X)
            etiquetasTime.append(Y)  # Almacenar fecha 1 / 2 / 3 / 4 (misma fecha para las X analíticas de un paciente)
            estratos.append(estrato)

    # Estratificación por paciente (NO puede haber un mismo paciente en train y test)
    trainIdx, testIdx = train_test_split(estratos, test_size=0.2, stratify=[e[0] for e in etiquetasTime])

    trainIdx = [x for xs in trainIdx for x in xs]
    testIdx = [x for xs in testIdx for x in xs]
    pacientes = [x for xs in pacientes for x in xs]
    etiquetasTime = [x for xs in etiquetasTime for x in xs]

    # Separar en train/test
    XtrainJerarquico2 = [pacientes[i] for i in trainIdx]
    YtrainJerarquico2 = [etiquetasTime[i] for i in trainIdx]
    XtestJerarquico2 = [pacientes[i] for i in testIdx]
    YtestJerarquico2 = [etiquetasTime[i] for i in testIdx]

    # %%
    def ordinal(y_true, y_pred):
        bAcc = balanced_accuracy_score(y_true, y_pred)
        wkappa = metrics.wkappa(y_true, y_pred)
        return (0.5 * bAcc + 0.5 * wkappa)
    
    # %%
    # CrossValidation  para REDSVM
    param_distributions_REDSVM = {
        "kernel": [2, 5],
        "gamma": np.logspace(-3, 3, 7),
        "C": np.logspace(-3, 3, 7),
        "coef0": np.linspace(-150, 150, 6)
    }

    # Scorer personalizado
    scorer = make_scorer(ordinal)

    # RandomizedSearchCV
    REDSVMModel = RandomizedSearchCV(
        estimator=REDSVM(),
        param_distributions=param_distributions_REDSVM,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        verbose=0,
        scoring=scorer,
        n_jobs=-1,
        return_train_score=True,
    )

    # %%
    # CrossValidation  para SVOREX
    param_distributions_SVOREX = {
        "kernel": [0, 1],
        "degree": [1, 2, 3],
        "C": np.logspace(-3, 3, 7),
        "kappa": np.logspace(-1, 1, 3),
    }

    # Scorer personalizado
    scorer = make_scorer(ordinal)

    # RandomizedSearchCV
    SVOREXModel = RandomizedSearchCV(
        estimator=SVOREX(),
        param_distributions=param_distributions_SVOREX,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        verbose=0,
        scoring=scorer,
        n_jobs=-1,
        return_train_score=True,
    )

    # %%
    # CrossValidation para LogisticAT
    param_distributions_LogisticAT = {
        "max_iter": [100, 1000, 5000],
        "alpha": np.logspace(-3, 3, 10)
    }

    # Scorer personalizado
    scorer = make_scorer(ordinal)

    # RandomizedSearchCV
    LogisticATModel = RandomizedSearchCV(
        estimator=LogisticAT(),
        param_distributions=param_distributions_LogisticAT,
        cv=StratifiedKFold(n_splits=3, shuffle=True),
        verbose=0,
        scoring=scorer,
        n_jobs=-1,
        return_train_score=True,
    )

    # %%
    # Método determinista

    for i in range(10):
        print(f"  Ejecución nº {i + 1} REDSVM")
        # Se selecciona los parámetros del modelo REDSVM
        #REDSVMModel = REDSVM() # REDSVM por defecto

        # Se entrena el modelo REDSVM con los datos de entrenamiento
        REDSVMModel = REDSVMModel.fit(XtrainJerarquico2, YtrainJerarquico2)

        # Se predicen las etiquetas de train
        REDSVMPredictionsTrain = REDSVMModel.predict(XtrainJerarquico2)

        # Se predicen las etiquetas de test
        REDSVMPredictions = REDSVMModel.predict(XtestJerarquico2)

        # Resultados train
        maeTrainREDSVM.append(mean_absolute_error(YtrainJerarquico2, REDSVMPredictionsTrain))
        ccrTrainREDSVM.append(accuracy_score(YtrainJerarquico2, REDSVMPredictionsTrain))
        amaeTrainREDSVM.append(metrics.amae(YtrainJerarquico2, REDSVMPredictionsTrain))
        wkappaTrainREDSVM.append(metrics.wkappa(YtrainJerarquico2, REDSVMPredictionsTrain))
        #print("Resultados obtenidos en train")
        #print("  MAE = {}".format(maeTrainREDSVM[i]))
        #print("  CCR = {}".format(ccrTrainREDSVM[i]))
        #print("  AMAE = {}".format(amaeTrainREDSVM[i]))
        #print("  WKAPPA = {}".format(wkappaTrainREDSVM[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtrainJerarquico2, REDSVMPredictionsTrain)}")

        # Resultados test
        maeTestREDSVM.append(mean_absolute_error(YtestJerarquico2, REDSVMPredictions))
        ccrTestREDSVM.append(accuracy_score(YtestJerarquico2, REDSVMPredictions))
        amaeTestREDSVM.append(metrics.amae(YtestJerarquico2, REDSVMPredictions))
        wkappaTestREDSVM.append(metrics.wkappa(YtestJerarquico2, REDSVMPredictions))
        #print("Resultados obtenidos en test")
        #print("  MAE = {}".format(maeTestREDSVM[i]))
        #print("  CCR = {}".format(ccrTestREDSVM[i]))
        #print("  AMAE = {}".format(amaeTestREDSVM[i]))
        #print("  WKAPPA = {}".format(wkappaTestREDSVM[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtestJerarquico2, REDSVMPredictions)}")
        print(f"    Mejor REDSVMModel: {REDSVMModel.best_params_}")

    # %%
    # Método determinista

    for i in range(10):
        print(f"  Ejecución nº {i + 1} SVOREX")
        # Se selecciona los parámetros del modelo SVOREX
        #SVOREXModel = SVOREX() # SVOREX por defecto

        # Se entrena el modelo SVOREX con los datos de entrenamiento
        SVOREXModel = SVOREXModel.fit(XtrainJerarquico2, YtrainJerarquico2)

        # Se predicen las etiquetas de train
        SVOREXPredictionsTrain = SVOREXModel.predict(XtrainJerarquico2)

        # Se predicen las etiquetas de test
        SVOREXPredictions = SVOREXModel.predict(XtestJerarquico2)

        # Resultados train
        maeTrainSVOREX.append(mean_absolute_error(YtrainJerarquico2, SVOREXPredictionsTrain))
        ccrTrainSVOREX.append(accuracy_score(YtrainJerarquico2, SVOREXPredictionsTrain))
        amaeTrainSVOREX.append(metrics.amae(YtrainJerarquico2, SVOREXPredictionsTrain))
        wkappaTrainSVOREX.append(metrics.wkappa(YtrainJerarquico2, SVOREXPredictionsTrain))
        #print("Resultados obtenidos en train")
        #print("  MAE = {}".format(maeTrainSVOREX[i]))
        #print("  CCR = {}".format(ccrTrainSVOREX[i]))
        #print("  AMAE = {}".format(amaeTrainSVOREX[i]))
        #print("  WKAPPA = {}".format(wkappaTrainSVOREX[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtrainJerarquico2, SVOREXPredictionsTrain)}")

        # Resultados test
        maeTestSVOREX.append(mean_absolute_error(YtestJerarquico2, SVOREXPredictions))
        ccrTestSVOREX.append(accuracy_score(YtestJerarquico2, SVOREXPredictions))
        amaeTestSVOREX.append(metrics.amae(YtestJerarquico2, SVOREXPredictions))
        wkappaTestSVOREX.append(metrics.wkappa(YtestJerarquico2, SVOREXPredictions))
        #print("Resultados obtenidos en test")
        #print("  MAE = {}".format(maeTestSVOREX[i]))
        #print("  CCR = {}".format(ccrTestSVOREX[i]))
        #print("  AMAE = {}".format(amaeTestSVOREX[i]))
        #print("  WKAPPA = {}".format(wkappaTestSVOREX[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtestJerarquico2, SVOREXPredictions)}")
        print(f"    Mejor SVOREXModel: {SVOREXModel.best_params_}")

    # %%
    # Método determinista

    for i in range(10):
        print(f"  Ejecución nº {i + 1} LogisticAT")
        # Se selecciona los parámetros del modelo LogisticAT
        #LogisticATModel = LogisticAT()

        # Se entrena el modelo LogisticAT con los datos de entrenamiento
        LogisticATModel = LogisticATModel.fit(np.array(XtrainJerarquico2), np.array(YtrainJerarquico2))

        # Se predicen las etiquetas de train
        LogisticATPredictionsTrain = LogisticATModel.predict(np.array(XtrainJerarquico2))

        # Se predicen las etiquetas de test
        LogisticATPredictions = LogisticATModel.predict(np.array(XtestJerarquico2))

        # Resultados train
        maeTrainLAT.append(mean_absolute_error(YtrainJerarquico2, LogisticATPredictionsTrain))
        ccrTrainLAT.append(accuracy_score(YtrainJerarquico2, LogisticATPredictionsTrain))
        amaeTrainLAT.append(metrics.amae(YtrainJerarquico2, LogisticATPredictionsTrain))
        wkappaTrainLAT.append(metrics.wkappa(YtrainJerarquico2, LogisticATPredictionsTrain))
        #print("Resultados obtenidos en train")
        #print("  MAE = {}".format(maeTrainLAT[i]))
        #print("  CCR = {}".format(ccrTrainLAT[i]))
        #print("  AMAE = {}".format(amaeTrainLAT[i]))
        #print("  WKAPPA = {}".format(wkappaTrainLAT[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtrainJerarquico2, LogisticATPredictionsTrain)}")

        # Resultados test
        maeTestLAT.append(mean_absolute_error(YtestJerarquico2, LogisticATPredictions))
        ccrTestLAT.append(accuracy_score(YtestJerarquico2, LogisticATPredictions))
        amaeTestLAT.append(metrics.amae(YtestJerarquico2, LogisticATPredictions))
        wkappaTestLAT.append(metrics.wkappa(YtestJerarquico2, LogisticATPredictions))
        #print("Resultados obtenidos en test")
        #print("  MAE = {}".format(maeTestLAT[i]))
        #print("  CCR = {}".format(ccrTestLAT[i]))
        #print("  AMAE = {}".format(amaeTestLAT[i]))
        #print("  WKAPPA = {}".format(wkappaTestLAT[i]))
        #print(f"  Confusion Matrix = \n{confusion_matrix(YtestJerarquico2, LogisticATPredictions)}")
        print(f"    Mejor LogisticATModel: {LogisticATModel.best_params_}")

    print("* * * * * * * * * * * * * *")

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train REDSVM - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainREDSVM), np.std(maeTrainREDSVM)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainREDSVM), np.std(ccrTrainREDSVM)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTrainREDSVM), np.std(amaeTrainREDSVM)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTrainREDSVM), np.std(wkappaTrainREDSVM)))

print("Resultados obtenidos en test REDSVM - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestREDSVM), np.std(maeTestREDSVM)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestREDSVM), np.std(ccrTestREDSVM)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTestREDSVM), np.std(amaeTestREDSVM)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTestREDSVM), np.std(wkappaTestREDSVM)))

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train SVOREX - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainSVOREX), np.std(maeTrainSVOREX)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainSVOREX), np.std(ccrTrainSVOREX)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTrainSVOREX), np.std(amaeTrainSVOREX)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTrainSVOREX), np.std(wkappaTrainSVOREX)))

print("Resultados obtenidos en test SVOREX - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestSVOREX), np.std(maeTestSVOREX)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestSVOREX), np.std(ccrTestSVOREX)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTestSVOREX), np.std(amaeTestSVOREX)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTestSVOREX), np.std(wkappaTestSVOREX)))

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")

print("Resultados obtenidos en train LAT - media")
print("  MAE = {}, std = {}".format(np.mean(maeTrainLAT), np.std(maeTrainLAT)))
print("  CCR = {}, std = {}".format(np.mean(ccrTrainLAT), np.std(ccrTrainLAT)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTrainLAT), np.std(amaeTrainLAT)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTrainLAT), np.std(wkappaTrainLAT)))

print("Resultados obtenidos en test LAT - media")
print("  MAE = {}, std = {}".format(np.mean(maeTestLAT), np.std(maeTestLAT)))
print("  CCR = {}, std = {}".format(np.mean(ccrTestLAT), np.std(ccrTestLAT)))
print("  AMAE = {}, std = {}".format(np.mean(amaeTestLAT), np.std(amaeTestLAT)))
print("  WKAPPA = {}, std = {}".format(np.mean(wkappaTestLAT), np.std(wkappaTestLAT)))

print("-- -- -- -- -- -- -- -- -- -- -- -- -- --")