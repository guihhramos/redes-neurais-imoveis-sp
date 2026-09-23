# classificacao_corrigida.py -- versao corrigida (Parte B: prever "estado_conservacao")
import pandas as pd
import numpy as np
import joblib
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("imoveis_sp.csv")

colunas_texto = ["tipo_imovel", "cidade", "bairro", "localizacao", "possui_piscina"]
X = pd.get_dummies(df.drop(columns=["estado_conservacao"]), columns=colunas_texto)

# ERRO 3 original: y estava como texto ("bom"/"ruim"). Uma rede neural com
# saida sigmoid/binary_crossentropy precisa de numeros 0/1.
y = (df["estado_conservacao"] == "bom").astype(int)  # bom=1, ruim=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Normalizar os atributos numericos ajuda a rede a convergir (boa pratica,
# igual foi feito na regressao).
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelo = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    # ERRO 1 original: Dense(1) sem ativacao. Para classificacao binaria a
    # saida deve ser "sigmoid" (produz uma probabilidade entre 0 e 1).
    keras.layers.Dense(1, activation="sigmoid"),
])

# ERRO 2 original: loss="mse" (é de regressao). Para classificacao binaria
# a loss correta é "binary_crossentropy".
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

modelo.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=20, batch_size=256, verbose=1
)

# ERRO 3 (parte 2) original: previsoes eram probabilidades continuas e foram
# comparadas diretamente com accuracy_score, que espera classes (0/1).
# É preciso aplicar o limiar de 0,5.
probs = modelo.predict(X_test_scaled).ravel()
previsoes = (probs >= 0.5).astype(int)

print("Acuracia:", accuracy_score(y_test, previsoes))
print("\nMatriz de confusao:\n", confusion_matrix(y_test, previsoes))
print("\nClassification report:\n",
      classification_report(y_test, previsoes, target_names=["ruim", "bom"]))

# --- salvar modelo e pre-processadores, para usar depois em prever_imovel.py ---
modelo.save("modelo_classificacao.keras")
joblib.dump(scaler, "scaler_classificacao.pkl")
joblib.dump(list(X_train.columns), "colunas_classificacao.pkl")
joblib.dump(colunas_texto, "colunas_texto_classificacao.pkl")
print("Modelo e pre-processadores de classificacao salvos em disco.")

# Prever um imovel novo (sem estado_conservacao, com preco)
def preparar_novo(dict_imovel, colunas_referencia, scaler):
    novo = pd.DataFrame([dict_imovel])
    novo = pd.get_dummies(novo, columns=colunas_texto)
    novo = novo.reindex(columns=colunas_referencia, fill_value=0)
    return scaler.transform(novo)

imovel_novo_jovem = {
    "tipo_imovel": "apartamento", "cidade": "Sao Paulo", "bairro": "Moema",
    "localizacao": "centro", "metro_quadrado_terreno": 95,
    "metro_quadrado_construido": 88, "quantidade_comodos": 5,
    "quantidade_suites": 1, "possui_piscina": "sim",
    "idade_anos": 3, "preco": 950000,
}

imovel_novo_antigo = {**imovel_novo_jovem, "idade_anos": 45}

for nome, imovel in [("imovel novo (idade baixa)", imovel_novo_jovem),
                      ("imovel antigo (idade alta)", imovel_novo_antigo)]:
    prep = preparar_novo(imovel, X_train.columns, scaler)
    prob = modelo.predict(prep).ravel()[0]
    classe = "bom" if prob >= 0.5 else "ruim"
    print(f"{nome}: probabilidade de 'bom' = {prob:.3f} -> classe prevista = {classe}")