# prever_imovel.py -- pergunta os dados de um imovel no terminal e mostra:
#  - o preco previsto (regressao)
#  - a probabilidade e classe de estado_conservacao (classificacao)
#
# IMPORTANTE: rode primeiro "regressao_corrigida.py" e "classificacao_corrigida.py"
# (na mesma pasta) para gerar os arquivos de modelo/scaler que este script carrega.

import pandas as pd
import joblib
from tensorflow import keras

# ---------------------------------------------------------------------------
# 1) Carregar os modelos e pre-processadores ja treinados
# ---------------------------------------------------------------------------
try:
    modelo_reg = keras.models.load_model("modelo_regressao.keras")
    scaler_X_reg = joblib.load("scaler_X_regressao.pkl")
    scaler_y_reg = joblib.load("scaler_y_regressao.pkl")
    colunas_reg = joblib.load("colunas_regressao.pkl")
    colunas_texto_reg = joblib.load("colunas_texto_regressao.pkl")
    tem_regressao = True
except Exception as e:
    print("Aviso: nao consegui carregar o modelo de regressao "
          f"(rode regressao_corrigida.py primeiro). Detalhe: {e}")
    tem_regressao = False

try:
    modelo_clf = keras.models.load_model("modelo_classificacao.keras")
    scaler_clf = joblib.load("scaler_classificacao.pkl")
    colunas_clf = joblib.load("colunas_classificacao.pkl")
    colunas_texto_clf = joblib.load("colunas_texto_classificacao.pkl")
    tem_classificacao = True
except Exception as e:
    print("Aviso: nao consegui carregar o modelo de classificacao "
          f"(rode classificacao_corrigida.py primeiro). Detalhe: {e}")
    tem_classificacao = False

if not tem_regressao and not tem_classificacao:
    raise SystemExit("Nenhum modelo disponivel. Treine pelo menos um antes de rodar este script.")


# ---------------------------------------------------------------------------
# 2) Funcoes auxiliares de leitura de input com validacao simples
# ---------------------------------------------------------------------------
def ler_texto(pergunta, opcoes=None):
    while True:
        valor = input(pergunta).strip().lower()
        if opcoes and valor not in opcoes:
            print(f"  -> valor invalido. Opcoes validas: {', '.join(opcoes)}")
            continue
        return valor


def ler_numero(pergunta):
    while True:
        valor = input(pergunta).strip().replace(",", ".")
        try:
            return float(valor)
        except ValueError:
            print("  -> digite um numero valido (ex: 85 ou 85.5)")


def preparar_para_modelo(dados: dict, colunas_texto, colunas_referencia, scaler):
    """Aplica o mesmo one-hot encoding + normalizacao usados no treino."""
    df_novo = pd.DataFrame([dados])
    df_novo = pd.get_dummies(df_novo, columns=colunas_texto)
    df_novo = df_novo.reindex(columns=colunas_referencia, fill_value=0)
    return scaler.transform(df_novo)


# ---------------------------------------------------------------------------
# 3) Coletar os dados do imovel manualmente
# ---------------------------------------------------------------------------
print("=== Preencha os dados do imovel ===\n")

dados_imovel = {
    "tipo_imovel": ler_texto(
        "Tipo do imovel (apartamento/casa/chacara/sitio): ",
        opcoes=["apartamento", "casa", "chacara", "sitio"],
    ),
    "cidade": input("Cidade: ").strip(),
    "bairro": input("Bairro: ").strip(),
    "localizacao": ler_texto(
        "Localizacao (centro/periferia): ", opcoes=["centro", "periferia"]
    ),
    "metro_quadrado_terreno": ler_numero("Metragem do terreno (m2): "),
    "metro_quadrado_construido": ler_numero("Metragem construida (m2): "),
    "quantidade_comodos": ler_numero("Quantidade de comodos: "),
    "quantidade_suites": ler_numero("Quantidade de suites: "),
    "possui_piscina": ler_texto("Possui piscina? (sim/nao): ", opcoes=["sim", "nao"]),
    "idade_anos": ler_numero("Idade do imovel (anos): "),
}

# ---------------------------------------------------------------------------
# 4) Previsao de PRECO (regressao) -- precisa tambem de estado_conservacao como atributo
# ---------------------------------------------------------------------------
if tem_regressao:
    estado_para_preco = ler_texto(
        "\n[Para estimar o preco] Estado de conservacao (bom/ruim), "
        "se souber (ou deixe 'bom' como padrao): ",
        opcoes=["bom", "ruim"],
    )
    dados_para_preco = {**dados_imovel, "estado_conservacao": estado_para_preco}

    X_prep = preparar_para_modelo(
        dados_para_preco, colunas_texto_reg, colunas_reg, scaler_X_reg
    )
    pred_scaled = modelo_reg.predict(X_prep, verbose=0)
    preco_previsto = scaler_y_reg.inverse_transform(pred_scaled).ravel()[0]
    print(f"\n>> Preco estimado: R$ {preco_previsto:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

# ---------------------------------------------------------------------------
# 5) Previsao de ESTADO DE CONSERVACAO (classificacao) -- precisa de preco como atributo
# ---------------------------------------------------------------------------
if tem_classificacao:
    preco_para_estado = ler_numero(
        "\n[Para estimar o estado de conservacao] Preco do imovel (R$), "
        "se souber (ou uma estimativa): "
    )
    dados_para_estado = {**dados_imovel, "preco": preco_para_estado}

    X_prep_clf = preparar_para_modelo(
        dados_para_estado, colunas_texto_clf, colunas_clf, scaler_clf
    )
    prob = modelo_clf.predict(X_prep_clf, verbose=0).ravel()[0]
    classe = "bom" if prob >= 0.5 else "ruim"
    print(f">> Estado de conservacao previsto: {classe}  (probabilidade de 'bom' = {prob:.1%})")

print("\nFim.")