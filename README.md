# IA para Avaliação de Imóveis — Redes Neurais com TensorFlow/Keras

Atividade prática do curso de **Programação em Inteligência Artificial**, aplicando
Deep Learning (TensorFlow/Keras) a dois problemas clássicos de aprendizado supervisionado
sobre uma base de 100.000 imóveis de São Paulo:

- 📊 **Regressão** — estimar o **preço** de um imóvel a partir de suas características.
- ✅ **Classificação binária** — prever se o imóvel está em **bom** ou **ruim** estado de conservação.

## 🎯 Objetivos

- Construir redes neurais com `keras.Sequential` (camadas `Dense`).
- Entender por que uma rede neural exige normalização dos dados (diferente de modelos
  baseados em árvore, como Random Forest).
- Escolher corretamente a camada de saída e a função de perda para cada tipo de tarefa
  (linear + `mse` para regressão; `sigmoid` + `binary_crossentropy` para classificação).
- Treinar, avaliar (R², MAE, acurácia, matriz de confusão) e usar os modelos para prever
  dados de imóveis novos.

## 📁 Estrutura do repositório

```
.
├── regressao_corrigida.py       # Treina e avalia o modelo de regressão (preço)
├── classificacao_corrigida.py   # Treina e avalia o modelo de classificação (estado de conservação)
├── prever_imovel.py             # Script interativo: informe os dados do imóvel e receba as previsões
├── respostas_atividade.md       # Relatório completo: prompts, erros corrigidos, métricas e reflexão
└── imoveis_sp.csv               # Base de dados (não incluída no repositório)
```

## 🚀 Como rodar

### 1. Instale as dependências
```bash
pip install tensorflow pandas scikit-learn joblib
```
> Em computadores mais antigos, use `pip install tensorflow-cpu` no lugar de `tensorflow`.

### 2. Treine os modelos
```bash
python regressao_corrigida.py
python classificacao_corrigida.py
```
Cada script imprime as métricas de avaliação e salva o modelo treinado + os
pré-processadores (scalers) em disco, para serem reaproveitados depois.

### 3. Faça previsões manuais
```bash
python prever_imovel.py
```
O script pergunta os dados do imóvel (tipo, metragem, cômodos, localização, idade etc.)
e devolve o **preço estimado** e o **estado de conservação previsto**.

## 📈 Resultados obtidos

| Métrica (Regressão) | Valor |
|---|---|
| R² | ~0,90 |
| MAE | ~R$ 225.000 |

O modelo capta relações coerentes com o domínio: localização, idade e metragem
influenciam o preço na direção esperada.

Detalhes completos — prompts usados, erros propositais corrigidos, matriz de confusão
da classificação e respostas às perguntas de reflexão — estão em
[`respostas_atividade.md`](./respostas_atividade.md).

## 🛠️ Tecnologias

- Python
- TensorFlow / Keras
- pandas
- scikit-learn

## 📚 Contexto

Projeto desenvolvido como atividade prática da disciplina de Programação em Inteligência
Artificial, com apoio de um assistente de IA para geração e depuração do código.
