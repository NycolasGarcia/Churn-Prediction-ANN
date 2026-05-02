<div align="center">

# Churn Prediction — Telco Customer Churn

Pipeline end-to-end para previsão de churn em telecomunicações

[Resultados](#resultados) | [Pipeline](#pipeline) | [Instalação](#instalação) | [Notebooks](#notebooks) | [API](#api-de-inferência) | [Roadmap](#roadmap) | [Contato](#contato)

Ferramentas:

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.x-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Lab-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

Detalhes:

![Version](https://img.shields.io/badge/version-1.0.0-darkgrey?style=flat)
![Tests](https://img.shields.io/badge/tests-69%20passing-brightgreen?style=flat)
![Coverage](https://img.shields.io/badge/coverage-88%25-green?style=flat)
![Ruff](https://img.shields.io/badge/ruff-passing-brightgreen?style=flat)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.865-blue?style=flat)

</div>

---

## Sobre o Projeto

Projeto desenvolvido para o **Tech Challenge — Fase 01** da pós-graduação em
Machine Learning Engineering (FIAP MLET).

Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado.
O objetivo é classificar o risco de cancelamento de cada cliente para permitir
ações de retenção proativas — antes que o churn ocorra. O custo de negócio é
assimétrico: perder um cliente (FN) custa R$500; uma ação desnecessária (FP)
custa R$50. Essa assimetria guia a escolha do threshold de deploy.

---

## Resultados

Modelo de produção: **`mlp_8010_ohe_b16`** — threshold de deploy **0,27**
(minimiza custo total no val holdout: −38% vs. threshold padrão 0,50).

<div align="center">

### Comparativo final — blind test ROC-AUC

| Modelo | Blind test ROC-AUC |
|---|---|
| **`mlp_8010_ohe_b16`** | **0,8651** ← modelo em produção |
| `rf_8010_orig_v2` | 0,8605 |
| Dummy | 0,500 |

### Val holdout — threshold 0,50

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|---|
| `dummy_baseline` | 0,735 | 0,000 | 0,000 | 0,000 | 0,500 | 0,265 |
| `logreg_nophone_noml_8010_le` | 0,786 | 0,566 | 0,829 | 0,673 | **0,873** | 0,697 |
| `rf_8010_orig_v2` | 0,784 | 0,569 | 0,775 | 0,656 | 0,870 | 0,678 |
| **`mlp_8010_ohe_b16`** | 0,783 | 0,560 | **0,845** | **0,674** | 0,870 | **0,691** |

</div>

<details>
<summary><strong>Baselines — Fase 2 (CV estratificada 5-fold)</strong></summary>

<div align="center">

| Run | ROC-AUC CV (mean ± std) | PR-AUC CV (mean ± std) |
|---|---|---|
| `dummy_baseline` | 0,5000 ± 0,0000 | 0,2654 ± 0,0004 |
| `logreg_baseline` | **0,8588 ± 0,0049** | **0,6854 ± 0,0209** |
| `logreg_no_phone_ablation` | 0,8590 ± 0,0050 | 0,6852 ± 0,0218 |
| `logreg_no_multilines_ablation` | 0,8589 ± 0,0047 | 0,6851 ± 0,0204 |
| `logreg_no_phone_no_multilines_ablation` | 0,8583 ± 0,0049 | 0,6812 ± 0,0216 |

</div>

Ablation 2×2 (`Phone Service` × `Multiple Lines`): spread de ~0,0006 ≪ std de CV ~0,0049 — variantes estatisticamente indistinguíveis. Análise completa em [03_baseline.ipynb](notebooks/03_baseline.ipynb).

</details>

---

## Pipeline

```mermaid
flowchart LR
    A[raw_data.xlsx\n7.043 clientes] --> B[01_eda.ipynb\nEDA + qualidade]
    B --> C[02_data_prep.ipynb\npreprocessing.py]
    C --> D{Treino}
    D --> E[03_baseline.ipynb\nDummy + LogReg]
    D --> F[04_mlp.ipynb\nMLP PyTorch]
    D --> G[05_rfm.ipynb\nRandom Forest]
    E --> H[MLflow\nchurn-prediction]
    F --> H
    G --> H
    H --> I[mlp_8010_ohe_b16\nAUC 0,865 — threshold 0,27]
    I --> J[FastAPI\nPOST /predict]
    J --> K[Cliente /\nAplicação]
```

---

## Dataset

**Telco Customer Churn (IBM)** — 7.043 clientes, 33 colunas, ~26,5% de churn.
O arquivo bruto está versionado em [data/raw/raw_data.xlsx](data/raw/raw_data.xlsx) (~1,3 MB)
para que qualquer `git clone` produza um setup funcional sem downloads adicionais.
Dicionário completo de colunas em [docs/data_description.md](docs/data_description.md).

---

## Stack

<div align="center">

| Camada | Tecnologia | Papel |
|---|---|---|
| Dados | Pandas + openpyxl | Leitura, limpeza e feature engineering |
| Pré-processamento | scikit-learn ColumnTransformer | Scaling, OneHot, pipeline reprodutível |
| Modelo principal | PyTorch 2.x | MLP com BatchNorm, Dropout e early stopping |
| Modelos baseline | scikit-learn | DummyClassifier, LogisticRegression, RandomForest |
| Tracking | MLflow 2.x | Parâmetros, métricas e artefatos por run |
| API | FastAPI + Uvicorn | Endpoint `/predict` com validação Pydantic v2 |
| Testes | pytest + pytest-cov | 69 testes, 88% de cobertura |
| Linting | ruff | Zero erros em `src/` e `tests/` |
| Notebooks | JupyterLab | EDA, treino, análise de custo e bias |

</div>

---

## Instalação

```bash
# 1. Clonar o repositório
git clone https://github.com/NycolasGarcia/Churn-Prediction-ANN.git
cd Churn-Prediction-ANN

# 2. Criar e ativar o ambiente virtual
python -m venv .venv

    # Windows — Git Bash / WSL
    source .venv/Scripts/activate

    # Windows — PowerShell / cmd
    .venv\Scripts\activate

# 3. Instalar o projeto em modo editável com extras de desenvolvimento
pip install -e ".[dev]"
```

> O dataset (`data/raw/raw_data.xlsx`) e o modelo treinado (`models/mlp_deploy.pt`,
> `models/preprocessor_deploy.joblib`) já estão versionados — nenhum download
> ou treino adicional é necessário para rodar a API.

---

## Quick Start

```bash
# Abrir os notebooks (EDA, treino, análise)
jupyter lab notebooks/

# Treinar os modelos e registrar no MLflow
make train-baseline   # Dummy + LogReg
make train-mlp        # MLP PyTorch

# Visualizar experimentos no MLflow UI
make mlflow-ui        # → http://localhost:5000

# Subir a API de inferência
make run              # → http://localhost:8000/docs

# Rodar testes com cobertura
make test

# Verificar linting
make lint
```

## Makefile

<div align="center">

| Comando | O que faz |
|---|---|
| `make install` | Instala deps + projeto em modo editável |
| `make lint` | `ruff check src/ tests/` |
| `make format` | `ruff format src/ tests/` |
| `make test` | Suite pytest com cobertura |
| `make train-baseline` | Treina Dummy + LogReg, loga no MLflow |
| `make train-mlp` | Treina MLP em PyTorch |
| `make mlflow-ui` | MLflow UI em `localhost:5000` |
| `make run` | API FastAPI em `localhost:8000` |

</div>


## Notebooks

<div align="center">

| Notebook | Conteúdo |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Análise exploratória — qualidade, distribuições, correlações, churn por segmento |
| [02_data_prep.ipynb](notebooks/02_data_prep.ipynb) | Pré-processamento — cleaning, feature engineering, split 80/10/10, exportação |
| [03_baseline.ipynb](notebooks/03_baseline.ipynb) | Baselines com CV 5-fold, tracking MLflow, ablation 2×2 e análise de threshold |
| [04_mlp.ipynb](notebooks/04_mlp.ipynb) | MLP PyTorch — arquitetura, 21 runs, análise de custo por threshold, blind test |
| [05_rfm.ipynb](notebooks/05_rfm.ipynb) | Random Forest — ablação de FE, busca estendida `n_iter=50`, comparativo final |
| [06_bias_analysis.ipynb](notebooks/06_bias_analysis.ipynb) | Análise de viés — ROC-AUC, F1 e FNR por gênero, sênior e faixas de tenure |

</div>


## API de Inferência

A API carrega o modelo na inicialização a partir dos artefatos versionados em
`models/` — funciona em qualquer clone sem precisar rodar os notebooks de treino.
Se o MLflow local estiver disponível (após rodar `04_mlp.ipynb`), ele é usado
como fonte primária; `models/` é o fallback.

<div align="center">

| Método | Path | Descrição |
|---|---|---|
| `GET` | `/health` | Liveness check — status e versão do modelo |
| `POST` | `/predict` | Recebe features de um cliente, retorna probabilidade e nível de risco |
| `GET` | `/docs` | Swagger UI interativo (gerado automaticamente pelo FastAPI) |

</div>

**Testando via Swagger UI:** suba a API com `make run`, abra `http://localhost:8000/docs`,
expanda `POST /predict`, clique em **Try it out** e cole um dos payloads abaixo.

<details>
<summary><strong>Cliente de alto risco</strong> — contrato mensal, fibra, 2 meses de tenure → <code>"risk_level": "high"</code></summary>

```json
{
  "gender": "Female",
  "senior_citizen": "No",
  "partner": "No",
  "dependents": "No",
  "tenure_months": 2,
  "phone_service": "Yes",
  "multiple_lines": "No",
  "internet_service": "Fiber optic",
  "online_security": "No",
  "online_backup": "No",
  "device_protection": "No",
  "tech_support": "No",
  "streaming_tv": "Yes",
  "streaming_movies": "Yes",
  "contract": "Month-to-month",
  "paperless_billing": "Yes",
  "payment_method": "Electronic check",
  "monthly_charges": 85.5,
  "total_charges": 171.0,
  "cltv": 3200
}
```

</details>

<details>
<summary><strong>Cliente de baixo risco</strong> — contrato bienal, DSL, 58 meses de tenure → <code>"risk_level": "low"</code></summary>

```json
{
  "gender": "Male",
  "senior_citizen": "No",
  "partner": "Yes",
  "dependents": "Yes",
  "tenure_months": 58,
  "phone_service": "Yes",
  "multiple_lines": "Yes",
  "internet_service": "DSL",
  "online_security": "Yes",
  "online_backup": "Yes",
  "device_protection": "Yes",
  "tech_support": "Yes",
  "streaming_tv": "No",
  "streaming_movies": "No",
  "contract": "Two year",
  "paperless_billing": "No",
  "payment_method": "Bank transfer (automatic)",
  "monthly_charges": 72.0,
  "total_charges": 4176.0,
  "cltv": 5800
}
```

</details>


## Documentação

<div align="center">

| Documento | Conteúdo |
|---|---|
| [MODEL_CARD.md](MODEL_CARD.md) | Model Card — métricas, limitações, análise de viés por subgrupo |
| [docs/ml_canvas.md](docs/ml_canvas.md) | ML Canvas — proposta de valor, stakeholders, métricas, SLOs |
| [docs/architecture.md](docs/architecture.md) | Architecture Decision Records — ADR-001 a ADR-010 |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | Plano de monitoramento — sinais, alertas, retreino, fairness |
| [docs/data_description.md](docs/data_description.md) | Dicionário completo das colunas do dataset |

</div>


## Estrutura do Projeto

```
churn-prediction/
├── data/
│   ├── raw/                # raw_data.xlsx versionado (~1,3 MB)
│   └── processed/          # splits + preprocessor (gitignored)
├── models/
│   ├── mlp_deploy.pt       # state_dict do modelo de produção (~22 KB)
│   ├── preprocessor_deploy.joblib  # pipeline sklearn serializado (~8 KB)
│   └── config.json         # hiperparâmetros e threshold para reconstrução
├── docs/                   # canvas, ADRs, plano de monitoramento, dicionário
├── notebooks/              # 01_eda … 06_bias_analysis (outputs salvos)
├── src/churn/              # pacote Python instalável
│   ├── config.py           # SEED, paths, custos, constantes do projeto
│   ├── data/               # loader.py + preprocessing.py
│   ├── models/             # baseline.py (LogReg/Dummy) + mlp.py (PyTorch)
│   ├── training/           # trainer.py + evaluate.py + tracking.py
│   └── api/                # FastAPI — main.py, schemas.py, middleware.py
├── tests/                  # pytest suite (69 testes, 88% cobertura)
├── pyproject.toml          # deps, ruff, pytest — single source of truth
└── Makefile                # install / lint / test / run / train / mlflow-ui
```


## Roadmap

- [x] **Fase 1 — Setup e EDA**
  - [x] Estrutura de pastas, `pyproject.toml` e `Makefile`
  - [x] Dataset IBM Telco versionado em `data/raw/` (sem download extra)
  - [x] EDA completa (`01_eda.ipynb`) — distribuições, correlações e churn por segmento
  - [x] Pipeline de pré-processamento (`02_data_prep.ipynb`) — feature engineering, split 80/10/10 estratificado

- [x] **Fase 2 — Baselines**
  - [x] `DummyClassifier` e `LogisticRegression` com `class_weight='balanced'`
  - [x] `ColumnTransformer` com `StandardScaler` + `OneHotEncoder`
  - [x] CV estratificada 5-fold com tracking MLflow completo
  - [x] Ablation 2×2 (`Phone Service` × `Multiple Lines`) — variantes estatisticamente indistinguíveis
  - [x] `03_baseline.ipynb` com métricas em dois thresholds (0,50 e 0,27)

- [x] **Fase 3 — MLP PyTorch e Random Forest**
  - [x] Arquitetura MLP com `BatchNorm`, `Dropout` e early stopping (patience=10)
  - [x] Bateria de 21 runs — ablação de FE (`orig`/`le`/`ohe`) e split (70/15/15 vs 80/10/10)
  - [x] Análise de custo por threshold (FP=R$50 / FN=R$500) → ótimo: **0,27**
  - [x] Random Forest com busca estendida `n_iter=50` — confirma `max_depth=10` como ótimo genuíno
  - [x] Comparativo final: MLP AUC **0,865** vs RF 0,861 no blind test

- [x] **Fase 4 — API e Testes**
  - [x] FastAPI com `/predict`, `/health` e Swagger UI em `/docs`
  - [x] Validação de entrada via Pydantic v2; middleware de latência estruturado
  - [x] 69 testes pytest (smoke, schema, API) com 88% de cobertura
  - [x] `ruff check src/ tests/` sem erros

- [x] **Fase 5 — Documentação e Model Card**
  - [x] `06_bias_analysis.ipynb` — ROC-AUC, F1 e FNR por gênero, sênior e tenure
  - [x] Model Card com análise de viés (sênior: AUC 0,778; tenure 37–60m: FNR 17%)
  - [x] ADR-001 superseded por ADR-009; ADR-009 e ADR-010 documentados formalmente
  - [x] Plano de monitoramento atualizado com subgrupos prioritários de fairness


## Reprodutibilidade

`SEED = 42` definido em [src/churn/config.py](src/churn/config.py) e aplicado em
**numpy**, **scikit-learn** e **PyTorch** (CPU e CUDA). O split 80/10/10
estratificado é determinístico dado o seed — verificado em `02_data_prep.ipynb`.


## Contato

<div align="center">

| Plataforma | Link |
|---|---|
| <img src="https://skills.syvixor.com/api/icons?i=linkedin" width="30"> | [LinkedIn](https://www.linkedin.com/in/NycolasAGRGarcia/) |
| <img src="https://skills.syvixor.com/api/icons?i=github" width="30"> | [GitHub](https://github.com/NycolasGarcia) |
| <img src="https://skills.syvixor.com/api/icons?i=gmail" width="30"> | [Gmail](mailto:nycolasagrg.work@gmail.com) |
| <img src="https://skills.syvixor.com/api/icons?i=vercel" width="30"> | [Portfólio](https://dev-nycolas-garcia.vercel.app/) |

</div>