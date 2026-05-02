# Churn Prediction — Telco Customer Churn

Pipeline end-to-end para previsão de churn em telecomunicações: rede neural
MLP (PyTorch) como modelo principal, baselines em scikit-learn para comparação,
tracking de experimentos com MLflow e API de inferência em FastAPI.

Projeto desenvolvido para o **Tech Challenge — Fase 01** da pós-graduação em
Machine Learning Engineering (FIAP MLET).

> **Status:** em desenvolvimento ativo. Este README é incremental — ele é
> atualizado ao final de cada fase com novos links e instruções; a versão
> definitiva (com diagrama de arquitetura e tabela de resultados) é entregue
> na fase 5.

## Estado do projeto

| Fase | Entregável | Status |
|---|---|---|
| 1 | Setup, EDA e pipeline de pré-processamento | ✅ |
| 2 | Baselines (Dummy + LogReg) com tracking MLflow + ablation 2×2 | ✅ |
| 3 | MLP PyTorch + Random Forest + análise comparativa de custo | ✅ |
| 4 | API FastAPI + testes pytest (69 testes, 88% coverage) | ✅ |
| 5 | Documentação final, Model Card completo, deploy | ⏳ |

## Resultados

Todos os modelos avaliados no mesmo holdout (val 10%, n=705) e blind test
(test 10%, n=705). Split canônico **80/10/10**, seed=42
([ADR-009](docs/architecture.md)). Feature engineering canônica: **one-hot +
tenure bins** ([ADR-010](docs/architecture.md)). Métrica primária: ROC-AUC;
alvo `≥ 0,80`.

### Comparativo final — val holdout (threshold=0,50)

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Log Loss |
|---|---|---|---|---|---|---|---|
| `dummy_baseline` | 0,735 | 0,000 | 0,000 | 0,000 | 0,500 | 0,265 | 9,561 |
| `logreg_nophone_noml_8010_le` | 0,786 | 0,566 | 0,829 | 0,673 | **0,873** | 0,697 | 0,451 |
| `rf_8010_orig_v2` | 0,784 | 0,569 | 0,775 | 0,656 | 0,870 | 0,678 | 0,425 |
| **`mlp_8010_ohe_b16`** | 0,783 | 0,560 | **0,845** | **0,674** | 0,870 | **0,691** | 0,443 |

### Blind test ROC-AUC

| Modelo | Blind test ROC-AUC |
|---|---|
| **`mlp_8010_ohe_b16`** | **0,8651** ← modelo para a API |
| `rf_8010_orig_v2` | 0,8605 |
| `rf_8010_orig` (v1) | 0,8592 |

**Modelo para a API:** `mlp_8010_ohe_b16`, threshold de deploy `0,27`
(minimiza custo de negócio com FP=R$50 e FN=R$500).

A busca estendida no RF confirmou que `max_depth=10` é o ótimo genuíno do
dataset — não sub-ajuste por busca insuficiente. MLP vence por Δ=+0,0046
no blind test. Análise completa em [05_rfm.ipynb](notebooks/05_rfm.ipynb).

### Baselines — Fase 2 (referência)

| Run | ROC-AUC CV (mean ± std) | PR-AUC CV (mean ± std) |
|---|---|---|
| `dummy_baseline` | 0,5000 ± 0,0000 | 0,2654 ± 0,0004 |
| `logreg_baseline` | **0,8588 ± 0,0049** | **0,6854 ± 0,0209** |
| `logreg_no_phone_ablation` | 0,8590 ± 0,0050 | 0,6852 ± 0,0218 |
| `logreg_no_multilines_ablation` | 0,8589 ± 0,0047 | 0,6851 ± 0,0204 |
| `logreg_no_phone_no_multilines_ablation` | 0,8583 ± 0,0049 | 0,6812 ± 0,0216 |

Ablation 2×2 (`Phone Service` × `Multiple Lines`): variantes indistinguíveis
(spread `~0,0006` ≪ std CV `~0,0049`). Análise em
[03_baseline.ipynb](notebooks/03_baseline.ipynb).

## API de inferência

A API FastAPI está funcional e serve predições a partir do modelo `mlp_8010_ohe_b16`
registrado no MLflow local.

### Subindo a API

```bash
# Ativar o venv e subir o servidor (reload automático no desenvolvimento)
source .venv/Scripts/activate     # Git Bash / macOS / Linux
# .venv\Scripts\activate          # PowerShell / cmd

make run
# equivalente a: uvicorn churn.api.main:app --reload --port 8000
```

A API carrega o modelo do MLflow local na inicialização. Se o MLflow ainda não
tiver o run canônico (`mlp_8010_ohe_b16`), execute o notebook `04_mlp.ipynb`
primeiro.

### Endpoints

| Método | Path | Descrição |
|---|---|---|
| `GET` | `/health` | Liveness check — retorna status e versão do modelo |
| `POST` | `/predict` | Recebe dados de um cliente, retorna probabilidade e nível de risco |
| `GET` | `/docs` | Swagger UI interativo (gerado automaticamente pelo FastAPI) |

### Testando via Swagger UI (recomendado)

1. Com a API no ar, abra `http://localhost:8000/docs`
2. Clique em **`POST /predict`** para expandir o endpoint
3. Clique em **"Try it out"** (botão no canto direito)
4. No campo **Request body**, apague o conteúdo e cole um dos payloads abaixo
5. Clique em **"Execute"**
6. A resposta aparece em **Response body**

### Dados de exemplo

**Cliente de alto risco** — contrato mensal, fibra, 2 meses de tenure, sem
serviços de segurança. Espera-se `"risk_level": "high"` e
`"churn_prediction": true`.

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

**Cliente de baixo risco** — contrato bienal, DSL, 58 meses de tenure, vários
serviços ativos, tem família. Espera-se `"risk_level": "low"` e
`"churn_prediction": false`.

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

### Testando via curl

```bash
# Health check
curl http://localhost:8000/health

# Predição (cole o JSON de qualquer um dos exemplos acima em data.json e execute)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data.json
```

## Requisitos

- Python **3.12** (versão fixada em [.python-version](.python-version))
- `make` disponível no PATH (opcional — todos os targets do [Makefile](Makefile)
  também podem ser executados diretamente como comandos shell)

## Setup rápido

```bash
# 1. Criar e ativar o ambiente virtual
python -m venv .venv
source .venv/Scripts/activate    # Git Bash no Windows
# ou: .venv\Scripts\activate     # PowerShell / cmd

# 2. Instalar o projeto em modo editável com extras de desenvolvimento
pip install -e ".[dev]"
```

## Comandos do Makefile

| Comando | O que faz | Disponível desde |
|---|---|---|
| `make install` | instala deps + projeto em modo editável | fase 1 |
| `make lint` | `ruff check src/ tests/` | fase 1 |
| `make format` | `ruff format src/ tests/` | fase 1 |
| `make test` | suite `pytest` com cobertura | fase 4 |
| `make train-baseline` | treina Dummy + LogReg, loga no MLflow | fase 2 |
| `make train-mlp` | treina MLP em PyTorch | fase 3 |
| `make mlflow-ui` | sobe MLflow UI em `localhost:5000` | fase 2 |
| `make run` | sobe API FastAPI em `localhost:8000` | fase 4 |

## Notebooks

Após o setup, executar com Jupyter Lab:

```bash
jupyter lab notebooks/
```

| Notebook | Conteúdo |
|---|---|
| [01_eda.ipynb](notebooks/01_eda.ipynb) | Análise exploratória completa (qualidade, leakage, distribuições, correlação, ranking de features) |
| [02_data_prep.ipynb](notebooks/02_data_prep.ipynb) | Pipeline de pré-processamento (cleaning, split 80/10/10 estratificado, encoding, exportação para `data/processed/`) |
| [03_baseline.ipynb](notebooks/03_baseline.ipynb) | Baselines (Dummy + LogReg) com tracking MLflow, ablation 2×2 e métricas completas em dois thresholds |
| [04_mlp.ipynb](notebooks/04_mlp.ipynb) | MLP PyTorch — arquitetura, bateria de treino, análise de custo, blind test e comparação com baselines |
| [05_rfm.ipynb](notebooks/05_rfm.ipynb) | Random Forest — ablação de FE (orig/le/ohe), busca estendida v2, comparativo final com MLP e seleção do modelo |

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/ml_canvas.md](docs/ml_canvas.md) | ML Canvas — proposta de valor, stakeholders, métricas, SLOs, decisões de feature |
| [docs/architecture.md](docs/architecture.md) | Architecture Decision Records — ADR-001 a ADR-010, registro vivo das decisões metodológicas |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | Plano de monitoramento (sinais, thresholds, retreino) |
| [docs/data_description.md](docs/data_description.md) | Dicionário das colunas do dataset bruto |
| [MODEL_CARD.md](MODEL_CARD.md) | Model Card — métricas, limitações, análise de viés, cenários de falha |
| [docs/Tese-do-Projeto.md](docs/Tese-do-Projeto.md) | Requisitos completos do Tech Challenge |

## Estrutura do projeto

```
churn-prediction/
├── data/
│   ├── raw/                # raw_data.xlsx versionado (~1,3 MB)
│   └── processed/          # splits + preprocessor (gitignored)
├── docs/                   # canvas, ADRs, plano de monitoramento, dicionário
├── notebooks/              # 01_eda … 05_rfm (todos executados com outputs)
├── src/churn/              # pacote Python
│   ├── config.py           # SEED, paths, custos, constantes do projeto
│   ├── data/               # loader.py + preprocessing.py
│   ├── models/             # baseline.py (LogReg/Dummy) + mlp.py (PyTorch)
│   ├── training/           # trainer.py + evaluate.py + tracking.py
│   └── api/                # FastAPI app (fase 4)
├── tests/                  # pytest suite (fase 4)
├── pyproject.toml          # deps, ruff, pytest
└── Makefile                # alvos de install/lint/test/run/train
```

## Dataset

**Telco Customer Churn (IBM)** — 7.043 clientes, 33 colunas. O arquivo bruto
está versionado em [data/raw/raw_data.xlsx](data/raw/raw_data.xlsx) (~1,3 MB)
para que qualquer clone do repositório tenha um setup funcional sem download
adicional. Justificativa em [ADR-002](docs/architecture.md#adr-002--versionar-raw_dataxlsx-no-repositório).

A descrição das colunas está em [docs/data_description.md](docs/data_description.md).

## Reprodutibilidade

`SEED = 42` está definido em [src/churn/config.py](src/churn/config.py) e é
aplicado em **numpy**, **scikit-learn** e **PyTorch** (CPU + CUDA). Splits
estratificados são determinísticos dado o seed — verificado em
`02_data_prep.ipynb` seção 8.2.

## Licença

Projeto educacional. Consultar o material da disciplina para os termos de uso.
