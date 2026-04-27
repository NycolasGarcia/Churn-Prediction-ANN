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
| 2 | Baselines (Dummy + LogReg) com tracking MLflow | ⏳ |
| 3 | MLP em PyTorch e análise de custo | ⏳ |
| 4 | Refatoração modular, API FastAPI e testes | ⏳ |
| 5 | Model Card, plano de monitoramento final, deploy | ⏳ |

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
| [02_data_prep.ipynb](notebooks/02_data_prep.ipynb) | Pipeline de pré-processamento (cleaning, split estratificado 70/15/15, encoding) |

## Documentação

| Documento | Conteúdo |
|---|---|
| [docs/ml_canvas.md](docs/ml_canvas.md) | ML Canvas — proposta de valor, stakeholders, métricas, SLOs, decisões de feature |
| [docs/architecture.md](docs/architecture.md) | Architecture Decision Records — registro vivo das decisões metodológicas |
| [docs/monitoring_plan.md](docs/monitoring_plan.md) | Plano de monitoramento (sinais, thresholds, retreino) |
| [docs/data_description.md](docs/data_description.md) | Dicionário das colunas do dataset bruto |
| [docs/Tese-do-Projeto.md](docs/Tese-do-Projeto.md) | Requisitos completos do Tech Challenge |

## Estrutura do projeto

```
churn-prediction/
├── data/
│   ├── raw/                # raw_data.xlsx versionado (~1,3 MB)
│   └── processed/          # splits + preprocessor (gitignored)
├── docs/                   # canvas, ADRs, plano de monitoramento, dicionário
├── notebooks/              # 01_eda.ipynb, 02_data_prep.ipynb (e seguintes)
├── src/churn/              # pacote Python
│   ├── config.py           # SEED, paths, target, schema esperado
│   ├── data/               # loader + preprocessing
│   ├── models/             # baseline.py, mlp.py (a partir da fase 2)
│   ├── training/           # trainer + evaluate (a partir da fase 2)
│   └── api/                # FastAPI app (a partir da fase 4)
├── tests/                  # pytest suite (a partir da fase 4)
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
