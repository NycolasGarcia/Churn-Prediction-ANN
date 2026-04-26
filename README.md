# Churn Prediction — Telco Customer Churn

Pipeline end-to-end para previsão de churn em telecomunicações: rede neural
MLP (PyTorch) como modelo principal, baselines em scikit-learn para comparação,
tracking de experimentos com MLflow e API de inferência em FastAPI.

Projeto desenvolvido para o **Tech Challenge — Fase 01** da pós-graduação em
Machine Learning Engineering (FIAP MLET).

> **Status:** em desenvolvimento ativo. Este README é um esboço — a versão
> final, com diagrama de arquitetura, tabela de resultados e walkthrough de
> setup completo, será entregue na fase de documentação final do projeto. *WIP.*

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

## Estrutura do projeto

A organização segue o padrão `src/`, `data/`, `notebooks/`, `tests/`, `docs/`,
com o pacote Python publicado em [src/churn](src/churn). A descrição
detalhada da estrutura final será incluída na fase de documentação. *WIP.*

## Dataset

**Telco Customer Churn (IBM)** — 7.043 clientes, 33 colunas. O arquivo bruto
está versionado em [data/raw/raw_data.xlsx](data/raw/raw_data.xlsx) (~1,3 MB)
para que qualquer clone do repositório tenha um setup funcional sem download
adicional.

> **Observação:** versionar o dado bruto é uma decisão **temporária**, voltada
> à conveniência durante o desenvolvimento. Em uma refatoração posterior, o
> arquivo poderá ser removido do repositório e substituído por instruções
> de download neste README.

A descrição das colunas está em
[docs/data_description.md](docs/data_description.md). Os requisitos completos
do Tech Challenge estão em [docs/Tese-do-Projeto.md](docs/Tese-do-Projeto.md).

## Licença

Projeto educacional. Consultar o material da disciplina para os termos de uso.
