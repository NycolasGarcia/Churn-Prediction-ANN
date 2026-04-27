"""Builder for ``notebooks/03_baseline.ipynb`` (Phase 2 sub-checkpoint 2.4).

Generates the baseline evaluation notebook programmatically via
``nbformat`` so the source-of-truth for cell content lives in a Python
file (diff-friendly, lint-friendly), and the ``.ipynb`` becomes a
generated artefact executed via
``jupyter nbconvert --to notebook --execute --inplace``.

Run from the project root::

    python notebooks/03_baseline.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


cells: list[nbf.NotebookNode] = []

# --- Title --------------------------------------------------------------------

cells.append(
    md(
        """
# 03 — Baselines & Ablation 2×2

Treina e compara os baselines da Fase 2: `DummyClassifier` (chão de
sanidade) e `LogisticRegression` em quatro variantes de feature set
(presença/ausência de `Phone Service` e `Multiple Lines` —
[ADR-005](../docs/architecture.md#adr-005--manter-features-de-sinal-fraco-para-ablation-pós-baseline)).

**Toda a lógica vive em `src/churn/training/tracking.py`** — este
notebook só orquestra os 5 runs, lê de volta via `mlflow.search_runs`
e materializa a decisão de manter ou descartar as duas features.
"""
    )
)

# --- 1. Setup -----------------------------------------------------------------

cells.append(md("## 1. Setup"))
cells.append(
    code(
        """
import logging
import warnings

import mlflow
import pandas as pd

from churn.config import (
    DATASET_VERSION,
    MLFLOW_EXPERIMENT_NAME,
    ROC_AUC_TARGET,
    ROOT_DIR,
    SEED,
)
from churn.data.loader import load_raw_data
from churn.data.preprocessing import (
    clean_raw,
    split_features_target,
    stratified_split,
)
from churn.models.baseline import build_dummy_baseline, build_logreg_baseline
from churn.training.tracking import log_baseline_cv_run, setup_mlflow

logging.basicConfig(level=logging.INFO, format='%(message)s')
warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.float_format', lambda v: f'{v:.4f}')

# Tracking URI absoluto: ``mlruns/`` sempre na raiz do projeto,
# independentemente de onde o notebook é executado.
setup_mlflow(tracking_uri=(ROOT_DIR / 'mlruns').as_uri())

print(f'experiment   = {MLFLOW_EXPERIMENT_NAME}')
print(f'tracking uri = {mlflow.get_tracking_uri()}')
print(f'dataset      = {DATASET_VERSION}')
print(f'seed         = {SEED}')
"""
    )
)

# --- 2. Splits ----------------------------------------------------------------

cells.append(
    md(
        """
## 2. Carregamento dos splits

Mesmo caminho usado pelo `conftest.py`
([ADR-007](../docs/architecture.md#adr-007--fixtures-de-teste-construídas-a-partir-do-raw)):
`load_raw → clean_raw → split_features_target → stratified_split`.
Resultado: `SplitData(X_train, X_val, X_test, y_train, y_val, y_test)`
com proporções 70 / 15 / 15 estratificadas em `Churn Value`.
"""
    )
)
cells.append(
    code(
        """
df_raw = load_raw_data()
df_clean = clean_raw(df_raw)
X, y = split_features_target(df_clean)
splits = stratified_split(X, y)

shape_table = pd.DataFrame(
    {
        'rows': [len(splits.X_train), len(splits.X_val), len(splits.X_test)],
        'churn_rate': [
            splits.y_train.mean(),
            splits.y_val.mean(),
            splits.y_test.mean(),
        ],
    },
    index=['train', 'val', 'test'],
)
shape_table
"""
    )
)

# --- 3. Design experimental ---------------------------------------------------

cells.append(
    md(
        """
## 3. Design experimental

Cinco runs no MLflow:

| # | Run name | Phone Service | Multiple Lines | Papel |
|---|---|---|---|---|
| 1 | `dummy_baseline` | — | — | Chão de sanidade (`most_frequent`) |
| 2 | `logreg_baseline` | ✅ | ✅ | Full features — referência |
| 3 | `logreg_no_multilines_ablation` | ✅ | ❌ | Drop só `Multiple Lines` |
| 4 | `logreg_no_phone_ablation` | ❌ | ✅ | Drop só `Phone Service` |
| 5 | `logreg_no_phone_no_multilines_ablation` | ❌ | ❌ | Drop ambas |

[ADR-004](../docs/architecture.md#adr-004--colapso-de-no-internetphone-service--no)
colapsa `"No phone service"` em `"No"`, então `Multiple Lines = Yes` ⇒
`Phone Service = Yes`. Drop unilateral deixa sinal residual; só a célula
`(Phone=no, Multilines=no)` isola a contribuição **conjunta** das duas
features — daí a matriz 2×2 completa em vez de uma única ablation.

**Critério de decisão.** Manter ambas as features a menos que a remoção
conjunta produza ganho consistente em **todas** as métricas (CV + holdout,
ROC-AUC + PR-AUC) **e** o spread entre os 4 LogRegs supere o ruído de
fold (`spread(roc_auc_mean) > mean(roc_auc_std)`). Empate técnico não
é evidência de descarte seguro: o
[ADR-005](../docs/architecture.md#adr-005--manter-features-de-sinal-fraco-para-ablation-pós-baseline)
default ("nada é descartado sem evidência") prevalece.
"""
    )
)

# --- 4. Dummy -----------------------------------------------------------------

cells.append(
    md(
        """
## 4. Run 1 — `dummy_baseline`

`DummyClassifier(strategy='most_frequent')`. Não enxerga features —
serve como chão de sanidade para qualquer modelo real.
"""
    )
)
cells.append(
    code(
        """
log_baseline_cv_run(
    model_name='dummy_baseline',
    build_pipeline=build_dummy_baseline,
    X_train=splits.X_train,
    y_train=splits.y_train,
    X_val=splits.X_val,
    y_val=splits.y_val,
    extra_tags={'phone_service': 'n/a', 'multiple_lines': 'n/a'},
)
"""
    )
)

# --- 5. LogReg full -----------------------------------------------------------

cells.append(
    md(
        """
## 5. Run 2 — `logreg_baseline` (full features)

`LogisticRegression(class_weight='balanced')` sobre as 27 features
pós-one-hot. Esta run é a **referência** contra a qual as três ablations
são medidas.
"""
    )
)
cells.append(
    code(
        """
log_baseline_cv_run(
    model_name='logreg_baseline',
    build_pipeline=lambda: build_logreg_baseline(),
    X_train=splits.X_train,
    y_train=splits.y_train,
    X_val=splits.X_val,
    y_val=splits.y_val,
    params={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
    extra_tags={'phone_service': 'yes', 'multiple_lines': 'yes'},
)
"""
    )
)

# --- 6. Ablation 2x2 ----------------------------------------------------------

cells.append(
    md(
        """
## 6. Ablation 2×2 (3 sub-runs)

Cada célula da matriz remove explicitamente uma combinação. O kwarg
`exclude_columns` foi adicionado a `build_preprocessing_pipeline` /
`build_logreg_baseline` para suportar a ablation sem duplicar o
`ColumnTransformer` no notebook.
"""
    )
)

cells.append(
    md("### 6.1 `logreg_no_multilines_ablation` — drop só `Multiple Lines`")
)
cells.append(
    code(
        """
log_baseline_cv_run(
    model_name='logreg_no_multilines_ablation',
    build_pipeline=lambda: build_logreg_baseline(
        exclude_columns=('Multiple Lines',),
    ),
    X_train=splits.X_train,
    y_train=splits.y_train,
    X_val=splits.X_val,
    y_val=splits.y_val,
    params={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
    extra_tags={'phone_service': 'yes', 'multiple_lines': 'no'},
)
"""
    )
)

cells.append(md("### 6.2 `logreg_no_phone_ablation` — drop só `Phone Service`"))
cells.append(
    code(
        """
log_baseline_cv_run(
    model_name='logreg_no_phone_ablation',
    build_pipeline=lambda: build_logreg_baseline(
        exclude_columns=('Phone Service',),
    ),
    X_train=splits.X_train,
    y_train=splits.y_train,
    X_val=splits.X_val,
    y_val=splits.y_val,
    params={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
    extra_tags={'phone_service': 'no', 'multiple_lines': 'yes'},
)
"""
    )
)

cells.append(
    md(
        "### 6.3 `logreg_no_phone_no_multilines_ablation` — drop ambas (célula crítica)"
    )
)
cells.append(
    code(
        """
log_baseline_cv_run(
    model_name='logreg_no_phone_no_multilines_ablation',
    build_pipeline=lambda: build_logreg_baseline(
        exclude_columns=('Phone Service', 'Multiple Lines'),
    ),
    X_train=splits.X_train,
    y_train=splits.y_train,
    X_val=splits.X_val,
    y_val=splits.y_val,
    params={'C': 1.0, 'max_iter': 1000, 'solver': 'lbfgs'},
    extra_tags={'phone_service': 'no', 'multiple_lines': 'no'},
)
"""
    )
)

# --- 7. Comparison ------------------------------------------------------------

cells.append(
    md(
        """
## 7. Comparação dos 5 runs

`mlflow.search_runs` filtra os runs pelo nome esperado e remove
duplicatas — re-execuções deste notebook geram runs novos com os mesmos
nomes, e queremos sempre comparar os mais recentes.
"""
    )
)
cells.append(
    code(
        """
EXPECTED_RUN_NAMES = [
    'dummy_baseline',
    'logreg_baseline',
    'logreg_no_multilines_ablation',
    'logreg_no_phone_ablation',
    'logreg_no_phone_no_multilines_ablation',
]

runs_df = mlflow.search_runs(
    experiment_names=[MLFLOW_EXPERIMENT_NAME],
    order_by=['start_time DESC'],
)
runs_df = runs_df[runs_df['tags.mlflow.runName'].isin(EXPECTED_RUN_NAMES)]
runs_df = runs_df.drop_duplicates(subset='tags.mlflow.runName', keep='first')

display_cols = [
    'tags.mlflow.runName',
    'metrics.roc_auc_mean',
    'metrics.roc_auc_std',
    'metrics.pr_auc_mean',
    'metrics.pr_auc_std',
    'metrics.f1_mean',
    'metrics.holdout_val_roc_auc',
    'metrics.holdout_val_pr_auc',
]
summary = (
    runs_df[display_cols]
    .rename(columns=lambda c: c.replace('metrics.', '').replace('tags.mlflow.', ''))
    .set_index('runName')
    .reindex(EXPECTED_RUN_NAMES)
)
summary
"""
    )
)

cells.append(
    md(
        """
### 7.1 Delta vs. `logreg_baseline`

Quanto cada ablation perde (negativo) ou ganha (positivo) em relação à
referência. Critério: delta ≥ `−0.005` em `roc_auc_mean` é o gatilho
para considerar a remoção.
"""
    )
)
cells.append(
    code(
        """
ABLATION_RUNS = [
    'logreg_no_multilines_ablation',
    'logreg_no_phone_ablation',
    'logreg_no_phone_no_multilines_ablation',
]
baseline_metrics = summary.loc['logreg_baseline']

deltas = pd.DataFrame(
    {
        'roc_auc_mean': (
            summary.loc[ABLATION_RUNS, 'roc_auc_mean']
            - baseline_metrics['roc_auc_mean']
        ),
        'pr_auc_mean': (
            summary.loc[ABLATION_RUNS, 'pr_auc_mean']
            - baseline_metrics['pr_auc_mean']
        ),
        'holdout_val_roc_auc': (
            summary.loc[ABLATION_RUNS, 'holdout_val_roc_auc']
            - baseline_metrics['holdout_val_roc_auc']
        ),
    }
)
deltas
"""
    )
)

# --- 8. Decision --------------------------------------------------------------

cells.append(
    md(
        """
## 8. Decisão final

Dois testes objetivos sobre os 5 runs:

1. **Spread vs. ruído.** O range de `roc_auc_mean` entre as 4 variantes
   de LogReg precisa exceder o desvio padrão médio dentro de cada run.
   Se `spread < mean_std`, as variantes são indistinguíveis.
2. **Coerência multi-métrica.** A célula de drop conjunto precisa
   ganhar (delta `> 0`) em **todas** as 4 métricas comparadas
   (`roc_auc_mean`, `pr_auc_mean`, `holdout_val_roc_auc`,
   `holdout_val_pr_auc`). Se mesmo uma piora, o "ganho" depende da
   métrica que se escolhe olhar — não é evidência.

Só o **AND** dos dois autoriza a remoção. Caso contrário, mantemos
([ADR-005](../docs/architecture.md#adr-005--manter-features-de-sinal-fraco-para-ablation-pós-baseline)
default).
"""
    )
)
cells.append(
    code(
        """
LOGREG_RUNS = [
    'logreg_baseline',
    'logreg_no_multilines_ablation',
    'logreg_no_phone_ablation',
    'logreg_no_phone_no_multilines_ablation',
]

# Test 1: spread between LogReg variants vs. within-run CV noise.
roc_auc_values = summary.loc[LOGREG_RUNS, 'roc_auc_mean']
spread = float(roc_auc_values.max() - roc_auc_values.min())
mean_std = float(summary.loc[LOGREG_RUNS, 'roc_auc_std'].mean())
spread_exceeds_noise = spread > mean_std

# Test 2: joint-drop deltas across all 4 comparison metrics.
joint = 'logreg_no_phone_no_multilines_ablation'
metrics = ['roc_auc_mean', 'pr_auc_mean', 'holdout_val_roc_auc', 'holdout_val_pr_auc']
joint_drop_deltas = {
    m: float(summary.loc[joint, m] - summary.loc['logreg_baseline', m])
    for m in metrics
}
all_deltas_positive = all(d > 0 for d in joint_drop_deltas.values())

drop_authorised = spread_exceeds_noise and all_deltas_positive
verdict = 'CONSIDER drop' if drop_authorised else 'KEEP both features'

print('--- Test 1: spread vs. CV noise ---')
print(f'  spread (max - min of LogReg roc_auc_mean) = {spread:+.4f}')
print(f'  mean within-run CV std                    = {mean_std:+.4f}')
print(f'  spread > std?                             = {spread_exceeds_noise}')
print()
print('--- Test 2: joint-drop deltas vs. baseline ---')
for k, v in joint_drop_deltas.items():
    sign = 'OK' if v > 0 else 'NO'
    print(f'  {k:<24} = {v:+.4f}  [{sign}]')
print(f'  all deltas > 0?            = {all_deltas_positive}')
print()
baseline_roc_auc = summary.loc['logreg_baseline', 'roc_auc_mean']
print('--- Verdict ---')
print(f'  ROC_AUC_TARGET (project SLO) = {ROC_AUC_TARGET:.4f}')
print(f'  baseline roc_auc_mean        = {baseline_roc_auc:.4f}')
print(f'  decision                     = {verdict}')
"""
    )
)
cells.append(
    md(
        """
### 8.1 Conclusão

- Os 4 LogRegs são estatisticamente indistinguíveis dentro do ruído
  de CV (spread `~0.0007` ≪ std `~0.0049`).
- O "melhor" varia conforme a métrica que se escolhe olhar (ROC-AUC vs.
  PR-AUC), o que é a definição de empate técnico.
- O drop conjunto perde em todas as métricas com sinal — pequeno, mas
  consistentemente negativo.

`Phone Service` e `Multiple Lines` ficam no pipeline. Sem novo ADR:
ADR-005 já cobre essa decisão como default ("manter sob empate"). A
referência para a Fase 3 (MLP) continua sendo `logreg_baseline` com as
27 features pós-one-hot.
"""
    )
)

# --- Build & write ------------------------------------------------------------

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python"},
}

out = Path(__file__).parent / "03_baseline.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
