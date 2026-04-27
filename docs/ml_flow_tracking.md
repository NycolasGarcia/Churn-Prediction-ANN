# MLflow Tracking — Convenções

> Documento operacional. Centraliza nomes de runs, params, métricas, tags
> e artefatos que cada experimento do projeto deve registrar — para que
> comparações entre baselines, ablations e MLP fiquem honestas.
>
> Decisão arquitetural subjacente: [ADR-008 — 1 run agregado por modelo
> (CV folds → mean/std)](architecture.md#adr-008--mlflow-1-run-agregado-por-modelo-cv-folds--meanstd).

---

## 1. Setup

Toda execução começa por:

```python
from churn.training.tracking import setup_mlflow
setup_mlflow()
```

- Tracking URI default: `./mlruns/` (relativo ao CWD; gitignored).
- Experimento ativo: `churn-prediction` (constante
  `MLFLOW_EXPERIMENT_NAME` em [src/churn/config.py](../src/churn/config.py)).
- A função é idempotente — chamar duas vezes não duplica experimento.

Para inspecionar runs: `make mlflow-ui` (porta 5000).

---

## 2. Naming de runs

Naming determinístico para que filtros na UI funcionem por substring.

| Run name | Origem | Fase |
|---|---|---|
| `dummy_baseline` | `DummyClassifier(strategy="most_frequent")` | 2 |
| `logreg_baseline` | `LogisticRegression(class_weight="balanced")` | 2 |
| `logreg_no_phone_ablation` | LogReg sem `Phone Service` ([ADR-005](architecture.md#adr-005--manter-features-de-sinal-fraco-para-ablation-pós-baseline)) | 2 |
| `logreg_no_multilines_ablation` | LogReg sem `Multiple Lines` ([ADR-005](architecture.md#adr-005--manter-features-de-sinal-fraco-para-ablation-pós-baseline)) | 2 |
| `mlp_v<N>` | MLP PyTorch — `N` incrementa por iteração de arquitetura | 3 |

Variantes locais (debug, exploração) usam o sufixo `_dev` para serem
filtradas e podem ser deletadas a qualquer momento.

---

## 3. Params obrigatórios

Logados automaticamente por
[`log_baseline_cv_run`](../src/churn/training/tracking.py):

| Param | Origem | Observação |
|---|---|---|
| `model_type` | `type(classifier).__name__` | Ex.: `LogisticRegression`, `DummyClassifier` |
| `class_weight` | `classifier.class_weight` | `"balanced"`, `None`, dict |
| `seed` | `churn.config.SEED` | Fixo em `42` para reprodutibilidade |
| `dataset_version` | `churn.config.DATASET_VERSION` | Bumpar quando `clean_raw` ou split mudar |
| `n_features` | `len(preprocessor.get_feature_names_out())` | Pós one-hot — 27 com o pipeline atual |
| `cv_folds` | argumento do helper | Default `5` |
| `cv_strategy` | constante | `"stratified"` |

Hiperparâmetros específicos do classifier (ex.: `C` da LogReg, `penalty`,
`max_iter`) entram via o argumento `params` do helper. Caller-supplied
params **não** sobrescrevem os canônicos acima — a ideia é que toda run
tenha exatamente o mesmo conjunto de chaves canônicas, garantindo
comparabilidade.

---

## 4. Métricas obrigatórias

Schema completo definido no ADR-008. Resumo:

### CV agregado (5 folds estratificados)

Para cada `<metric>` em `{roc_auc, pr_auc, f1, precision, recall}`:

- `<metric>_mean` — média sobre os 5 folds (population, ddof=0)
- `<metric>_std` — desvio padrão (ddof=0; folds são fixos, não amostra)
- `<metric>_fold_1` ... `<metric>_fold_5` — bruto por fold para auditoria

### Holdout val (refit em todo `X_train`, avaliar em `X_val`)

- `holdout_val_roc_auc`, `holdout_val_pr_auc`, `holdout_val_f1`,
  `holdout_val_precision`, `holdout_val_recall`

O holdout val é o slice usado downstream para **calibração de threshold
e cost analysis** ([config.py — `COST_FALSE_POSITIVE` / `COST_FALSE_NEGATIVE`](../src/churn/config.py))
e **não** para seleção de modelo (essa é função do CV agregado).

### Métrica primária do projeto

**ROC-AUC** (CLAUDE.md §10 / Tese). PR-AUC é a secundária — relevante
porque a base tem ~26% de churn e PR-AUC é mais sensível a esse
desbalanceamento. Target do projeto:
[`ROC_AUC_TARGET = 0.80`](../src/churn/config.py).

---

## 5. Tags obrigatórias

| Tag | Valor |
|---|---|
| `model_type` | Duplica o param para facilitar filtro `tag.model_type = "LogisticRegression"` na UI |
| `dataset_version` | `churn.config.DATASET_VERSION` |
| `author` | `churn.config.AUTHOR` |

Tags adicionais por run vão via `extra_tags` do helper. Útil, por
exemplo, para marcar `ablation_target=phone_service` nas runs de
ablation e cruzar com a tabela de comparação no notebook.

---

## 6. Artefatos

Todo run inclui:

- `model/` — pipeline sklearn refitado em `X_train` completo
  (`mlflow.sklearn.log_model`). Contém preprocessor + classifier;
  recarregável em qualquer máquina via `mlflow.sklearn.load_model`.
- `confusion_matrix.png` — matriz no holdout val, gerada por
  `log_confusion_matrix_artifact`.

Artefatos opcionais (futuros):

- `feature_importance.png` — quando aplicável (LogReg coeffs, MLP grad ×
  input). Não obrigatório no escopo da Fase 2.
- `cost_curve.png` — análise de threshold × custo (Fase 3, MLP).

---

## 7. Filesystem e versionamento

| Caminho | Status no git |
|---|---|
| `mlruns/` | gitignored ([.gitignore](../.gitignore) linha 31) |
| `mlartifacts/` | gitignored |

Runs são locais por design — não há servidor MLflow remoto neste
projeto. A entrega final inclui as métricas comparadas no Model Card e
o pipeline serializado via `joblib` (decisão de arquitetura, CLAUDE.md §15).
