# PLANO DE TREINAMENTO — Bateria Sistemática de Experimentos

> Criado em 2026-04-28. Retome este documento se a sessão for interrompida.
> Objetivo: garantir com máxima certeza que o melhor modelo é selecionado antes da Fase 4 (API).

---

## Contexto

O dataset Telco Churn (~7k registros, 26 features pós-limpeza) apresenta um teto natural de ~0.861 AUC-ROC para MLP e LogReg.
A bateria aqui descrita esgota sistematicamente as variáveis de configuração relevantes antes de avançar para a API.

---

## Técnicas adicionais aprovadas (além das já implementadas)

### Feature Engineering (implementar em `preprocessing.py`)

| Feature | Descrição | Implementação |
|---------|-----------|---------------|
| `service_count` | Número de serviços ativos | Soma de colunas: Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies |
| `is_new` | Flag: Tenure ≤ 3 meses | `(Tenure Months <= 3).astype(int)` |
| `charges_per_tenure` | Monthly Charges / (Tenure + 1) | Divide antes de StandardScaler |

### Treino (implementar em `trainer.py`)

| Técnica | Parâmetro | Valor sugerido |
|---------|-----------|----------------|
| Gradient clipping | `max_norm` | 1.0 — ativar quando batch=16 |
| Label smoothing | `label_smoothing` | 0.05 — suaviza BCE |

---

## Naming Convention

```
Baselines: {modelo}_{split}_{tenure}
MLP:       mlp_{split}_{tenure}_b{batch}
```

**Split:** `7015` = 70/15/15 | `8010` = 80/10/10 (+ blind test nos 10% finais)  
**Tenure:** `orig` = sem binning | `le` = label encoding 0–3 | `ohe` = one-hot com cutoffs explícitos  
**Batch:** `b16` | `b32` | `b64`

**Cutoffs do tenure_bin:**
- `0-12m`: Tenure Months ≤ 12 (cliente novo — maior risco de churn)
- `13-24m`: 13 a 24 meses
- `25-48m`: 25 a 48 meses
- `49+m`: 49+ meses (cliente fiel — menor risco)

---

## Tabela de 48 Runs

### Baselines (30 runs — batch N/A)

| # | Run name | Split | Tenure | Status |
|---|----------|-------|--------|--------|
| 1 | `dummy_7015_orig` | 70/15/15 | orig | ⬜ |
| 2 | `logreg_7015_orig` | 70/15/15 | orig | ⬜ |
| 3 | `logreg_noml_7015_orig` | 70/15/15 | orig | ⬜ |
| 4 | `logreg_nophone_7015_orig` | 70/15/15 | orig | ⬜ |
| 5 | `logreg_nophone_noml_7015_orig` | 70/15/15 | orig | ⬜ |
| 6 | `dummy_7015_le` | 70/15/15 | le | ⬜ |
| 7 | `logreg_7015_le` | 70/15/15 | le | ⬜ |
| 8 | `logreg_noml_7015_le` | 70/15/15 | le | ⬜ |
| 9 | `logreg_nophone_7015_le` | 70/15/15 | le | ⬜ |
| 10 | `logreg_nophone_noml_7015_le` | 70/15/15 | le | ⬜ |
| 11 | `dummy_7015_ohe` | 70/15/15 | ohe | ⬜ |
| 12 | `logreg_7015_ohe` | 70/15/15 | ohe | ⬜ |
| 13 | `logreg_noml_7015_ohe` | 70/15/15 | ohe | ⬜ |
| 14 | `logreg_nophone_7015_ohe` | 70/15/15 | ohe | ⬜ |
| 15 | `logreg_nophone_noml_7015_ohe` | 70/15/15 | ohe | ⬜ |
| 16 | `dummy_8010_orig` | 80/10/10 | orig | ⬜ |
| 17 | `logreg_8010_orig` | 80/10/10 | orig | ⬜ |
| 18 | `logreg_noml_8010_orig` | 80/10/10 | orig | ⬜ |
| 19 | `logreg_nophone_8010_orig` | 80/10/10 | orig | ⬜ |
| 20 | `logreg_nophone_noml_8010_orig` | 80/10/10 | orig | ⬜ |
| 21 | `dummy_8010_le` | 80/10/10 | le | ⬜ |
| 22 | `logreg_8010_le` | 80/10/10 | le | ⬜ |
| 23 | `logreg_noml_8010_le` | 80/10/10 | le | ⬜ |
| 24 | `logreg_nophone_8010_le` | 80/10/10 | le | ⬜ |
| 25 | `logreg_nophone_noml_8010_le` | 80/10/10 | le | ⬜ |
| 26 | `dummy_8010_ohe` | 80/10/10 | ohe | ⬜ |
| 27 | `logreg_8010_ohe` | 80/10/10 | ohe | ⬜ |
| 28 | `logreg_noml_8010_ohe` | 80/10/10 | ohe | ⬜ |
| 29 | `logreg_nophone_8010_ohe` | 80/10/10 | ohe | ⬜ |
| 30 | `logreg_nophone_noml_8010_ohe` | 80/10/10 | ohe | ⬜ |

### MLP PyTorch (18 runs — split 80/10/10 incluem blind test)

| # | Run name | Split | Tenure | Batch | Blind test | Status |
|---|----------|-------|--------|-------|-----------|--------|
| 31 | `mlp_7015_orig_b64` | 70/15/15 | orig | 64 | — | ⬜ |
| 32 | `mlp_7015_orig_b32` | 70/15/15 | orig | 32 | — | ⬜ |
| 33 | `mlp_7015_orig_b16` | 70/15/15 | orig | 16 | — | ⬜ |
| 34 | `mlp_7015_le_b64` | 70/15/15 | le | 64 | — | ⬜ |
| 35 | `mlp_7015_le_b32` | 70/15/15 | le | 32 | — | ⬜ |
| 36 | `mlp_7015_le_b16` | 70/15/15 | le | 16 | — | ⬜ |
| 37 | `mlp_7015_ohe_b64` | 70/15/15 | ohe | 64 | — | ⬜ |
| 38 | `mlp_7015_ohe_b32` | 70/15/15 | ohe | 32 | — | ⬜ |
| 39 | `mlp_7015_ohe_b16` | 70/15/15 | ohe | 16 | — | ⬜ |
| 40 | `mlp_8010_orig_b64` | 80/10/10 | orig | 64 | ✓ | ⬜ |
| 41 | `mlp_8010_orig_b32` | 80/10/10 | orig | 32 | ✓ | ⬜ |
| 42 | `mlp_8010_orig_b16` | 80/10/10 | orig | 16 | ✓ | ⬜ |
| 43 | `mlp_8010_le_b64` | 80/10/10 | le | 64 | ✓ | ⬜ |
| 44 | `mlp_8010_le_b32` | 80/10/10 | le | 32 | ✓ | ⬜ |
| 45 | `mlp_8010_le_b16` | 80/10/10 | le | 16 | ✓ | ⬜ |
| 46 | `mlp_8010_ohe_b64` | 80/10/10 | ohe | 64 | ✓ | ⬜ |
| 47 | `mlp_8010_ohe_b32` | 80/10/10 | ohe | 32 | ✓ | ⬜ |
| 48 | `mlp_8010_ohe_b16` | 80/10/10 | ohe | 16 | ✓ | ⬜ |

---

## Implementação necessária antes de treinar

### 1. `preprocessing.py` — corrigir tenure_bin OHE

```python
# No clean_raw():
# OHE variant: tenure como string com cutoffs explícitos
bins = [-1, 12, 24, 48, float("inf")]
labels = ["0-12m", "13-24m", "25-48m", "49+m"]
cleaned["tenure_bin_ohe"] = pd.cut(
    cleaned["Tenure Months"], bins=bins, labels=labels, right=True
).astype(str)

# LE variant: ordinal 0-3
cleaned["tenure_bin_le"] = pd.cut(
    cleaned["Tenure Months"], bins=bins, labels=[0, 1, 2, 3], right=True
).astype(int)
```

Adicionar `tenure_bin_le` em `ENGINEERED_NUMERIC_COLUMNS` e `tenure_bin_ohe` em `ENGINEERED_CATEGORICAL_COLUMNS`.

### 2. `build_preprocessing_pipeline()` — modo tenure

```python
def build_preprocessing_pipeline(
    *,
    exclude_columns: tuple[str, ...] = (),
    tenure_variant: str = "orig",  # "orig" | "le" | "ohe"
) -> ColumnTransformer:
```

### 3. `baseline.py` + `log_baseline_cv_run` — passar tenure_variant

### 4. `log_mlp_cv_run` — passar tenure_variant + batch_size

---

## Artefatos esperados ao final

- [ ] 48 runs no MLflow (nomes conforme naming convention acima)
- [ ] `docs/resultados_bateria.md` — tabela completa com AUC-CV, holdout, blind test
- [ ] Melhor modelo identificado (run_id) para servir na Fase 4 (FastAPI)
- [ ] Commit: `feat(experiments): systematic 48-run training battery`

---

## Retomada (se sessão interrompida)

1. Verificar quais runs já existem: `mlflow.search_runs(experiment_names=["churn-prediction"])`
2. Checar a coluna `tags.mlflow.runName` contra a tabela acima
3. Rodar apenas os runs faltantes com o script `scripts/_run_battery.py`
4. Atualizar `docs/resultados_bateria.md` com os novos resultados
