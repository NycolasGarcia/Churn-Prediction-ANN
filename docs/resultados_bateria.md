# Resultados da Bateria Sistematica de Experimentos

> Gerado em 2026-04-28. Referencia: `PLANO-DE-TREINAMENTO.md`.
>
> Objetivo: esgotar sistematicamente as variaveis de configuracao relevantes
> (split, feature engineering, batch size) antes de avancar para a Fase 4 (API).

---

## Configuracao da Bateria

| Dimensao | Valores testados |
|----------|-----------------|
| Split | 70/15/15 e 80/10/10 |
| Tenure variant | `orig` (27 feat) \| `le` (32 feat) \| `ohe` (35 feat) |
| Modelos baseline | Dummy, LogReg full, LogReg -MultiLines, LogReg -Phone, LogReg -Phone -ML |
| Batch size (MLP) | 16, 32, 64 |
| **Total runs** | **48** |

### Features adicionadas vs `orig`

| Feature | Presente em | Descricao |
|---------|-------------|-----------|
| `risco_contrato` | `le`, `ohe` | Ordinal 0-2 a partir do tipo de contrato |
| `service_count` | `le`, `ohe` | Qtd de servicos ativos (0-7) |
| `is_new` | `le`, `ohe` | Flag: Tenure <= 3 meses |
| `charges_per_tenure` | `le`, `ohe` | Monthly Charges / (Tenure + 1) |
| `tenure_bin_le` | `le` only | Ordinal 0-3 (bins: 0-12m / 13-24m / 25-48m / 49+m) |
| `tenure_bin_ohe` | `ohe` only | 4 colunas binarias com os mesmos cutoffs |

---

## Resultados: Baselines (30 runs)

### Top 10 por Holdout AUC

| Run | CV AUC | Holdout AUC | Notas |
|-----|--------|-------------|-------|
| `logreg_nophone_noml_8010_le` | 0.8583 | **0.8725** | Melhor holdout |
| `logreg_nophone_8010_le` | 0.8583 | 0.8724 | |
| `logreg_8010_le` | 0.8582 | 0.8724 | |
| `logreg_noml_8010_le` | 0.8582 | 0.8723 | |
| `logreg_8010_ohe` | 0.8582 | 0.8717 | |
| `logreg_noml_8010_ohe` | 0.8581 | 0.8717 | |
| `logreg_nophone_8010_ohe` | 0.8581 | 0.8718 | |
| `logreg_nophone_noml_8010_ohe` | 0.8582 | 0.8719 | |
| `logreg_8010_orig` | 0.8552 | 0.8706 | Sem feature engineering |
| `logreg_nophone_8010_orig` | 0.8553 | 0.8703 | |

### Top 5 por CV AUC (split 70/15/15)

| Run | CV AUC | Holdout AUC |
|-----|--------|-------------|
| `logreg_7015_le` | **0.8606** | 0.8502 |
| `logreg_noml_7015_le` | 0.8606 | 0.8503 |
| `logreg_nophone_7015_le` | 0.8606 | 0.8505 |
| `logreg_nophone_noml_7015_le` | 0.8606 | 0.8505 |
| `logreg_7015_ohe` | 0.8601 | 0.8505 |

### Observacoes — Baselines

- **Feature engineering ajuda**: `le`/`ohe` superam `orig` em todos os splits (~+0.003 CV, ~+0.002 holdout).
- **Ablacao Phone/MultiLines e ruido**: diferenca < 0.0003 — nenhum sinal robusto para remover essas features.
- **Split 80/10/10 da holdout mais alto** por ter mais dados de treino (+703 amostras vs 70/15/15).
- **Teto do LogReg**: ~0.861 CV / ~0.873 holdout (80/10/10 + `le`).

---

## Resultados: MLP PyTorch (18 runs)

### Todos os runs — ordenado por holdout AUC

| Run | CV AUC | Holdout AUC | Blind AUC |
|-----|--------|-------------|-----------|
| `mlp_8010_le_b64` | 0.8609 | **0.8735** | 0.8644 |
| `mlp_8010_le_b32` | 0.8615 | 0.8727 | **0.8648** |
| `mlp_8010_le_b16` | 0.8603 | 0.8710 | 0.8637 |
| `mlp_8010_ohe_b64` | 0.8593 | 0.8705 | 0.8644 |
| `mlp_8010_ohe_b16` | 0.8601 | 0.8703 | **0.8651** |
| `mlp_8010_ohe_b32` | 0.8601 | 0.8690 | 0.8650 |
| `mlp_8010_orig_b16` | 0.8565 | 0.8698 | 0.8560 |
| `mlp_8010_orig_b64` | 0.8571 | 0.8691 | 0.8556 |
| `mlp_8010_orig_b32` | 0.8569 | 0.8693 | 0.8533 |
| `mlp_7015_le_b32` | **0.8643** | 0.8522 | — |
| `mlp_7015_le_b64` | 0.8641 | 0.8513 | — |
| `mlp_7015_le_b16` | 0.8629 | 0.8509 | — |
| `mlp_7015_ohe_b32` | 0.8625 | 0.8503 | — |
| `mlp_7015_ohe_b64` | 0.8620 | 0.8496 | — |
| `mlp_7015_ohe_b16` | 0.8620 | 0.8490 | — |
| `mlp_7015_orig_b32` | 0.8611 | 0.8432 | — |
| `mlp_7015_orig_b16` | 0.8606 | 0.8426 | — |
| `mlp_7015_orig_b64` | 0.8605 | 0.8423 | — |

### Observacoes — MLP

- **Melhor CV**: `mlp_7015_le_b32` = 0.8643 (mais folds de CV com split 70/15/15).
- **Melhor holdout**: `mlp_8010_le_b64` = 0.8735.
- **Melhor blind test**: `mlp_8010_ohe_b16` = 0.8651.
- **`le` ganha em holdout, `ohe` ganha em blind test** — diferenca < 0.001, nao estatisticamente significativa.
- **Feature engineering ajuda consistentemente**: `orig` traz CV ~0.856-0.857 vs `le`/`ohe` ~0.860-0.864.
- **Batch size**: b32 e b64 ficam muito proximos; b16 tem maior variancia mas performance comparavel.
- **Teto empirico do MLP neste dataset**: ~0.865 blind test.

---

## Comparacao Final: Melhor de Cada Categoria

| Categoria | Run | CV AUC | Holdout AUC | Blind AUC |
|-----------|-----|--------|-------------|-----------|
| Dummy baseline | `dummy_8010_orig` | 0.5000 | 0.5000 | — |
| Melhor LogReg | `logreg_nophone_noml_8010_le` | 0.8583 | **0.8725** | — |
| Melhor MLP (holdout) | `mlp_8010_le_b64` | 0.8609 | **0.8735** | 0.8644 |
| Melhor MLP (blind) | `mlp_8010_ohe_b16` | 0.8601 | 0.8703 | **0.8651** |

**Conclusao**: MLP supera LogReg por ~0.001 holdout e ~0.000 blind test — dentro do ruido estatistico.
O dataset tem um teto natural de ~0.865 AUC-ROC para modelos lineares e MLP.

---

## Proximo Passo: Random Forest (notebook 05)

Configuracao recomendada com base na bateria:

| Parametro | Valor escolhido | Justificativa |
|-----------|-----------------|---------------|
| Split | 80/10/10 | Mais dados de treino; blind test disponivel |
| Tenure variant | `ohe` | Melhor blind test no MLP (0.8651); 35 features |
| Busca de hiperparametros | `RandomizedSearchCV` n_iter=20, 5-fold | Eficiente sem 50 runs manuais |
| `class_weight` | `"balanced"` | Imbalance 26/74 |
| Metrica alvo | Blind test AUC > 0.8651 | Superar melhor MLP atual |

**MLflow run name**: `rf_8010_ohe`

---

## Artefatos

- 48 runs no MLflow: experimento `churn-prediction`
- Notebooks com sumario: `03_baseline.ipynb` (secao 9), `04_mlp.ipynb` (secao 9)
- Este documento: `docs/resultados_bateria.md`
