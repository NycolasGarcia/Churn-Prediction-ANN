# Model Card — Churn Prediction MLP

> Documento preliminar — será expandido na Fase 5 com análise de viés por
> subgrupo, métricas de produção e plano de retreino definitivo.

---

## 1. Informações Básicas

| Campo | Valor |
|---|---|
| **Nome** | `mlp_8010_ohe_b16` |
| **Versão** | 1.0 (Fase 3) |
| **Tipo** | Rede neural MLP — classificação binária |
| **Framework** | PyTorch 2.x |
| **Data de treino** | Maio 2026 |
| **Dataset** | Telco Customer Churn (IBM) — 7.043 clientes |
| **MLflow run** | `mlp_8010_ohe_b16` (experimento `churn-prediction`) |

---

## 2. Uso Pretendido

**Casos de uso:**
- Classificar clientes de telecomunicações por risco de cancelamento
- Priorizar ações de retenção proativas (campanhas, ofertas, contato)
- Estimar custo esperado de churn por cohort via threshold ajustável

**Casos de não-uso:**
- Não usar como único critério de demissão ou penalização de clientes
- Não generalizar para operadoras com perfil demográfico muito diferente
  (dataset é de empresa norte-americana; contexto BR pode divergir)
- Não usar sem retreino após 6+ meses de dados novos (data drift esperado)

---

## 3. Arquitetura

```
Input (35 features, ohe+tenure_bins)
  → BatchNorm1d
  → Linear(35 → 64) → ReLU → Dropout(0.3)
  → Linear(64 → 32) → ReLU → Dropout(0.2)
  → Linear(32 → 1)  → Sigmoid
```

| Hiperparâmetro | Valor |
|---|---|
| Optimizer | Adam, lr=1e-3 |
| Loss | BCEWithLogitsLoss, pos_weight=2.87 (balanceamento) |
| Batch size | 16 |
| Early stopping | patience=10, monitor=val_loss |
| Max epochs | 100 |
| Seed | 42 |

---

## 4. Performance

Split canônico **80/10/10** (seed=42). Threshold de deploy: **0,27**
(minimiza custo de negócio: FP=R$50, FN=R$500).

### 4.1 Métricas principais

| Conjunto | Threshold | Accuracy | Precision | Recall | F1 | ROC AUC | PR AUC | Log Loss |
|---|---|---|---|---|---|---|---|---|
| Val holdout | 0,50 | 0,783 | 0,560 | 0,845 | 0,674 | 0,870 | 0,691 | 0,443 |
| Val holdout | 0,27 | 0,685 | 0,455 | 0,941 | 0,613 | 0,870 | 0,691 | 0,443 |
| **Blind test** | 0,50 | — | — | — | — | **0,865** | — | — |

> ROC AUC e PR AUC são independentes do threshold.

### 4.2 Comparativo com modelos candidatos (val holdout, ROC AUC)

| Modelo | ROC AUC | Blind AUC | Delta vs MLP |
|---|---|---|---|
| Dummy | 0,500 | — | −0,365 |
| LogReg melhor | **0,873** | — | +0,003 |
| RF melhor (v2) | 0,870 | 0,861 | −0,005 |
| **MLP (este modelo)** | 0,870 | **0,865** | referência |

### 4.3 Análise de custo

No val holdout, o threshold 0,27 reduz o custo total estimado em
aproximadamente **38%** versus threshold=0,50.

| Threshold | Custo estimado (val) |
|---|---|
| 0,50 | R$ ~55.000 |
| **0,27** | **R$ ~34.000** |
| Ótimo (curva) | R$ ~33.000 |

Análise completa em [04_mlp.ipynb §6](notebooks/04_mlp.ipynb) e
[05_rfm.ipynb §7](notebooks/05_rfm.ipynb).

---

## 5. Limitações

- **Distribuição geográfica:** dataset de empresa norte-americana; padrões de
  churn podem diferir em operadoras brasileiras (contratos, regulação, ticket
  médio).
- **Período fixo:** dados de um período estático; sem sazonalidade ou tendência
  temporal modelada.
- **Features ausentes:** não inclui dados de uso de rede, reclamações ao SAC
  ou NPS — variáveis potencialmente preditivas.
- **Teto de AUC:** o dataset (~7k registros, 35 features) tem um teto natural
  em torno de 0,87 AUC para arquiteturas tabulares convencionais; ganhos
  adicionais exigiriam novas fontes de dados.

---

## 6. Análise de Viés

> **Seção a expandir na Fase 5** com métricas por subgrupo calculadas
> programaticamente.

Features demográficas presentes no dataset: `Gender`, `Senior Citizen`,
`Partner`, `Dependents`.

Análise preliminar qualitativa (a ser quantificada):
- O modelo não usa `Gender` como feature direta (removida no pré-processamento
  por baixa correlação com churn). Bias indireto via proxies não foi avaliado.
- Clientes sênior (8,5% do dataset) têm taxa de churn mais alta — o modelo
  pode calibrar de forma diferente para este subgrupo.
- Clientes sem dependentes e sem parceiro têm perfil de churn diferente;
  o modelo captura isso via features correlacionadas.

**TODO (Fase 5):** Calcular ROC AUC, F1 e taxa de falsos negativos
separadamente para `Gender`, `Senior Citizen` e faixas de `tenure`.

---

## 7. Cenários de Falha

| Cenário | Impacto | Mitigação |
|---|---|---|
| Clientes com tenure muito alto (> 60 meses) | Pouca representação no treino; modelo pode sub-estimar churn tardio | Monitorar recall neste segmento |
| Contrato month-to-month + novo cliente | Alta probabilidade de FP; ação de retenção desnecessária | Custo por FP baixo (R$50), tolerável |
| Mudança de produto/oferta pela operadora | Distribuição de features muda (data drift) | Retreino periódico (ver §8) |
| Input fora do schema esperado | Erro na API (422 Unprocessable Entity) | Validação Pydantic v2 no endpoint |

---

## 8. Plano de Monitoramento

> Detalhamento completo em [docs/monitoring_plan.md](docs/monitoring_plan.md).

| Sinal | Frequência | Alerta |
|---|---|---|
| ROC AUC em dados de produção | Mensal | Δ > 0,02 abaixo do baseline |
| Taxa de churn real vs predita | Mensal | Divergência > 5 p.p. |
| PSI das features numéricas | Mensal | PSI > 0,2 em qualquer feature |
| Latência do endpoint `/predict` | Contínua | P95 > 100ms |

**Frequência de retreino sugerida:** semestral, ou imediatamente se qualquer
alerta de drift for acionado.
