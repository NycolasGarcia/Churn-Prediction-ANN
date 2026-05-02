# Model Card — Churn Prediction MLP

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

Métricas calculadas no **blind test set** (n=705, seed=42, split 80/10/10)
com o threshold de deploy **0,27**. Script de referência: `scripts/bias_analysis.py`.

### 6.1 Resultados por subgrupo

| Subgrupo | N | Churners | ROC-AUC | F1 | FNR |
|---|---|---|---|---|---|
| **Overall** | 705 | 187 | **0,865** | 0,607 | 0,070 |
| Gender — Male | 352 | 95 | **0,881** | 0,631 | 0,063 |
| Gender — Female | 353 | 92 | 0,850 | 0,584 | 0,076 |
| Senior Citizen — Não | 591 | 133 | **0,867** | 0,567 | 0,098 |
| Senior Citizen — Sim | 114 | 54 | 0,778 | **0,720** | **0,000** |
| Tenure 0–12 meses | 206 | 100 | 0,778 | **0,690** | **0,010** |
| Tenure 13–36 meses | 186 | 52 | **0,865** | 0,607 | 0,096 |
| Tenure 37–60 meses | 159 | 23 | 0,771 | 0,409 | 0,174 |
| Tenure 61+ meses | 154 | 12 | **0,937** | 0,474 | 0,250 |

> **FNR** (False Negative Rate) = proporção de churners que o modelo classifica
> como não-churn. Custo de negócio: cada FN vale R$500 em perda.

### 6.2 Interpretação e achados

**Gênero (Δ AUC = 0,031):**
O modelo discrimina melhor entre clientes do sexo masculino (AUC 0,881) do que
feminino (0,850). O gap é moderado e não surpreende dado que `Gender` tem
sinal praticamente nulo na EDA (Cramér's V = 0,008) — a diferença emerge de
correlações indiretas com features de serviço e contrato, não de sinal direto
de gênero. Nenhum viés sistêmico identificado que justifique ação corretiva.

**Clientes sênior (Δ AUC = 0,089 — gap mais significativo):**
O segmento sênior (n=114, 16% do test set) tem AUC notavelmente inferior
(0,778 vs. 0,867). O FNR=0,000 parece positivo (nenhum churner sênior
é perdido), mas é artefato do threshold baixo (0,27) combinado com a alta
taxa de churn do segmento (47% vs. 22% dos não-sênior): o modelo simplesmente
classifica quase todos os sênior como churn. Isso infla FP neste subgrupo.
**Recomendação:** monitorar taxa de FP em sênior na produção; considerar
threshold diferenciado em futuras versões.

**Faixas de tenure — padrão não-linear:**

| Faixa | Insight |
|---|---|
| 0–12m | AUC 0,778 mas FNR quase zero — modelo é agressivo (muitos FP), adequado dado o alto custo de FN nesta faixa de maior churn |
| 13–36m | Pior performance; mistura de segmentos com sinais menos distintos |
| 37–60m | FNR em ascensão (0,174) — clientes maduros que decidem cancelar têm comportamento difícil de capturar |
| 61+m | AUC mais alto (0,937) mas FNR=0,250 — poucos churners (n=12); quando ocorre, é difícil prever; segmento requer monitoramento especial |

**Conclusão:** O modelo cumpre o SLO geral (AUC ≥ 0,80) para os principais
segmentos demográficos. O gap mais relevante está em **clientes sênior**
(AUC 0,778) e **tenure longa 37–60m** (AUC 0,771 + FNR alto). Ambos os
segmentos devem ser monitorados prioritariamente após o deploy.

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
