# ML Canvas — Churn Prediction (Telco)

> Documento finalizado. Todas as decisões de feature selection, preprocessing,
> modelagem e threshold foram resolvidas e estão implementadas.
> Referências cruzadas: [`src/churn/data/preprocessing.py`](../src/churn/data/preprocessing.py),
> [`MODEL_CARD.md`](../MODEL_CARD.md), [`docs/architecture.md`](architecture.md).

| Item | Valor de referência |
|---|---|
| Tipo de tarefa | Classificação binária supervisionada |
| Modelo principal | MLP (PyTorch), 2 camadas ocultas |
| Métrica técnica primária | ROC-AUC (alvo ≥ 0,80) |
| Métrica de negócio | Custo total esperado (FP × R$50 + FN × R$500) |
| SLO de latência | P95 < 100 ms por predição na API |
| SLO de disponibilidade | ≥ 99,5% |
| Cadência de retreino | Trimestral, ou sob alerta de drift |
| Seed de reprodutibilidade | `SEED = 42` (em `src/churn/config.py`) |

---

## 1. Proposta de Valor

Operadora de telecomunicações enfrenta churn em ritmo elevado e perde
receita recorrente. O modelo entrega, para cada cliente ativo, uma
**probabilidade de cancelamento no próximo período**, permitindo ao
time de retenção agir de forma proativa — antes do cancelamento
acontecer — e priorizar esforço onde o custo esperado de inação é maior.

Sucesso = **redução do custo total de churn** (cliente perdido + ação
desnecessária), não simplesmente acertar a classificação.

---

## 2. Stakeholders e Usuários

| Papel | Como interage com o sistema |
|---|---|
| Diretoria comercial | *Sponsor.* Define meta de redução de churn e orçamento de retenção. Consome KPIs agregados. |
| Time de retenção | *Usuário primário.* Recebe lista priorizada de clientes em risco e executa as ações (oferta, contato, desconto). |
| Time de CRM / canais | Operacionaliza o contato (e-mail, ligação, push) na ferramenta atual. |
| Engenharia de dados / MLOps | Mantém pipeline, monitoramento e retreino. |
| Cliente final | Impactado indiretamente; recebe ofertas de retenção quando classificado como risco. |

---

## 3. Tarefa de ML

- **Tipo:** classificação binária supervisionada.
- **Variável-alvo:** `Churn Value` (1 = cancelou no trimestre, 0 = permaneceu).
- **Saída do modelo:** `P(churn = 1)` ∈ [0, 1].
- **Decisão downstream:** comparar a probabilidade contra um *threshold*
  calibrado por análise de custo e classificar em **risco baixo / médio / alto**.

---

## 4. Fonte de Dados

- **Dataset:** Telco Customer Churn (IBM), versão pública.
- **Volume:** 7.043 observações × 33 colunas.
- **Granularidade:** uma linha por cliente, snapshot trimestral.
- **Localização no repo:** `data/raw/raw_data.xlsx` (dicionário em
  [`docs/data_description.md`](data_description.md)).
- **Class balance esperado:** ~26% de churn — desbalanceado, exige
  tratamento (`class_weight` no baseline, `pos_weight` no MLP).

**Limitações conhecidas da fonte:**
- Snapshot único (não há série temporal por cliente).
- Distribuição geográfica concentrada (Califórnia, EUA) — generalização
  para outras regiões/operadoras não está validada.
- Algumas colunas são derivadas do próprio target e devem ser **removidas
  antes do treino** para evitar vazamento (ver seção 5).

---

## 5. Features (decisões finais — pós-EDA)

Decisões justificadas em [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb)
(seção 11) e implementadas em
[`src/churn/data/preprocessing.py`](../src/churn/data/preprocessing.py).
Política mantida: **nenhuma feature foi descartada sem evidência empírica**
(correlação ≈ 0 ou redundância semântica).

### 5.1 Pipeline final — 20 features de entrada → 35 de saída (variante `ohe`)

Variante canônica de produção: `tenure_variant="ohe"` (ADR-010).

| Grupo | Colunas | Tratamento |
|---|---|---|
| Numéricas (4) | `Tenure Months`, `Monthly Charges`, `Total Charges`, `CLTV` | `StandardScaler` |
| Binárias (13) | `Gender`, `Senior Citizen`, `Partner`, `Dependents`, `Phone Service`, `Paperless Billing`, `Online Security`, `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`, `Multiple Lines` | `OneHotEncoder(drop="if_binary")` → 1 col / feature |
| Multi-classe (3) | `Contract`, `Internet Service`, `Payment Method` | `OneHotEncoder` (mantém todos os níveis) → 10 cols totais |
| Engineered numéricas (4) | `risco_contrato`, `service_count`, `is_new`, `charges_per_tenure` | `StandardScaler` |
| Engineered categórica (1) | `tenure_bin_ohe` | `OneHotEncoder` com categorias fixas → 4 colunas (`0-12m`, `13-24m`, `25-48m`, `49+m`) |

Total na matriz de entrada do modelo: **35 features**. Nenhum NaN na saída; pipeline é
determinístico dado `SEED = 42`.

### 5.2 Limpeza determinada pela EDA

- **`Total Charges` (object → float64):** 11 linhas têm `' '`. Todas com
  `Tenure Months == 0` e `Churn Value == 0`. **Decisão:** imputar `0.0`
  (clientes novos ainda sem cobrança total).
- **Colapso de `"No internet service"` em 6 colunas** (`Online Security`,
  `Online Backup`, `Device Protection`, `Tech Support`, `Streaming TV`,
  `Streaming Movies`) e de `"No phone service"` em `Multiple Lines`,
  ambos para `"No"`. Razão: a informação "sem internet/telefone" já está
  em `Internet Service` / `Phone Service` — replicá-la em 7 dummies cria
  colinearidade e infla peso desse mesmo sinal nos modelos lineares.

### 5.3 Descartes (12 colunas) — todos com evidência

| Categoria | Colunas | Evidência |
|---|---|---|
| Vazamento de target | `Churn Label`, `Churn Score`, `Churn Reason` | `Churn Label` é duplicata 1:1; `Churn Score` tem AUC isolada **0,9417** (gerado por modelo IBM SPSS sobre o próprio target); `Churn Reason` é 100% nulo nos não-churners. |
| Identificadores e constantes | `CustomerID`, `Count` | `CustomerID` é único por linha; `Count` é constante = 1. |
| Geográficas | `Country`, `State`, `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | `Country` e `State` têm valor único (`United States`, `California`); `City` tem 1.129 categorias com churn rate sem padrão sistemático; `Zip Code`/`Latitude`/`Longitude` têm \|Pearson\| e \|Spearman\| < 0,01 com o target. |

### 5.4 Mantidas com sinal fraco — justificativas explícitas

- **`CLTV`** (AUC isolada = **0,58**) — sinal fraco mas legítimo, não é
  vazamento. Mantida.
- **`Gender`** (Cramér's V = **0,008**, Female 26,92% vs Male 26,16%) —
  praticamente ortogonal ao target. **Mantida pelo Model Card** (fase 5):
  é necessária pra reportar performance diferencial por gênero, mesmo
  que o modelo possa ignorá-la.
- **`Phone Service`** (Cramér's V = 0,01) e **`Multiple Lines`** (0,04) —
  sinais fracos mas não-zero. **Mantidas após ablation 2×2 da fase 2.4**
  (matriz Phone × Multiple Lines em 4 LogRegs + dummy de sanidade): os 4
  LogRegs são estatisticamente indistinguíveis (spread em ROC-AUC `~0,0006`
  ≪ desvio padrão CV `~0,0049`), e a célula de drop conjunto perde em
  todas as 4 métricas comparadas (ROC-AUC, PR-AUC, holdout val ROC-AUC,
  holdout val PR-AUC). Empate técnico → ADR-005 default ("nada é
  descartado sem evidência") prevalece, sem abrir novo ADR. Análise
  completa em [`notebooks/03_baseline.ipynb`](../notebooks/03_baseline.ipynb)
  §7–§8.

### 5.5 Top sinais (ranking unificado, sem leakers)

| # | Feature | Tipo | Score |
|---|---|---|---|
| 1 | **Contract** | cat | 0,4101 (Cramér's V) |
| 2 | **Tenure Months** | num | 0,3671 (max\|Pearson, Spearman\|) |
| 3 | Online Security | cat | 0,3474 |
| 4 | Tech Support | cat | 0,3429 |
| 5 | Internet Service | cat | 0,3225 |
| 6 | Payment Method | cat | 0,3034 |
| 7–10 | Online Backup, Device Protection, Dependents, Total Charges | mix | 0,23–0,29 |

---

## 6. Métricas

### Técnicas (offline, conjunto de validação/teste)

| Métrica | Por quê | Alvo |
|---|---|---|
| **ROC-AUC** | Métrica primária; robusta a threshold e desbalanceamento moderado. | ≥ 0,80 |
| **PR-AUC** | Mais sensível que ROC-AUC sob desbalanceamento; foca na classe positiva. | ≥ 0,69 (MLP deploy: 0,691 no val holdout) |
| **F1 / Precision / Recall** | Análise por threshold; trade-off operacional. | Reportar curva |
| **Matriz de confusão** | Suporta análise de custo. | — |

### Negócio

- **Custo total esperado** = `FP × C_FP + FN × C_FN`, com
  `C_FP = R$ 50` (ação de retenção desnecessária),
  `C_FN = R$ 500` (cliente perdido sem intervenção).
- O **threshold ótimo** é aquele que minimiza esse custo no conjunto
  de validação (não no teste).
- Reportar também: **redução estimada de churn** vs. cenário sem modelo
  (assumindo conversão `r` da ação de retenção, parametrizada).

---

## 7. SLOs (Service Level Objectives)

| Dimensão | Alvo |
|---|---|
| Latência por predição (API, P95) | < 100 ms |
| Disponibilidade do endpoint | ≥ 99,5% |
| ROC-AUC em produção (rolling 30d) | ≥ 0,75 *(margem vs. baseline offline)* |
| Drift de input (PSI por feature) | < 0,2 (verde), 0,2–0,3 (amarelo), > 0,3 (alerta) |

Deploy alvo: **inferência real-time** — ações de retenção dependem de
janela curta após sinal de risco. Batch noturno foi descartado por não
atender o caso de uso (justificativa em `docs/architecture.md`, fase 5).

---

## 8. Avaliação Offline

- **Split estratificado 80 / 10 / 10** (treino / validação / teste),
  preservando proporção de `Churn Value`. Supersede o split inicial 70/15/15 — ver ADR-009.
  Resultado: train=5.633, val=705, test=705 (churn rate ~26,5% em todos).
- **Validação cruzada estratificada 5-fold** nos baselines (dentro do conjunto de treino).
- **Conjunto de teste é tocado uma única vez**, ao final, para reportar
  performance externa (blind test).
- **Reprodutibilidade:** `SEED = 42` em numpy, sklearn, torch (CPU + CUDA).

---

## 9. Construção dos Modelos

Métricas completas no [MODEL_CARD.md](../MODEL_CARD.md). Resumo de performance no val holdout (threshold 0,50):

| Modelo | Configuração | Acc | Prec | Rec | F1 | ROC-AUC | PR-AUC | Blind AUC |
|---|---|---|---|---|---|---|---|---|
| `DummyClassifier` | `most_frequent` | 0,735 | 0,000 | 0,000 | 0,000 | 0,500 | 0,265 | 0,500 |
| `LogisticRegression` | `class_weight=balanced`, C=1.0, variante `le` | 0,786 | 0,566 | 0,829 | 0,672 | **0,873** | 0,697 | 0,861 |
| `RandomForestClassifier` | `n_estimators=300`, `max_depth=10`, variante `orig` | 0,784 | 0,569 | 0,775 | 0,656 | 0,870 | 0,678 | 0,861 |
| **MLP PyTorch** | BatchNorm→64→32→1, Adam lr=1e-3, `pos_weight=2.87`, batch=16, variante `ohe` | 0,783 | 0,560 | **0,845** | **0,674** | 0,870 | 0,691 | **0,865** |

Pré-processamento via `ColumnTransformer` (`StandardScaler` em
numéricas + `OneHotEncoder` em categóricas), serializado com `joblib`.

**Tracking:** todos os experimentos no MLflow, experimento único
`churn-prediction`, com params, métricas, modelo serializado e matriz
de confusão como artefato.

---

## 10. Predições em Produção

- **Modo:** real-time, single prediction por request.
- **Endpoint:** `POST /predict` (FastAPI), validação via Pydantic v2.
- **Resposta:** `churn_probability`, `churn_prediction` (bool a partir
  do threshold calibrado), `risk_level` (low/medium/high),
  `model_version`.
- **Health check:** `GET /health` retorna status, versão do modelo e
  timestamp.

---

## 11. Decisões e Ações Acionáveis

| Faixa de probabilidade | Risco | Ação proposta |
|---|---|---|
| `< t_low` | Baixo | Nenhuma ação ativa; manter no funil regular. |
| `t_low ≤ p < t_high` | Médio | Comunicação de valor (e-mail / push), oferta soft. |
| `≥ t_high` | Alto | Contato ativo do time de retenção, oferta dirigida. |

`t_low` e `t_high` são **calibrados** pelo time de retenção a partir do
*budget* mensal de ações × custo esperado por faixa. Default técnico definido
pela análise de custo (fase 3): **`t_high` = 0,27** (minimiza FP×R$50 + FN×R$500
no val holdout — custo R$16.050 vs R$20.700 no threshold padrão 0,50).

---

## 12. Monitoramento e Aprendizado Contínuo

| Sinal | Como medir | Alerta |
|---|---|---|
| **Data drift** (input) | PSI por feature, semanal | PSI > 0,3 em ≥ 3 features |
| **Prediction drift** | Distribuição da prob. predita | Shift > 10pp na média mensal |
| **Performance real** | ROC-AUC rolling 30d quando o ground truth chega | < 0,75 |
| **Latência** | P95 do endpoint | > 100 ms por > 5 min |

Plano detalhado em `docs/monitoring_plan.md` (fase 5).

**Política de retreino:**
- **Programada:** trimestral.
- **Reativa:** ao bater alerta de drift ou queda de AUC sustentada.

---

## 13. Riscos, Hipóteses e Considerações Éticas

- **Concept drift:** mudanças no mercado (concorrência, preço,
  pandemia) podem invalidar o modelo rapidamente — mitigado pelo
  monitoramento e retreino.
- **Viés por subgrupo:** análise obrigatória de performance diferencial
  por gênero, faixa etária (`Senior Citizen`) e tipo de contrato no
  Model Card. Diferenças sistemáticas precisam ser reportadas, mesmo
  quando não houver ação imediata. Notar que `Gender` foi
  **deliberadamente mantida** no input (Cramér's V = 0,008, sem sinal
  preditivo confirmado) justamente para viabilizar essa medição —
  removê-la antes do treino impediria comparar AUC por gênero.
- **Uso indevido:** o score **não deve** ser usado para precificação
  discriminatória ou redução de qualidade de atendimento de quem é
  classificado como "perdido". Uso autorizado: priorizar ações
  positivas de retenção.
- **Generalização limitada:** dataset cobre uma operadora específica
  em uma região específica; transferir o modelo para outro contexto
  exige revalidação completa.
