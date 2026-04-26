# ML Canvas — Churn Prediction (Telco)

> Documento vivo. Algumas células dependem da EDA (fase 1.4) e da análise de
> custo (fase 3) e serão revisadas conforme as decisões forem sendo tomadas
> com base em dados reais. As seções marcadas com **[a confirmar pós-EDA]**
> contêm hipóteses de partida.

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

## 5. Features

**Categorias-base (a confirmar pós-EDA):**

| Grupo | Exemplos | Tratamento previsto |
|---|---|---|
| Demográficos | `Gender`, `Senior Citizen`, `Partner`, `Dependents` | One-hot / binário |
| Contrato e billing | `Contract`, `Paperless Billing`, `Payment Method`, `Tenure Months` | One-hot + StandardScaler em numéricos |
| Serviços contratados | `Phone Service`, `Internet Service`, `Online Security`, `Tech Support`, `Streaming TV`, etc. | One-hot |
| Faturamento | `Monthly Charge`, `Total Charges` | StandardScaler; investigar missing em `Total Charges` |

**Descartes prováveis (a confirmar com correlação na EDA):**

- **Identificadores e constantes:** `CustomerID`, `Count`.
- **Localização:** `Country`, `State`, `City`, `Zip Code`, `Lat Long`,
  `Latitude`, `Longitude` — provavelmente irrelevantes (todos no mesmo
  país/estado, alta cardinalidade em cidade). Investigar antes de
  descartar definitivamente.
- **Vazamentos diretos do target (remoção obrigatória):**
  - `Churn Label` — duplicata textual do target.
  - `Churn Score` — gerado por modelo IBM SPSS sobre o próprio target.
  - `Churn Reason` — só preenchido para quem já cancelou.
- **CLTV** — *[a confirmar pós-EDA]* potencialmente informativo, mas
  pode ser proxy do target dependendo de como foi calculado.

> **Política:** nenhuma feature é descartada sem antes inspecionar
> distribuição e correlação com o target. Justificativas ficam no
> notebook 02_data_prep.

---

## 6. Métricas

### Técnicas (offline, conjunto de validação/teste)

| Métrica | Por quê | Alvo |
|---|---|---|
| **ROC-AUC** | Métrica primária; robusta a threshold e desbalanceamento moderado. | ≥ 0,80 |
| **PR-AUC** | Mais sensível que ROC-AUC sob desbalanceamento; foca na classe positiva. | ≥ 0,60 *[a calibrar]* |
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

- **Split estratificado 70 / 15 / 15** (treino / validação / teste),
  preservando proporção de `Churn Value`. SEED fixa.
- **Validação cruzada estratificada 5-fold** nos baselines.
- **Conjunto de teste é tocado uma única vez**, ao final, para reportar
  performance externa.
- **Reprodutibilidade:** `SEED = 42` em numpy, sklearn, torch (CPU + CUDA).

---

## 9. Construção dos Modelos

| Modelo | Papel | Configuração base |
|---|---|---|
| `DummyClassifier(strategy="most_frequent")` | Piso absoluto | — |
| `LogisticRegression(class_weight="balanced")` | Baseline interpretável | C=1.0, l2 |
| **MLP PyTorch** | Modelo principal | BatchNorm → 64 → ReLU → Dropout(0.3) → 32 → ReLU → Dropout(0.2) → 1; Adam lr=1e-3; BCEWithLogitsLoss + `pos_weight`; batch=64; early stopping patience=10 |

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
*budget* mensal de ações × custo esperado por faixa. Default técnico:
`t_high` = threshold de menor custo total na validação.

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
  por gênero, faixa etária (Senior Citizen) e tipo de contrato no
  Model Card. Diferenças sistemáticas precisam ser reportadas, mesmo
  quando não houver ação imediata.
- **Uso indevido:** o score **não deve** ser usado para precificação
  discriminatória ou redução de qualidade de atendimento de quem é
  classificado como "perdido". Uso autorizado: priorizar ações
  positivas de retenção.
- **Generalização limitada:** dataset cobre uma operadora específica
  em uma região específica; transferir o modelo para outro contexto
  exige revalidação completa.
