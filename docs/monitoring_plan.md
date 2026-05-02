# Plano de Monitoramento

> **Modelo em produção:** `mlp_8010_ohe_b16` — ROC-AUC 0,865 (blind test),
> threshold de deploy **0,27** (minimiza custo FP=R$50 / FN=R$500).
> API: `POST /predict` em FastAPI, versão `1.0.0`.

---

## 1. Objetivo

Garantir que o modelo de churn em produção mantenha **performance
preditiva**, **relevância dos dados de entrada** e **SLOs de
operação** ao longo do tempo. Detectar regressões antes que impactem
decisões de negócio (campanhas de retenção, *churn budget* alocado).

Três tipos de degradação são monitorados:

1. **Operacional** — latência, disponibilidade, erros do endpoint.
2. **Estatística** — drift do input (PSI por feature) e drift da saída
   (distribuição de `P(churn)`).
3. **Preditiva** — performance real (ROC-AUC rolling) quando o ground
   truth se torna observável.

---

## 2. Sinais monitorados

| Categoria | Sinal | Como medir | Cadência | Owner |
|---|---|---|---|---|
| Operacional | Latência P95 do `POST /predict` | middleware de timing (fase 4) | tempo real | Engenharia |
| Operacional | Disponibilidade do endpoint | health check externo via `GET /health` | a cada 1 min | Engenharia |
| Operacional | Taxa de erro 5xx | logs estruturados | tempo real | Engenharia |
| Operacional | Volume diário | contagem de requests | diário | Engenharia |
| Estatística | Data drift (PSI por feature) | comparar produção × baseline do `train.parquet` | semanal | Data Science |
| Estatística | Prediction drift | quantis e média móvel da prob. predita | semanal | Data Science |
| Preditiva | ROC-AUC rolling | recalcular com ground truth observado | mensal | Data Science |
| Preditiva | Custo de churn evitado | `FN × R$500 + FP × R$50` na janela | mensal | Negócio |

---

## 3. Thresholds e níveis de alerta

Os limites abaixo refletem os SLOs declarados no Canvas (seção 7) e são
revisados após o baseline (fase 2) e o MLP final (fase 3).

| Sinal | 🟢 Verde | 🟡 Amarelo | 🔴 Vermelho |
|---|---|---|---|
| Latência P95 | < 100 ms | 100 – 150 ms | > 150 ms por > 5 min |
| Disponibilidade (rolling 7d) | ≥ 99,5% | 99 – 99,5% | < 99% |
| Erro 5xx | < 0,1% | 0,1 – 0,5% | > 0,5% por > 5 min |
| PSI por feature | < 0,2 | 0,2 – 0,3 | > 0,3 em ≥ 3 features |
| Prediction shift mensal | < 5 pp | 5 – 10 pp | > 10 pp |
| ROC-AUC rolling 30d | ≥ 0,80 | 0,75 – 0,80 | < 0,75 |

> **Notação:** `pp` = pontos percentuais. Os valores em vermelho disparam
> alerta com paging; amarelo gera aviso no canal de Data Science sem
> paging.

---

## 4. Playbook de resposta

| Cenário | Ação imediata | Investigação | Decisão de retreino |
|---|---|---|---|
| Latência > 150 ms sustentada | scale-up horizontal + alerta on-call | profiling do hot path; checar batching/cache; checar carga no DB | — |
| Erros 5xx > 0,5% | rollback se for deploy recente; senão investigar | log dive, RCA padrão | — |
| Drift PSI > 0,3 em ≥ 3 features | flag no dashboard, **sem rollback automático** | identificar quais features e a magnitude do shift; verificar coleta upstream | retreinar se confirmado e sustentado por ≥ 2 semanas |
| Prediction shift > 10 pp | revisar entrada (drift?) e segmentação atendida | comparar mix de clientes ativos | retreinar se causado por mudança real do dataset |
| ROC-AUC rolling < 0,75 | confirmar que o ground truth está completo e correto | possível bug de coleta de label, atraso, mudança de definição | retreinar com janela recente após confirmar a queda |
| Disponibilidade < 99% | alerta on-call + rollback se aplicável | RCA padrão, post-mortem em 48h | — |

---

## 5. Política de retreino

- **Programada:** **trimestral**, sobre janela de 12 meses anteriores
  (ou todo o histórico disponível, se < 12 meses em produção).
- **Reativa:** ao bater alerta de drift sustentado por **≥ 2 semanas**
  ou queda de ROC-AUC por **≥ 30 dias**.
- **Forçada:** após mudança contratual significativa do dataset
  upstream (ex: novo plano comercial, fusão).
- **Procedimento:**
  1. Snapshot do dataset atual + modelo em produção.
  2. Re-rodar pipeline (`02_data_prep`) e treinos (`03_baseline`,
     MLP) sobre janela atualizada.
  3. **Champion–challenger**: novo modelo só vai pra produção se
     superar o atual em ROC-AUC + custo total no novo holdout.
  4. Versionamento: `model_version` semver minor a cada retreino
     bem-sucedido (`1.0.x → 1.1.0`).

---

## 6. Coleta de ground truth

Modelo prevê churn no **próximo trimestre**. O ground truth fica
observável com **lag de 3 meses** após cada predição. Implicação:

- Performance preditiva em tempo real **não é mensurável**.
- Em até 3 meses após a predição, dependemos de **proxies**:
  drift de input, drift de prediction, comportamento agregado.
- ROC-AUC rolling 30d só passa a fazer sentido **3 meses após o início
  da operação**, e fica continuamente defasada por 3 meses.

**Tabela de junção sugerida:**

```
predictions_log (request_id, customer_id, timestamp, churn_probability, prediction)
    JOIN
churn_events (customer_id, cancellation_date)
    ON customer_id AND cancellation_date BETWEEN timestamp AND timestamp + 90 days
```

SLA de chegada do ground truth: 90 dias após a predição. ROC-AUC mensal
calculado sobre cohort com GT disponível (janela rolante).

---

## 7. Logs e dashboards

- **Logs:** structured logging via `logging` padrão do Python com campos
  `probability`, `prediction`, `threshold`, `latency_ms` emitidos pelo
  `LatencyLoggingMiddleware` e pelo endpoint `/predict`. Formato JSON.
  Retenção sugerida: **90 dias**.
- **Storage:** A definir conforme plataforma de deploy. Opções recomendadas:
  S3 + Athena (AWS), BigQuery (GCP), ou stack local ELK para demonstração.
- **Dashboards:** Grafana (open-source) com data source nos logs estruturados
  é a opção de menor custo para um setup de demonstração. Em produção real,
  CloudWatch (AWS) ou Cloud Monitoring (GCP) integram nativamente.

---

## 8. Subgrupos prioritários para monitoramento (fairness)

A análise de viés da Fase 5 (`scripts/bias_analysis.py`, `MODEL_CARD.md §6`)
identificou dois segmentos com performance abaixo da média que merecem
monitoramento diferenciado em produção:

| Segmento | AUC no test | Delta vs overall | FNR | Ação sugerida |
|---|---|---|---|---|
| Senior Citizen (Sim) | 0,778 | −0,087 | 0,000 | Monitorar FP — modelo tende a sobre-prever churn neste grupo |
| Tenure 37–60 meses | 0,771 | −0,094 | 0,174 | FNR alto — churners tardios são subdetectados; alerta especial |
| Tenure 61+ meses | 0,937 | +0,072 | 0,250 | Poucos casos (n=12); acompanhar conforme volume cresce |
