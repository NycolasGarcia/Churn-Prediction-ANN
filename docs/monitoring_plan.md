# Plano de Monitoramento

> **Documento vivo.** Esta versão é o **esqueleto do plano** — os sinais,
> thresholds e cadência refletem o que já foi decidido no
> [`ml_canvas.md`](ml_canvas.md). Itens marcados **[a definir na fase X]**
> dependem de entregas futuras (API, deploy, infra de telemetria) e serão
> preenchidos quando o contexto estiver maduro o suficiente.

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

**[a refinar na fase 5]** — definir tabela de junção entre predições
históricas e o evento real de churn, com SLA de chegada do GT.

---

## 7. Logs e dashboards

**[a definir na fase 4 — depende da decisão de deploy]**

- **Logs:** structured logging (JSON, `python-json-logger`) com
  `customer_id`, `model_version`, `churn_probability`, `latency_ms`,
  `request_id`. Retenção sugerida: **90 dias**.
- **Storage:** [a definir]. Opções: S3 + Athena, BigQuery, ELK.
- **Dashboards:** [a definir]. Opções: Grafana com data source local,
  CloudWatch, Datadog.

---

## 8. Itens explicitamente diferidos

| Item | Fase prevista | Motivo do diferimento |
|---|---|---|
| Estrutura exata dos logs (schema JSON) | 4 | Dependente do FastAPI e middleware |
| Implementação do PSI no código | 4–5 | Função utilitária + script de monitoria |
| Stack de monitoring (Grafana / Datadog / CloudWatch) | 5 | Depende do alvo de deploy |
| SLAs com time de retenção (response time pra alertas) | 5 | Negociação com stakeholder |
| Sistema de alerting concreto (PagerDuty / Slack / e-mail) | 5 | Depende da stack |
| Procedimento operacional para `champion–challenger` em produção | 5 | Depende da arquitetura de serving |
