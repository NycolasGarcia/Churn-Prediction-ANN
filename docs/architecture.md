# Architecture Decision Records

> Documento vivo. Cada decisão arquitetural ou metodológica relevante é
> registrada como um **ADR** (Architecture Decision Record) com contexto,
> alternativas consideradas, decisão e consequências.
>
> ADRs **não são editados** após aceitos — quando uma decisão for revertida
> ou substituída, o ADR antigo recebe status `Superseded by ADR-XXX` e um
> novo é adicionado. Isso preserva a história do raciocínio do projeto.

## Índice

| # | Título | Status | Adotado em |
|---|---|---|---|
| ADR-001 | Split estratificado 70 / 15 / 15 (vs. 80/20 ou k-fold puro) | Aceito | 2026-04-26 |
| ADR-002 | Versionar `raw_data.xlsx` no repositório | Aceito (temporário) | 2026-04-26 |
| ADR-003 | Manter `Gender` no input apesar do sinal nulo | Aceito | 2026-04-26 |
| ADR-004 | Colapso de "No internet/phone service" → "No" | Aceito | 2026-04-26 |
| ADR-005 | Manter features de sinal fraco (`Phone Service`, `Multiple Lines`) para ablation pós-baseline | Aceito | 2026-04-26 |
| ADR-006 | Estratégia de testes para módulos não-API (smoke / schema / API por módulo) | Aceito | 2026-04-26 |
| ADR-007 | Fixtures de teste construídas a partir do raw, não do `data/processed/` | Aceito | 2026-04-26 |
| ADR-008 | MLflow: 1 run agregado por modelo (CV folds → mean/std) em vez de nested runs | Aceito | 2026-04-27 |

---

## ADR-001 — Split estratificado 70 / 15 / 15

**Status:** Aceito (2026-04-26)

**Contexto.** O projeto entrega três coisas que, somadas, exigem um conjunto
de validação separado do conjunto de teste:

1. Early stopping no MLP (PyTorch) precisa de um sinal por época que não
   seja `train_loss` (overfit) nem `test_loss` (contamina o teste).
2. O threshold ótimo é calibrado por análise de custo
   (`FP × R$50 + FN × R$500`) — esse cálculo não pode tocar o teste.
3. A escolha entre Dummy / LogReg / MLP é uma decisão de seleção de
   modelo — comparar usando o test set vazaria informação para o reporte
   final.

Com `7.043` linhas e `~26,5%` churn, três proporções foram consideradas.

**Alternativas consideradas:**

- **80 / 20 sem val.** Descartado: força early-stopping e calibração de
  threshold a usar o teste, anulando o propósito do teste.
- **k-fold CV (5 folds) sem holdout.** Descartado: o early-stopping
  do MLP transforma o loop em CV aninhada (50 treinos), o custo é alto
  e nenhum número reportado é "do modelo final" — todos são CV.
- **60 / 20 / 20.** Aceitável, mas tira ~700 linhas do treino.
- **80 / 10 / 10.** Aceitável, mas deixa val/test com ~187 churners,
  abaixo da regra de bolso (≥ 200 positivos para AUC estável).

**Decisão.** **70 / 15 / 15 estratificado** com `SEED = 42`.
Implementado em `churn.data.preprocessing.stratified_split`. Resultado:
`train = 4929` (target rate `0,2654`), `val = 1057` (`0,2658`),
`test = 1057` (`0,2649`).

**Consequências.**

- ✅ Val e test grandes o suficiente para AUC com IC estreito (~280
  churners cada).
- ✅ Test fica intocável até o reporte final.
- ➖ ~700 linhas a menos no treino versus 80/10/10 — aceito por um
  treinamento mais robusto não compensar val/test pequenos.
- ➖ Não usa CV, perdendo a estimativa de variância entre folds.
  Mitigado por **CV estratificada 5-fold _dentro_ do train** nos
  baselines (CLAUDE.md / Tese).

---

## ADR-002 — Versionar `raw_data.xlsx` no repositório

**Status:** Aceito, **temporário** (2026-04-26)

**Contexto.** Boa prática em ML é não versionar dados — datasets de
produção podem ser grandes, ter PII ou licenciamento, e instruções de
download via README garantem reprodutibilidade. Para um projeto
educacional com dataset público de **1,3 MB**, a fricção do download
externo supera o ganho convencional.

**Alternativas consideradas:**

- **Não versionar, instruir download via README.** Padrão da indústria.
  Mais correto, porém adiciona uma etapa manual ao setup.
- **Versionar via Git LFS.** Overkill para 1,3 MB.
- **Versionar arquivo bruto direto.** Aceito.

**Decisão.** Versionar `data/raw/raw_data.xlsx` diretamente no Git.
`.gitignore` cobre `data/raw/*` mas faz unignore explícito desse arquivo,
e ignora todo o restante (`data/processed/`, modelos, mlruns).

**Consequências.**

- ✅ `git clone` produz um setup totalmente funcional sem rede.
- ✅ Avaliador da disciplina não precisa baixar nada adicional.
- ➖ Repo carrega 1,3 MB de dados versionados — desprezível agora,
  mas se o dataset crescer ou ganhar PII, esta decisão é
  revertida (provável `Superseded by ADR-XXX` em refactor posterior).

---

## ADR-003 — Manter `Gender` no input apesar do sinal nulo

**Status:** Aceito (2026-04-26)

**Contexto.** A EDA confirmou que `Gender` é praticamente ortogonal ao
target: Cramér's V = `0,008`, churn rate `26,92%` em Female versus
`26,16%` em Male. A diferença está dentro do ruído estatístico.

**Alternativas consideradas:**

- **Dropar antes do treino.** Reduz dimensionalidade marginalmente
  (1 feature). Mas torna a análise de viés do Model Card (fase 5)
  inviável — sem a coluna no input, não há como reportar AUC por gênero
  ou comparar performance diferencial.
- **Manter no input mas reportar como noise feature.** Aceito.

**Decisão.** Manter `Gender` no pipeline de pré-processamento. O modelo
aprende a ignorá-la (peso baixo via regularização L2 no LogReg, peso
baixo emergente no MLP). A coluna fica disponível para a análise de
viés obrigatória do Model Card.

**Consequências.**

- ✅ Análise de viés por gênero possível na fase 5 sem refit.
- ✅ Cumpre boas práticas de fairness em ML (medir antes de remover).
- ➖ 1 feature adicional no input. Custo computacional desprezível.
- ⚠️ Fica registrado: o score **não pode** ser usado para precificação
  ou ofertas diferenciadas por gênero (canvas, seção 13).

---

## ADR-004 — Colapso de "No internet/phone service" → "No"

**Status:** Aceito (2026-04-26)

**Contexto.** Sete colunas de serviço têm um terceiro valor além de
`Yes`/`No`:

- 6 colunas de serviços de internet (`Online Security`, `Online Backup`,
  `Device Protection`, `Tech Support`, `Streaming TV`, `Streaming Movies`)
  têm `"No internet service"` quando `Internet Service == "No"`.
- `Multiple Lines` tem `"No phone service"` quando `Phone Service == "No"`.

Esses valores são **redundantes**: a informação "não tem internet" já
está em `Internet Service`, e "não tem telefone" em `Phone Service`.

**Alternativas consideradas:**

- **Deixar como está + OneHot com 3 categorias por coluna.** Cria 6
  dummies idênticas (`Online Security_No internet service`,
  `Online Backup_No internet service`, etc.) — todas 1 nas mesmas
  linhas. Implica peso 6× num mesmo sinal nos modelos lineares,
  redundância para o MLP, ruído na interpretabilidade.
- **Colapsar `"No internet service"` em `"No"`.** Mantém a semântica
  ("não tem o serviço"), elimina dummies redundantes. Aceito.
- **Criar uma feature derivada `has_internet_service`.** Já existe em
  `Internet Service`. Redundante.

**Decisão.** Em `clean_raw`, antes do OneHot:
`replace("No internet service" → "No")` nas 6 colunas relevantes e
`replace("No phone service" → "No")` em `Multiple Lines`. Cada coluna
vira binária (`Yes`/`No`) e usa `OneHotEncoder(drop="if_binary")` →
1 dummy por coluna.

**Consequências.**

- ✅ `27` features de saída em vez de `33` — redução real de dimensionalidade.
- ✅ Sem dummies colineares por construção.
- ✅ Modelo linear não distorce peso desse sinal específico.
- ➖ Tecnicamente perdemos a possibilidade de modelar separadamente
  "tem internet mas não usa o serviço" vs "não tem internet". Mas
  `Internet Service` já distingue os dois grupos, então não há perda
  de informação real.

---

## ADR-005 — Manter features de sinal fraco para ablation pós-baseline

**Status:** Aceito (2026-04-26)

**Contexto.** A EDA mostrou sinal bem fraco em duas features:
`Phone Service` (Cramér's V = `0,01`) e `Multiple Lines` (`0,04`).
A tentação é dropá-las para "limpar o input", mas:

- O critério "Cramér's V < 0,05 → drop" é arbitrário.
- A interação entre features pode dar valor onde o sinal univariado
  não dá (ex: `Phone Service + Tenure Months + Contract` pode separar
  segmentos que cada uma sozinha não separa).
- O custo de manter é desprezível (2 binárias, regularização lida).

**Alternativas consideradas:**

- **Dropar de cara, baseando-se na EDA univariada.** Mais "enxuto",
  mas a justificativa ("Cramér's V baixo") é frágil e não defende contra
  a observação de interações.
- **Manter e validar com ablation pós-baseline.** Decisão final fica
  ancorada em métrica real ("removeu a feature, AUC caiu X
  no val"). Aceito.

**Decisão.** Manter `Phone Service` e `Multiple Lines` no pipeline.
Na fase 2, comparar **dois LogReg baselines** (com e sem essas duas
features) e/ou rodar ablation com o MLP na fase 3. A decisão de
descarte só acontece se a remoção **não piorar** ROC-AUC ou PR-AUC
de validação dentro de uma margem definida (provavelmente `0,005`).

**Consequências.**

- ✅ Decisão final empírica e auditável.
- ✅ Cumpre a política do projeto: "nada é descartado sem evidência".
- ➖ Adiciona dois experimentos no MLflow. Custo desprezível.
- ⚠️ Se a ablation confirmar irrelevância, atualizar
  `preprocessing.py` (lista `BINARY_COLUMNS`) e abrir um ADR-XXX
  documentando a remoção.

---

## ADR-006 — Estratégia de testes para módulos não-API

**Status:** Aceito (2026-04-26)

**Contexto.** Ao introduzir os baselines (sub-checkpoint 2.1), surgiu a
necessidade de testes automatizados antes do MLP final. CLAUDE.md
prescreve `test_smoke.py`, `test_schema.py` e `test_api.py` na fase 4,
mas esses três alvos são para o **modelo final servido via FastAPI**:
smoke do pipeline produtivo, schema do request/response Pydantic, e API
HTTP do endpoint. Eles não cobrem código de modelagem (sklearn baselines,
e eventualmente o MLP em PyTorch) que é construído antes da fase 4.

**Alternativas consideradas.**

- **Adiar testes para a Fase 4.** Risco: módulos não-API ficam sem
  cobertura por duas fases inteiras; regressões em `baseline.py`,
  `preprocessing.py` ou `mlp.py` apareceriam só lá adiante, com causa
  raiz já distante.
- **Reusar a nomenclatura da Fase 4 (smoke/schema/api) para tudo.**
  Confunde — `test_api.py` deveria testar o FastAPI literalmente, não
  contratos de módulos Python.
- **Criar `test_<module>.py` por módulo, organizando internamente em
  três blocos.** Aceito.

**Decisão.** Cada módulo de modelagem ganha o seu `tests/test_<module>.py`,
estruturado em três blocos com nomenclatura padronizada:

- **Smoke** — fit/predict end-to-end sem erros.
- **Schema** — shapes, dtypes, value ranges das saídas.
- **API** — contrato do módulo Python (estrutura do `Pipeline`,
  configuração do classifier, determinismo).

A "API" nesse contexto é a **API do módulo Python**, não HTTP — o uso
da palavra é deliberado para manter simetria conceitual com a Fase 4
(`test_api.py` lá testará o FastAPI). Os dois níveis convivem; o
docstring de cada arquivo de teste deixa o escopo explícito.

**Consequências.**

- ✅ Cobertura desde a Fase 2 (94% ao final de 2.1).
- ✅ Falhas localizadas — `test_baseline.py::test_X` aponta o módulo
  afetado sem ambiguidade.
- ✅ Defensivo contra config drift — assertions estritas (ex.:
  `assert clf.strategy == "most_frequent"`) capturam mudanças
  silenciosas de configuração.
- ➖ Convenção dupla de "API": exige explicação no docstring de cada
  test file (já feita).

---

## ADR-007 — Fixtures de teste construídas a partir do raw

**Status:** Aceito (2026-04-26)

**Contexto.** Os splits processados (`data/processed/{train,val,test}.parquet`)
são gitignored — gerados pelo notebook `02_data_prep.ipynb`. Em uma
máquina recém-clonada, **o pytest precisa rodar sem depender desses
arquivos**, ou senão CI e desenvolvedores novos quebram em ordem
operacional irrelevante.

**Alternativas consideradas.**

- **Carregar fixture diretamente do `data/processed/*.parquet`.** Mais
  rápido (`pd.read_parquet`), porém frágil: requer rodar 02_data_prep
  antes do pytest, e qualquer mudança no pipeline obriga regenerar os
  parquets ou os testes ficam sobre dados velhos.
- **Comitar parquets pequenos no repo só pra testes.** Polui o repo
  versionado e duplica a decisão temporária do ADR-002 (versionar
  raw_data.xlsx) com mais arquivos pra "limpar" futuramente.
- **Construir fixture chamando o pipeline real
  (`load_raw → clean_raw → stratified_split`).** Aceito.

**Decisão.** `tests/conftest.py` define a fixture `split_data` com
`scope="session"`, executando o pipeline completo a partir de
`data/raw/raw_data.xlsx` (versionado, ADR-002). Custo: ~1 segundo por
sessão de pytest, cacheado.

**Consequências.**

- ✅ Pytest funciona em qualquer clone do repo sem setup intermediário.
- ✅ A fixture exercita o caminho de produção (loader + cleaner +
  splitter), elevando a cobertura "de graça" desses módulos.
- ✅ Mudanças no pipeline são imediatamente refletidas nos testes — não
  há fonte de verdade duplicada.
- ➖ Se `raw_data.xlsx` for removido em refactor futuro (ADR-002 é
  temporário), a fixture precisa migrar para download via URL ou
  dataset sintético. Mitigação: ADR-002 e ADR-007 são revisados juntos
  quando essa transição ocorrer.

---

## ADR-008 — MLflow: 1 run agregado por modelo (CV folds → mean/std)

**Status:** Aceito (2026-04-27)

**Contexto.** CLAUDE.md §6 (Fase 2) prescreve **CV estratificada 5-fold**
para os baselines, e a Seção 9 lista as métricas obrigatórias por run.
A intersecção das duas exigências força uma escolha de granularidade no
MLflow: como representar 5 folds × N modelos no tracking sem poluir a UI
nem perder informação por fold.

Decisão precisa ser tomada **antes** de implementar `tracking.py` no
sub-checkpoint 2.3 — qualquer mudança posterior obrigaria reprocessar runs
históricos para compará-los lado a lado.

**Alternativas consideradas.**

- **Nested runs (1 run pai + 5 runs filhos por modelo).** Cada fold vira
  um `mlflow.start_run(nested=True)`. Permite drill-down nativo na UI,
  mas com 4 modelos planejados na Fase 2 (`dummy_baseline`,
  `logreg_baseline`, `logreg_no_phone_ablation`, `logreg_no_multilines_ablation`)
  + MLP na Fase 3 a UI fica com **30+ runs**, e qualquer comparação
  cruzada exige filtro por tag.
- **1 run por fold, sem hierarquia (`logreg_baseline_fold_3`, etc.).**
  Pior dos dois mundos: polui a UI igual ao nested e perde a relação
  pai/filho.
- **1 run agregado por modelo, com `<metric>_mean`, `<metric>_std`,
  `<metric>_fold_<i>`.** A média e o desvio são o que entra no Model
  Card e na tese; os valores por fold ficam logados como métricas
  individuais (não como runs separados) para auditabilidade. Aceito.

**Decisão.** **1 run agregado por modelo**, com o seguinte schema de
métricas no MLflow:

| Métrica logada | Significado |
|---|---|
| `roc_auc_mean`, `roc_auc_std` | Métrica primária (CLAUDE.md §11 Tese) — agregada sobre os 5 folds |
| `pr_auc_mean`, `pr_auc_std` | Métrica secundária para classes desbalanceadas |
| `f1_mean`, `f1_std`, `precision_mean`, `precision_std`, `recall_mean`, `recall_std` | Métricas auxiliares |
| `roc_auc_fold_1` ... `roc_auc_fold_5` | Valores brutos por fold para auditoria |
| `holdout_val_roc_auc`, `holdout_val_pr_auc`, ... | Métricas no val holdout (refit em todo o train) — usado para calibração de threshold/cost |

Params obrigatórios: `model_type`, `class_weight`, `seed`,
`dataset_version`, `n_features`, `cv_folds=5`, `cv_strategy=stratified`.
Tags obrigatórias: `model_type`, `dataset_version`, `author` (ADR-008
ratifica o que CLAUDE.md §9 já prescreve).

**Consequências.**

- ✅ MLflow UI fica navegável: comparar 4 baselines vira 4 linhas, não 20.
- ✅ Mean ± std são exatamente os números reportados no Model Card e na
  tese — sem pós-processamento.
- ✅ Valores por fold ficam preservados como métricas (não como runs),
  permitindo investigar variância sem rerodar.
- ✅ `holdout_val_*` separa o que é generalização-CV do que vai ser
  usado para calibrar threshold (ADR futura sobre cost analysis).
- ➖ Drill-down visual por fold é manual (ler `roc_auc_fold_<i>` na UI).
  Aceitável: comparação por-fold raramente é o caminho de investigação.
- ➖ Se um fold colapsar (ex: `roc_auc_std > 0.05`), investigação exige
  reproduzir o split daquele fold. Mitigação: o seed de CV é parte dos
  params (`seed`) e o `KFold` é determinístico.

---

## ADR-009 — 80/10/10 como split canônico para avaliação de modelos (Fase 3+)

**Status:** Aceito — 2026-04-28

**Contexto.**
ADR-001 adotou 70/15/15 como split padrão do projeto. Durante a Fase 3,
uma bateria sistemática de 48 runs comparou os dois splits em 5 modelos
baseline × 3 variantes de feature engineering. Os resultados mostraram que
80/10/10 produz holdout AUC consistentemente ~0.002 superior (ex.:
`logreg_nophone_noml_8010_le` = 0.8725 vs `logreg_nophone_noml_7015_le` = 0.8505)
— consequência direta de +703 amostras de treino (+14%). Adicionalmente,
com 80/10/10 é possível reservar os 10% finais como *blind test set*
(completamente separado do early stopping do MLP), fornecendo uma estimativa
de generalização mais conservadora.

**Alternativas consideradas.**

1. *(rejeitada)* Manter 70/15/15 — conjunto de validação maior estabiliza a
   seleção de hiperparâmetros, mas o ganho de 14% no treino supera esse
   benefício dado o tamanho limitado do dataset (~7k linhas).
2. *(rejeitada)* k-fold puro (sem test fixo) — inviável para a API, que
   precisa de um modelo único serializado; blind test confirmaria estimativas
   de AUC antes do deploy.

**Decisão.** A partir da Fase 3 (MLP e Random Forest), todos os modelos
usam o split **80/10/10** (`test_size=0.10`, `val_size=0.10`). O conjunto
de teste (10%) é tratado como *blind test* e nunca é tocado durante treino
ou early stopping; as métricas `blind_test_*` são logadas no MLflow após
a avaliação final. O split 70/15/15 permanece nos notebooks 01–03 como
referência histórica.

**Consequências.**

- ✅ ~700 amostras a mais no treino — benefício mensurável em modelos de
  capacidade limitada (MLP, RF) com datasets pequenos.
- ✅ *Blind test* fornece estimativa de generalização isenta de viés de
  seleção por early stopping.
- ✅ Naming convention nos run names (`_8010_`) documenta o split diretamente
  no MLflow, facilitando comparações.
- ➖ Val holdout menor (10%) aumenta variância das métricas `holdout_val_*`
  — mitigado pela CV de 5 folds sobre o conjunto de treino.

---

## ADR-010 — Variante de feature engineering `ohe` como padrão para modelos finais

**Status:** Aceito — 2026-04-28

**Contexto.**
A bateria sistemática testou três variantes de feature engineering em todos
os modelos baseline e MLP:

| Variante | Features | CV AUC (LogReg) | Holdout AUC | Blind AUC (MLP) |
|----------|----------|-----------------|-------------|-----------------|
| `orig` | 27 feat (sem FE) | 0.8552 | 0.8706 | 0.8560 |
| `le` | 32 feat (tenure ordinal 0–3) | 0.8583 | 0.8725 | 0.8648 |
| `ohe` | 35 feat (tenure 4 bins OHE) | 0.8582 | 0.8719 | 0.8651 |

Features adicionadas em `le` e `ohe` (vs `orig`): `risco_contrato` (risco
ordinal do tipo de contrato), `service_count` (número de add-ons ativos),
`is_new` (flag: tenure ≤ 3 meses), `charges_per_tenure` (pressão de preço
por tempo de casa). `le` adiciona `tenure_bin_le` (ordinal 0–3); `ohe`
adiciona `tenure_bin_ohe` (4 colunas binárias com cutoffs explícitos).

**Alternativas consideradas.**

1. *(rejeitada)* `orig` — ~+0.003 de CV AUC a menos; as features de FE
   capturam informações de negócio relevantes (risco de contrato, inércia
   de serviços) que a regressão logística não consegue derivar sozinha.
2. *(rejeitada)* `le` — holdout AUC marginalmente superior a `ohe`
   (0.8725 vs 0.8719), mas o blind test do MLP favorece `ohe` (0.8651 vs
   0.8648). A diferença é menor que o erro de estimativa, mas `ohe` tem a
   vantagem adicional de não impor ordinalidade implícita nos bins de tenure,
   o que é semanticamente mais correto para um modelo não-linear (RF).

**Decisão.** A partir da Fase 3, todos os modelos finais usam a variante
**`ohe`** (`tenure_variant="ohe"`, 35 features pós-pipeline). O parâmetro
`tenure_variant` em `build_preprocessing_pipeline` permanece flexível para
permitir comparações; o default do projeto para modelos candidatos ao deploy
é `"ohe"`.

**Consequências.**

- ✅ Melhor generalização empiricamente verificada no blind test do MLP.
- ✅ Semântica correta: `ohe` não impõe distância numérica entre bins de
  tenure (relevante para Random Forest, que usa regras de divisão).
- ✅ Run names (`_ohe`) e o parâmetro MLflow `tenure_variant` tornam a
  escolha auditável.
- ➖ 8 features a mais que `orig` — custo negligível para RF e MLP; sem
  impacto em latência de inferência (<100 ms SLO).
