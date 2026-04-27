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
