# FIAP - Tech Challenge - Fase 01

**Tema:** Rede Neural para Previsão de Churn com Pipeline Profissional End-to-End

**Contexto:** 
Uma operadora de telecomunicações está perdendo clientes em ritmo acelerado. A diretoria precisa de um modelo preditivo de churn que classifique clientes com risco de cancelamento. Assim, o grupo deve construir o projeto do zero ao modelo servido via API, aplicando todas as boas práticas de engenharia de ML aprendidas na Fase 1.

O modelo central de entrega é uma rede neural (MLP), treinada com PyTorch, comparada com baselines (Scikit-Learn) e rastreada com MLflow.

O modelo deve usar FastAPI para a API de inferência do modelo

## **Entrega obrigatória:** 

**Repositório GitHub**

- Estrutura organizada: src/, data/, models/, tests/, notebooks/, docs/.
- README.md completo com instruções de setup, execução e descrição do projeto.
- pyproject.toml como single source of truth (dependências, linting, pytest).
- Histórico de commits limpo e significativo (não 1 commit gigante).
- .gitignore adequado para projetos de ML.


**Vídeo de 5 minutos utilizando a metodologia S.T.A.R.**

- Situation: Qual o problema de negócio e o contexto do dataset?
- Task: Qual a tarefa do grupo e os objetivos técnicos?
- Action: Quais decisões técnicas foram tomadas (arquitetura, features, modelo, métricas)?
- Result: Quais os resultados obtidos e as lições aprendidas?

## **Entrega opcional:** 

- Deploy em ambiente de produção em nuvem (AWS, Azure ou GCP).

### Bibliotecas Requeridas

- PyTorch — construção e treinamento da rede neural (MLP).
- Scikit-Learn — pipelines de pré-processamento e modelos baseline.
- MLflow — tracking de experimentos (parâmetros, métricas, artefatos).
- FastAPI — API de inferência do modelo.

### Boas Práticas Obrigatórias
- Seeds fixados para reprodutibilidade.
- Validação cruzada estratificada.
- Model Card documentando limitações e vieses.
- Testes automatizados (≥ 3: smoke test, schema, API).
- Logging estruturado (sem print()).
- Linting com ruff sem erros.

### Dataset Sugerido
Dataset público de telecomunicações com variáveis tabulares (ex.: Telco Customer Churn — IBM). Alternativas: qualquer dataset de classificação binária com ≥ 5.000 registros e ≥ 10 features.

### Passo a Passo Resumido
1. EDA + ML Canvas + Baselines → MLflow tracking.
2. MLP PyTorch + comparação de modelos + análise de custo.
3. Refatoração + API FastAPI + testes + Makefile.
4. Model Card + README + vídeo STAR + (opcional) deploy em nuvem.

### Etapas de Desenvolvimento
**1. Entendimento e Preparação**

1. Preencher ML Canvas (stakeholders, métricas de negócio, SLOs).
2. EDA completa: volume, qualidade, distribuição, data readiness.
3. Definir métrica técnica (AUC-ROC, PR-AUC, F1) e métrica de negócio (custo de churn evitado).
4. Treinar baseline com DummyClassifier e Regressão Logística (Scikit-Learn).

**2. Modelagem com Redes Neurais**

1. Construir MLP em PyTorch: definir arquitetura, função de ativação, loss function.
2. Implementar loop de treinamento com early stopping e batching.
3. Comparar MLP vs. baselines (lineares + árvores) usando ≥ 4 métricas.
4. Analisar trade-off de custo (falso positivo vs. negativo).
5. Registrar todos os experimentos (MLP e ensembles) no MLflow.

**3. Engenharia e API**

1. Refatorar código em módulos (src/) com estrutura limpa.
2. Criar pipeline reprodutível (sklearn + transformadores custom).
3. Escrever testes (pytest): unitários, schema (pandera), smoke test.
4. Construir API FastAPI: /predict, /health, validação Pydantic.
5. Adicionar logging estruturado e middleware de latência.
6. Configurar pyproject.toml, ruff, Makefile (lint, test, run).

**4. Documentação e Entrega Final**

1. Gerar Model Card completo (performance, limitações, vieses, cenários de falha).
2. Documentar arquitetura de deploy escolhida (batch vs. real-time) + justificativa.
3. Criar plano de monitoramento (métricas, alertas, playbook de resposta).
4. Finalizar README com instruções de setup + execução + arquitetura.
5. Gravar vídeo de 5 min (método STAR) demonstrando o projeto.
6. (Opcional) Deploy da API em nuvem (AWS/Azure/GCP) com endpoint público.

#### Critérios de Avaliação

| Critério | Peso | Descrição |
|----------|------|------------|
| Qualidade do código e estrutura. | 20% | Organização, modularidade, SOLID, linting sem erros. |
| Rede neural (PyTorch). | 25% | MLP funcional, treinamento com early stopping, comparação com baselines. |
| Pipeline e reprodutibilidade. | 15% | Pipeline sklearn, seeds, pyproject.toml, instala do zero. |
| API de inferência. | 15% | FastAPI funcional, Pydantic, logging, testes passando. |
| Documentação e Model Card. | 10% | Model Card completa, README claro, plano de monitoramento. |
| Vídeo STAR. | 10% | Clareza, cobertura dos quatro elementos STAR, dentro de cinco minutos. |
| Bônus: deploy em nuvem. | 5% | API acessível via URL pública. |