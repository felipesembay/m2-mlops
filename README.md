# M2 Brasil — MLOps para previsão do crescimento mensal

Coloca em produção os modelos do artigo *Previsão do crescimento mensal do M2 brasileiro*
(o código original da pesquisa está no notebook do artigo; aqui ele foi refatorado em pacote,
com testes de paridade contra os resultados publicados).

| Horizonte | Campeão (validação 2021–2025) | RMSE validação | Registro MLflow |
|---|---|---|---|
| 1 mês | STL + ETS | 0,610 | `m2_h1@champion` |
| 3 meses | Random Forest (otimização bayesiana) | 0,588 | `m2_h3@champion` |
| 6 meses | SARIMA(3,0,0)(0,1,1)12 | 0,543 | `m2_h6@champion` |
| 12 meses | Random Forest (otimização bayesiana) | 0,495 | `m2_h12@champion` |

O benchmark (Média Sazonal) também é registrado em `m2_benchmark_h{1,3,6,12}`.

## Produção

| Peça | Onde | Papel |
|---|---|---|
| App web + API (só leitura) | **Vercel** (`app.py` → `api/main.py`) | serve front e endpoints lendo o Postgres |
| Banco | **Neon Postgres** (integração do Vercel) | série, previsões, monitoramento, projeções pré-calculadas |
| Orquestração | **GitHub Actions** (`.github/workflows/pipeline.yml`, diário 06:00 BRT) | mesmo encadeamento das DAGs do Airflow |
| Registro de modelos | **MLflow no DagsHub** | versões e aliases `champion`/`challenger` por horizonte |

```
GitHub Actions (cron) ── scripts/rodar_pipeline.py
   ingerir → treinar_e_prever → monitorar → reotimizar (se gatilho) → publicar
        │ grava                                  │ registra
        ▼                                        ▼
   Neon Postgres  ◄── lê ── Vercel (FastAPI + front)     MLflow (DagsHub)
```
`publicar()` pré-calcula as projeções de 24 meses de todos os modos e um retrato do registro. Assim a função
do Vercel instala só o núcleo leve (pandas, SQLAlchemy, FastAPI) e responde em milissegundos, sem reajustar modelos.
O Airflow + Docker Compose abaixo continua sendo a orquestração completa para rodar localmente.

### Replicar
1. **Vercel:** importe o repositório (framework detectado: FastAPI) e adicione a integração **Neon**
   (Storage → Neon, região São Paulo). Ela injeta `DATABASE_URL` no projeto.
2. **MLflow:** crie um repositório no [DagsHub](https://dagshub.com) e pegue a URI
   `https://dagshub.com/<usuario>/<repo>.mlflow` e um token.
3. **Secrets do GitHub** (Settings → Secrets → Actions): `DATABASE_URL` (a mesma do Neon),
   `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, `MLFLOW_TRACKING_PASSWORD` (token).
4. **Actions → pipeline → Run workflow → `bootstrap`**: carga inicial. Depois disso o cron diário mantém tudo atualizado.

## Arquitetura (stack local completa)

```
BCB/SGS 27810 ──► Airflow: m2_ingestao (diário 06:00)
                     │ só segue se houver mês novo ou revisão
                     ▼
                  m2_treino_previsao ── reajusta campeão + benchmark (janela 120m)
                     │                   ├─► MLflow: nova versão, alias "champion"
                     │                   └─► Postgres: tabela previsoes
                     ▼
                  m2_monitoramento ──── RMSE móvel 12m vs benchmark → regra de retreino
                     │ (se gatilho A ou B)
                     ▼
                  m2_reotimizacao ───── BO do RF + walk-forward dos candidatos
                                         → "challenger" ou promoção a "champion"

FastAPI (:8000) ── lê Postgres + MLflow ── front HTML/Chart.js
```

| Serviço | URL |
|---|---|
| App web / API | http://localhost:8000 (docs em `/docs`) |
| MLflow | http://localhost:5000 |
| Airflow | http://localhost:8080 |

## Como rodar

### Docker (stack completa)
```bash
cp .env.example .env            # troque as senhas
docker-compose up -d --build
docker-compose run --rm api python scripts/bootstrap.py      # carga inicial (baixa do BCB)
```
O bootstrap faz o backtest walk-forward desde a origem 2021-02 (mesmo protocolo do artigo), para que o
painel de monitoramento já tenha histórico, e registra a v1 de cada modelo no MLflow.
Depois disso o Airflow mantém tudo atualizado sozinho.

### Local (sem Docker: SQLite + MLflow em arquivo)
```bash
uv venv -p 3.11 .venv && uv pip install -p .venv/bin/python -e ".[ml,api,dev]"
.venv/bin/python scripts/bootstrap.py --snapshot tests/fixtures/bcb_27810_snapshot.json
.venv/bin/uvicorn api.main:app --reload       # http://localhost:8000
.venv/bin/mlflow ui                            # opcional (precisa do pacote mlflow completo)
```

### Testes
```bash
uv pip install -p .venv/bin/python -e ".[ml,api,dev]"
.venv/bin/python -m pytest -q
```
`tests/test_paridade_artigo.py` reproduz as Tabelas 6–9 (validação) e 11 (teste 2026) do artigo
a partir da série congelada, garantindo que o código do pacote é o mesmo do notebook.

## Aplicação web

- **Projeção:** escolha o modelo e de 1 a 24 meses à frente. Saem a trajetória mês a mês, uma faixa indicativa
  (± 1,96 × RMSE de validação), o crescimento acumulado e o M2 projetado em R$.
  - *Melhor modelo por horizonte* (padrão): mês 1 → STL+ETS; meses 2–3 → RF (h=3); 4–6 → SARIMA; 7+ → RF (h=12).
  - *Campeão de um h*: usa só aquele modelo. SARIMA, ETS, STL+ETS e Média Sazonal projetam a trajetória
    com um único ajuste; o Random Forest (previsão direta) é reajustado para cada distância.
  - Além de 12 meses a projeção fica fora do que foi validado no artigo (a tabela marca isso).
- **Série histórica completa** (jan/2002 em diante), com seletor de 5 anos / 10 anos / tudo.
- **Previsto × realizado** por horizonte, com o monitoramento explicado (ver abaixo).
- Endpoints: `/api/projecao?modelo=melhor|1|3|6|12|benchmark&meses=1..24`, `/api/serie`, `/api/previsoes`,
  `/api/monitoramento`, `/api/modelos`, `/api/prever/{h}`. No Docker (`M2_SERVIR_MLFLOW=1`), `/api/prever/{h}` carrega
  o modelo do MLflow; em produção devolve a última previsão gravada pelo pipeline.

## Monitoramento e regra de retreino

Para cada horizonte *h*, sobre os erros fora da amostra dos últimos 12 meses-alvo (mínimo 6):

**R<sub>h</sub> = RMSE<sub>12m</sub>(campeão) / RMSE<sub>12m</sub>(Média Sazonal)**

Como ler: **R = 0,90** quer dizer que, nos últimos 12 meses, o modelo errou 10% menos que a média sazonal;
**R = 1,20**, que errou 20% mais. R abaixo de 1 = o modelo agrega valor; acima de 1 = perde para o benchmark.

- **Gatilho A:** perda persistente para o benchmark: R<sub>h</sub> > 1,0 por 3 meses seguidos.
- **Gatilho B:** piora absoluta: RMSE<sub>12m</sub> > 1,25 × RMSE de validação do campeão.
- A **ou** B dispara `m2_reotimizacao` para aquele horizonte.
- **Carência:** depois de uma re-otimização, o mesmo horizonte não dispara de novo por 6 meses.
- **Promoção:** o novo modelo só vira `champion` se reduzir o RMSE em ≥ 2% na mesma janela
  (58 meses); caso contrário fica como `challenger`, para auditoria.
- **Alertas** (não disparam retreino): |viés<sub>12m</sub>| > 0,30 p.p.; nova observação com |z| > 3
  em relação aos 120 meses anteriores (sinal de quebra estrutural).

Por que razão de RMSE e não Diebold-Mariano? O artigo mostra que, com menos de 60 previsões, o teste não
distingue diferenças de até 15% no RMSE. A razão com persistência temporal é estável e fácil de
interpretar, e o benchmark é o mesmo da Tabela 10. Todos os limiares ficam em `configs/modelos.yaml`.

Reajuste ≠ retreino: assim como no artigo, todo modelo é **reestimado todo mês** na nova janela
de 120 meses (DAG 2). O gatilho decide quando **re-otimizar hiperparâmetros e re-selecionar a
família do modelo** (DAG 4).

## Estrutura
```
configs/            modelos.yaml (campeões, limiares), rf_h3.json / rf_h12.json (hiperparâmetros da OB)
src/m2forecast/     data, features, models/, backtest, tracking (MLflow), monitoring, reoptimize, pipeline, db
dags/               m2_ingestao, m2_treino_previsao, m2_monitoramento, m2_reotimizacao
api/                FastAPI + static/ (front-end)
scripts/            bootstrap.py, reotimizar.py, rodar_pipeline.py (pipeline completo p/ GitHub Actions)
app.py, vercel.json entrypoint e configuração do Vercel
.github/workflows/  pipeline.yml (cron de produção), testes.yml
tests/              paridade com o artigo, monitoramento, dados, pipeline ponta a ponta
```

## Observação sobre o SARIMA
Com statsmodels/scipy locais, as previsões do SARIMA no teste de 2026 diferem do notebook (Colab) em até
0,07 p.p. com o mesmo código (otimização numérica da máxima verossimilhança). O RMSE de 2026 fica em 0,369
contra 0,389 no artigo, e a validação (0,543) bate exatamente.
