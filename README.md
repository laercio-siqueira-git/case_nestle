# Previsão de produção de açúcar e confeitaria

Solução de ponta a ponta para previsão mensal de produção industrial, com
horizonte de 12 meses, voltada a planejamento de capacidade e estoque (S&OP).

**Base:** FRED `IPG3113N` — Industrial Production: Sugar and Confectionery
Product Manufacturing (NAICS 3113), EUA, jan/1972 a ago/2017, 548 observações
mensais.

---

## Resultado

O critério não é "quem tem o menor erro". É **o ganho se reproduz quando muda o
período avaliado?** Medido em quatro janelas de teste, de 3 a 23 anos:

<!-- BEGIN:resumo-gerado — bloco escrito por scripts/build_results.py. Não editar à mão. -->

| Horizonte | Decisão | Ganho sobre o naive | Veredito |
|---|---|---|---|
| **1 mês** — operacional | **usar modelo** | +32% a +42% em todas as janelas | **estável** ✓ |
| **3 meses** | usar naive | +4% a +13%, nunca significativo | estável, mas pequeno |
| **12 meses** — S&OP | usar naive | **−29% a +10% — troca de sinal** | **não reproduzível** ✗ |

Em h=1 a vantagem se mantém ao longo de toda a série. Em h=12 ela **inverte de sinal** conforme o período: parece +10% testando 2014-09 a 2017-08, e vira −18% testando 1994-12 a 2017-08.

<sub>Bloco gerado por `scripts/build_results.py` a partir de `sensitivity.json` e `metrics.json`. Números exatos e figuras em `reports/RESULTS.md`.</sub>
<!-- END:resumo-gerado -->

**Um modelo cuja vantagem medida muda de sinal conforme a janela não é um modelo
indeciso — o ganho dele não é reproduzível.** Isso basta para não colocá-lo em
produção nesse horizonte, e é conclusão mais firme que qualquer p-valor.

### O que a empresa faz com isso

**São dois produtos, não um.** O ajuste operacional de curto prazo roda com o
booster e economiza ~40% de erro, de forma reproduzível. O plano anual roda com
o naive sazonal: barato, explicável, sem manutenção — e o modelo que competiria
com ele não sustenta a própria vantagem.

**Avaliar em janela única esconde isso.** O ganho aparente em h=12 existia
numa janela de 36 meses e some ao alargá-la. A recomendação de engenharia é que múltiplas
janelas sejam padrão na esteira, não exceção — `make sensitivity` faz isso.

**No pico é onde o dinheiro está, e onde o modelo erra.** Set–dez concentram
38% da produção e 79% das anomalias, e todos os modelos subestimam. Qualquer
plano de capacidade para o quarto trimestre precisa de margem explícita.

<details>
<summary><b>Por que a vantagem colapsa, e por que ela é frágil em h=12</b></summary>

A previsão direta restringe as defasagens legais ao horizonte: em h=1 o modelo
conhece o mês recém-fechado (`lag_1`); em h=12 sobram apenas `lag_12`, `lag_13`
e `lag_24` — e as defasagens anuais correlacionam 0,93 entre si. É informação
redundante e cada vez mais velha. Medido: em h=12 as previsões do campeão e do
baseline correlacionam **0,98**.

E a fragilidade tem mecanismo. O Ridge extrapola tendência, o que ajuda em
período calmo — 2014–2017 foi calmo — e machuca ao atravessar quebra de regime.
A janela longa inclui a contração dos anos 2000 e a compressão sazonal. O naive,
que não extrapola nada, é indiferente a isso. Para planejamento de capacidade,
onde o risco que importa é justamente a quebra, isso é desqualificante.

</details>

<details>
<summary><b>E por que a janela de 36 meses não bastava</b></summary>

Com 36 pontos e previsões de 12 passos, a chance de detectar um ganho do tamanho
do observado ficava na casa de **10%** — o experimento não tinha resolução
para decidir.
A resposta não era buscar dados novos: a matriz já tinha 524 linhas e ~273
pontos de teste disponíveis. Estávamos usando 13% deles.

</details>

</details>

> **A tabela de resultado é gerada, não digitada.** Ela vive entre marcadores
> `<!-- BEGIN:resumo-gerado -->` e é reescrita por `scripts/build_results.py`
> a cada rodada, a partir de `sensitivity.json` e `metrics.json`. O restante
> deste README é metodologia, que é estável — e o texto foi escrito para não
> depender de valor específico, porque uma prosa com número cravado envelhece
> em silêncio. Os números completos, com figuras, estão em
> `reports/RESULTS.md`.

> **Nota metodológica.** Uma versão anterior deste repositório reportava
> Gradient Boosting com 3,34% em h=12. Esse número vinha de um backtest que
> incluía no treino alvos ainda não observados na origem da previsão. O bug
> foi encontrado por auditoria, corrigido, e está travado por testes. Rode
> `make compare` para ver os dois protocolos lado a lado, ou `make audit`
> para a demonstração completa com experimento de controle.

---

## Instalação e execução

Requer Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make test        # suíte de testes (inclui anti-vazamento e embargo)
make audit       # AUDITORIA: verifica as afirmações do relatório do zero
make eda         # análise exploratória -> reports/figures/
make benchmark   # todos os modelos x 3 horizontes
make compare     # benchmark nos dois protocolos, para reproduzir o achado
make forecast    # campeão de cada horizonte + previsão + explicação
make uncertainty # intervalo do ganho (bootstrap) + poder do teste
make sensitivity # o ganho sobrevive à troca da janela de teste?
make layers      # materializa refined/ e gold/ em Parquet
make results     # RESULTS.md gerado a partir dos artefatos da rodada
```

**`make` não vem instalado no Windows.** Para rodar tudo em qualquer sistema,
sem depender de ferramenta externa:

```bash
python scripts/run_all.py                      # pipeline completo (~10-15 min)
python scripts/run_all.py --skip audit         # sem a etapa mais lenta
python scripts/run_all.py --only eda benchmark
python scripts/run_all.py --config outro.yaml  # repassa a todas as etapas
```

Ele usa o mesmo interpretador que o invocou — rodando pelo Python do ambiente
virtual, todas as etapas herdam esse ambiente. Para no primeiro erro, porque
seguir depois de uma falha produz relatório parcial com cara de completo.

Ou os comandos avulsos, **nesta ordem** — `build_results.py` lê o que os três
anteriores produzem e falha alto se faltar algum:

```bash
ruff check src/ scripts/ tests/      # make lint
python -m pytest tests/ -v           # make test
python scripts/audit.py              # make audit
python scripts/run_eda.py            # make eda
python scripts/run_benchmark.py      # make benchmark   (ou: --compare)
python scripts/run_final.py          # make forecast
python scripts/run_uncertainty.py    # make uncertainty
python scripts/run_sensitivity.py    # make sensitivity
python scripts/build_layers.py       # make layers
python scripts/build_results.py      # make results
```

Todos aceitam `--config outro.yaml`, para rodar contra outra base ou outra
janela sem tocar no código.

**Comece por `make audit`.** Ele reproduz sozinho as quatro verificações que
sustentam o relatório e imprime PASSOU/FALHOU para cada uma, sem que você
precise confiar na documentação. Leva de 3 a 6 minutos.

### XGBoost e LightGBM

São opcionais. `src/models/registry.py` importa os dois de forma tolerante: se
não estiverem instalados, o pipeline roda normalmente e apenas não registra
esses modelos. Para incluí-los:

```bash
pip install xgboost lightgbm
python scripts/run_benchmark.py
```

Eles entram no benchmark automaticamente, configurados com a mesma
profundidade, taxa de aprendizado e semente dos demais boosters, para que a
comparação isole a implementação e não os hiperparâmetros.

Os dados brutos ficam em `data/raw/candy_production.csv`. Nenhuma etapa
requer acesso à rede: as variáveis de calendário são calculadas, não baixadas.

---

## Estrutura

```
.
├── config/config.yaml          # parâmetros centralizados (lido de fato)
├── data/
│   ├── raw/                    # imutável, como veio da fonte
│   ├── refined/                # validado e conformado (Parquet)
│   └── gold/                   # pronto para consumo (Parquet)
├── src/
│   ├── config.py               # lê o YAML e monta os objetos tipados
│   ├── data/
│   │   ├── loader.py           # ingestão + contrato de dados
│   │   └── calendar_features.py# Páscoa, feriados US, dias úteis (offline)
│   ├── features/
│   │   └── build_features.py   # matriz supervisionada por horizonte
│   ├── models/
│   │   └── registry.py         # baselines + modelos de ML
│   └── evaluation/
│       ├── metrics.py          # MAE, RMSE, MAPE, MASE, Diebold-Mariano
│       ├── uncertainty.py      # bootstrap de blocos + análise de poder
│       └── backtest.py         # walk-forward multi-horizonte
├── scripts/                    # orquestração executável
│   └── run_all.py              # pipeline completo, sem depender de `make`
├── tests/                      # garantias do pipeline
└── reports/                    # figuras, benchmark.csv, eda.json, metrics.json
```

Regra de dependência: `data` → `features` → `models` → `evaluation`. Nenhum
módulo importa de camada superior, e nenhum módulo de `src/` escreve arquivo
ou imprime na tela — isso fica nos `scripts/`. Assim cada peça é testável em
isolamento.

`src/config.py` é a única exceção deliberada, e fica *acima* de todas as
camadas: importa de `data` e de `evaluation` para montar `SeriesContract` e
`BacktestConfig`, e **nenhum módulo de `src/` o importa** — quem o usa são os
`scripts/`. Não há ciclo, e cada camada continua testável sem carregar
configuração de disco.

**Toda a parametrização vem de `config/config.yaml`.** Horizontes, defasagens,
janelas móveis, tamanho do backtest, semente, campeão de cada horizonte e
caminhos de saída são lidos de lá — mudar um número no YAML muda o pipeline
inteiro, sem editar código. Os caminhos são relativos à raiz do projeto e
resolvidos contra ela, então os scripts rodam de qualquer diretório e em
qualquer sistema operacional. Todos os scripts aceitam `--config outro.yaml`,
o que permite rodar o mesmo pipeline contra outra base ou outra janela.

**Nenhum texto de figura é fixo.** Períodos, meses destacados, vereditos de
teste e nome do campeão são calculados na própria rodada. Verificado rodando o
pipeline contra uma janela diferente da série: os títulos acompanham. Uma
figura que afirma o que a rodada não mostra é pior que uma sem texto, porque
parece verificada.

**Dados em camadas, materializadas.** `raw` é imutável; `refined` só recebe o
que passou pelo contrato de dados — é essa fronteira que dá sentido à camada,
não é "raw com outro nome"; `gold` é o que o negócio lê, sem precisar saber o
que é uma defasagem legal. Ambas em Parquet, que carrega o esquema dentro do
arquivo: medido nesta base, **3 de 16 colunas mudam de tipo** ao passar por
CSV — a data vira texto e os inteiros de 8 e 16 bits viram 64.

Em produção estas viram tabelas Delta, que **são** Parquet com um log de
transações por cima. Migrar não muda o formato dos dados, só o que existe em
volta deles — por isso o projeto para em Parquet e documenta Delta como próximo
passo, em vez de simular um Delta sem infraestrutura por baixo.

---

## Decisões técnicas que valem destaque

**Previsão direta, não recursiva.** Treina-se um modelo por horizonte, em vez
de realimentar a própria previsão 12 vezes. Custa mais modelos, mas o erro de
cada horizonte é medido diretamente e não se acumula.

**A restrição de vazamento está em código.** Para prever `t+h`, só defasagens
`>= h` são legais. `legal_lags_for_horizon()` filtra e `make_lag_features()`
levanta `ValueError` se receber uma defasagem ilegal. As janelas móveis
terminam em `t-h`, não em `t-1` — esse era um vazamento real na primeira
versão, hoje coberto por `test_no_leakage_at_horizon_12`.

**Baseline obrigatório, e testado.** Nenhum modelo é aceito sem superar o naive
sazonal — e "superar" significa passar no teste de Diebold-Mariano, não apenas
exibir MAPE menor. Com 36 pontos de teste e previsões de 12 passos, meio ponto
percentual cabe folgado dentro do ruído. `diebold_mariano()` usa variância HAC
com peso de Bartlett (o diferencial de perda é autocorrelacionado até `h-1`) e
a correção de amostra pequena de Harvey-Leybourne-Newbold. A coluna
`p_vs_naive` sai no `benchmark.csv` e o veredito vai no título da figura.

**Campeão por horizonte, não campeão único.** "Qual o melhor modelo" é pergunta
mal formulada sem dizer para qual decisão. `model.champion` no YAML é um mapa
`{horizonte: modelo}`: em h=1 vence um booster, em h=12 um linear — e em h=12
o honesto é o próprio baseline. A previsão direta já treina um modelo por
horizonte, então isso não custa arquitetura nenhuma.

**Explicação segue o modelo, não a moda.** Campeão de árvore recebe TreeSHAP
(atribuição local: "por que ESTE mês"). Campeão linear recebe o coeficiente
padronizado, que já é a explicação exata, global e aditiva — SHAP sobre um
modelo linear devolve exatamente `coef_j * (x_j - média_j)`, a mesma
informação, aproximada e mais cara. Ressalva registrada: `lag_12` e `lag_24`
correlacionam 0,93, e nenhum dos dois métodos separa contribuições de
preditores quase intercambiáveis.

**Variáveis de calendário calculadas, não baixadas.** Páscoa pelo algoritmo
gregoriano anônimo, feriados federais pela regra de 5 U.S.C. § 6103. Zero
dependência de API em produção e valores conhecidos com anos de antecedência
— condição necessária para uso como regressor futuro.

**Dois resultados negativos documentados.**

*Preço do açúcar não explica a estagnação.* A hipótese era que o programa
açucareiro americano — que mantém o açúcar doméstico caro — teria empurrado a
fabricação de doce para fora do país. Testada com as séries do FRED (No. 16
americano e No. 11 mundial), anualmente, em 25 anos: a correlação entre o
prêmio e a produção é **+0,59**, o oposto do previsto. Quem se correlaciona com
a produção é o **preço mundial (−0,68)** — choque de custo global, não política
doméstica. Os fatos históricos seguem de pé, a explicação causal não.

*As exógenas de calendário não agregam* — e por isso **estão desligadas**
(`features.use_exogenous: false`). Removê-las melhora os dois horizontes curtos
e piora só o longo, sempre por décimos de ponto percentual sobre o campeão:

| horizonte | campeão | efeito de remover as exógenas |
|---|---|---|
| h=1 | hist_gradient_boosting | −0,076 p.p. de MAPE (melhora) |
| h=3 | hist_gradient_boosting | −0,113 p.p. (melhora) |
| h=12 | ridge_fourier | +0,031 p.p. (piora) |

Melhoram os horizontes curtos, entre eles o único com ganho reproduzível sobre
o baseline; piora apenas h=12, que não vai a produção. A decisão não é de
acurácia — décimos de p.p. contra um ganho de 42% em h=1 não movem conclusão
nenhuma. É de manutenção: quatro colunas que não pagam o próprio custo saem da
matriz.

O achado é consistente com a EDA, que já não detectava efeito de Páscoa (Welch
p = 0,76 / 0,89 / 0,23, figura 05) nem correlação com o resíduo mensal (−0,064).

Reproduza a ablação completa com o config dedicado, que escreve num CSV próprio
para não sobrescrever o oficial:

```bash
python scripts/run_benchmark.py --config config/ablacao_com_exogenas.yaml
# -> reports/benchmark_com_exogenas.csv
```

---

## A exploração não está esgotada — e isso é declarado, não omitido

A EDA foi dimensionada para responder às perguntas de que a **decisão** precisava:
a série tem sinal previsível? de onde ele vem? o que o calendário explica? onde
o erro custa caro? Ela não tenta esgotar a série, e há um conjunto identificável
de análises que ficaram de fora — nenhuma delas por descuido:

- **Autocorrelação (ACF/PACF).** A ausência mais notável para quem vem de séries
  temporais. Não entrou porque a estratégia adotada é tabular e direta: a escolha
  das defasagens é governada pela regra `k ≥ h`, não pela leitura de um
  correlograma. Ela volta a ser necessária no momento em que SARIMA entrar como
  concorrente — e é por isso que os dois estão no mesmo item do roadmap.
- **Testes formais de estacionariedade** (ADF, KPSS). Mesma razão: modelos de
  árvore não exigem estacionariedade, e o Ridge recebe a tendência como
  regressor explícito. Vira pré-requisito com modelos paramétricos.
- **Teste formal de quebra estrutural** (Chow, Bai-Perron). A quebra do fim dos
  anos 90 foi identificada por comparação entre blocos e por janela móvel — o
  efeito é grande e visível, mas a **data** da quebra é estimada de olho, não
  por procedimento. Para datar com intervalo de confiança, faltou o teste.
- **Decomposição multiplicativa.** A adotada é aditiva. Como a amplitude sazonal
  mudou ao longo do histórico, é plausível que a sazonalidade seja proporcional
  ao nível — o que mudaria o perfil sazonal e a leitura da compressão dos anos
  90. É a análise que eu faria primeiro se retomasse a EDA.
- **As 24 anomalias, uma a uma.** Elas são contadas e localizadas no calendário,
  não investigadas individualmente. Duas das maiores negativas recentes são
  consecutivas (nov e dez de 2016), o que sugere evento único em vez de dois
  choques — e isso não foi apurado.

O critério para parar foi o mesmo do resto do projeto: cada análise adicional
precisa mudar uma **decisão**. As acima mudariam, se o escopo incluísse modelos
paramétricos ou datação de regime. Como não inclui, ficam registradas aqui em
vez de ausentes em silêncio.

## Limitações conhecidas mais sérias

**Em h=12 o experimento é cego, e nós medimos a cegueira.** O campeão empata com
o naive no Diebold-Mariano, e a tentação é concluir "os modelos são
equivalentes" — o que seria errado. A análise de poder (`make uncertainty`)
mostra por quê: com 36 pontos, só uma vantagem grande seria detectada 80% das
vezes, e a observada fica muito abaixo desse limiar. "Empate" descreve o
experimento, não os modelos. Os valores da rodada estão em
`reports/uncertainty.json` e na seção correspondente do `RESULTS.md`.

Uma ressalva sobre a própria análise, que o script reporta sozinho: em h=12
sobram 3 blocos por reamostragem, e nesse regime o teste rejeita 27% das vezes
quando a vantagem imposta é zero — deveria rejeitar 5%. O efeito mínimo
detectável ali é grosseiro e, se algo, otimista. Registrado em vez de escondido:
uma análise de poder mal calibrada que se apresenta como precisa seria pior que
não ter análise nenhuma.

**Nenhum hiperparâmetro foi ajustado, e a escolha do campeão usou a janela de
teste.** Os valores são os defaults (`alpha=1.0`, `n_estimators=400`,
`damping=0.5`). Pior: os pares modelo×horizonte foram comparados nas mesmas
36 observações que produzem o número reportado, então a métrica do vencedor é
otimista por seleção — é a única etapa do pipeline sem embargo. Medimos a
exposição: varrendo o `alpha` do Ridge de 0,01 a 1,0 o MAPE em h=12 varia
0,01 p.p. (platô), ou seja o viés vem da escolha de *família de modelo*, não do
ajuste fino.

O p-valor herda o mesmo problema — por isso o benchmark reporta também o
**q-valor de Benjamini-Hochberg** sobre todas as comparações da rodada. O
resultado é tranquilizador para a conclusão principal: sobrevivem a `q < 0,05`
exatamente os seis modelos de ML em h=1, e nenhum em h=3 ou h=12. A afirmação
"ML ganha em h=1" não depende de qual modelo venceu — todos vencem
individualmente, e o conjunto resiste à correção. A correção adequada é seleção aninhada — um walk-forward interno
por origem de previsão — e ela foi deliberadamente não implementada: com
n=36 e o DM já indicando ausência de diferença, o aninhamento tornaria a
estimativa não-enviesada sem mudar a conclusão. Fica registrado como o primeiro
item a construir quando o pipeline virar job de retreino recorrente, onde a
re-eleição periódica do campeão tem função real.

**Todos os modelos subestimam.** O viés é negativo em toda a tabela — do Ridge,
o menos enviesado, aos boosters, que erram mais para baixo. O próprio naive tem viés −2,38: a janela de
teste é um período de alta. Para S&OP, subestimar significa subdimensionar
turno e insumo, cujo custo é ruptura.

**Duas correções foram testadas e falharam.** Alvo em diferença sazonal
(`y[t] − y[t−12]`) e híbrido Ridge-tendência + árvores no resíduo pioraram o
resultado em h=12. A hipótese de que o gargalo era extrapolação de nível não
se sustentou. O próximo passo é regressão quantílica com perda assimétrica.
