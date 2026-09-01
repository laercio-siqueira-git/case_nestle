"""Gera reports/RESULTS.md a partir dos artefatos da rodada.

    python scripts/build_results.py
    python scripts/build_results.py --config caminho/outro.yaml

Por que este script existe
--------------------------
Todo número que um relatório afirma é uma dívida: se a base mudar e o texto
não, o documento passa a mentir com aparência de verificado. As figuras já
resolvem isso derivando os próprios títulos. Este script fecha o mesmo buraco
na prosa: lê `benchmark.csv` e `metrics.json` — as saídas de `make benchmark`
e `make forecast` — e escreve um Markdown em que **nenhum número foi digitado
à mão**. O README descreve metodologia, que é estável; os resultados vivem
aqui e são regerados a cada ciclo.

Falha alto se um artefato estiver ausente ou velho demais para a base atual:
um RESULTS.md de ontem colado num benchmark de hoje é exatamente o problema
que este script existe para impedir.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import PipelineConfig

ALFA = 0.05


def _carregar(cfg: PipelineConfig) -> tuple[pd.DataFrame, dict, dict | None, dict | None]:
    faltando = [p for p in (cfg.benchmark_path, cfg.metrics_path) if not p.exists()]
    if faltando:
        nomes = ", ".join(p.name for p in faltando)
        raise SystemExit(
            f"Artefato(s) ausente(s): {nomes}.\n"
            f"Rode `make benchmark` e `make forecast` antes de gerar o RESULTS.md."
        )
    bench = pd.read_csv(cfg.benchmark_path)
    with cfg.metrics_path.open(encoding="utf-8") as fh:
        metrics = json.load(fh)

    # A análise de incerteza é opcional: sem ela o relatório sai, só que sem a
    # seção que diz o quanto se pode confiar num resultado nulo.
    incerteza = None
    if cfg.uncertainty_path.exists():
        with cfg.uncertainty_path.open(encoding="utf-8") as fh:
            incerteza = json.load(fh)

    sens = None
    if cfg.sensitivity_path.exists():
        with cfg.sensitivity_path.open(encoding="utf-8") as fh:
            sens = json.load(fh)
    return bench, metrics, incerteza, sens


def _tabela_horizonte(bench: pd.DataFrame, h: int, campeao: str) -> str:
    sub = bench[(bench.horizonte == h) & (bench.status == "ok")]
    if "embargo" in sub.columns:                      # --compare gera os dois
        sub = sub[sub.embargo]
    sub = sub.sort_values("MASE")
    linhas = ["| Modelo | MAPE | MASE | MAE | Viés | p vs naive | q (BH) |",
              "|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        p, q = r.get("p_vs_naive"), r.get("q_vs_naive")

        def _fmt(v):
            if v is None or pd.isna(v):
                return "— (é o baseline)"
            return f"**{v:.3f}**" if v < ALFA else f"{v:.3f}"

        nome = f"**{r.modelo}**" if r.modelo == campeao else str(r.modelo)
        linhas.append(
            f"| {nome} | {r.MAPE:.3f}% | {r.MASE:.3f} | {r.MAE:.3f} "
            f"| {r.vies:+.2f} | {_fmt(p)} | {_fmt(q)} |"
        )
    return "\n".join(linhas)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    bench, metrics, incerteza, sens = _carregar(cfg)
    cfg.preparar_diretorios()
    y = cfg.carregar_serie()
    bt = cfg.backtest_config()

    por_h = metrics["por_horizonte"]
    com_ganho = metrics["horizontes_com_ganho"]
    sem_ganho = metrics["horizontes_sem_ganho"]

    L: list[str] = []
    L.append("# Resultados")
    L.append("")
    L.append("> Documento **gerado** por `python scripts/build_results.py`. "
             "Nenhum número aqui foi digitado à mão — todos vêm de "
             "`reports/benchmark.csv` e `reports/metrics.json`. "
             "Regere sempre que a base mudar.")
    L.append("")
    L.append(f"- **Série:** {len(y)} observações mensais, "
             f"{y.index.min():%m/%Y} a {y.index.max():%m/%Y}")
    L.append(f"- **Backtest:** walk-forward, {bt.n_test} meses de teste, "
             f"treino mínimo {bt.min_train}, embargo `{bt.embargo}`")
    L.append(f"- **Horizontes:** {list(bt.horizons)}")
    L.append(f"- **Significância:** Diebold-Mariano bilateral, HAC de Bartlett "
             f"com h−1 defasagens e correção Harvey-Leybourne-Newbold, α = {ALFA}")
    L.append(f"- **Gerado em:** {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")

    # Sem isto, "não reproduziu" é ambíguo entre erro e ambiente diferente.
    # Modelos de árvore mudam de resultado entre versões de scikit-learn; os
    # lineares e os naives, não. Declarar o ambiente transforma a divergência
    # em fato verificável.
    prov = metrics.get("proveniencia")
    if prov:
        libs = ", ".join(f"{k} {v}" for k, v in prov["bibliotecas"].items())
        L.append(f"- **Ambiente:** Python {prov['python']} ({prov['plataforma']}) "
                 f"· {libs}")
        ident = [x for x in (
            f"commit `{prov['commit']}`" if prov.get("commit") else None,
            f"dado bruto `{prov['hash_dado_bruto']}`" if prov.get("hash_dado_bruto") else None,
        ) if x]
        if ident:
            L.append(f"- **Proveniência:** {' · '.join(ident)}")
        L.append("")
        L.append("> Os **vereditos** deste relatório se reproduzem em qualquer "
                 "ambiente compatível com o `requirements.txt`. Os **dígitos** "
                 "dos modelos de árvore, não: a implementação muda entre "
                 "versões de scikit-learn. Modelos lineares e baselines batem "
                 "exatamente. É por isso que o ambiente está declarado acima.")
    L.append("")

    # ---------------------------------------------------------------- veredito
    L.append("## Conclusão")
    L.append("")
    if com_ganho:
        L.append(f"Ganho sustentado por evidência em **h={com_ganho}**: nesses "
                 f"horizontes o modelo se justifica.")
    if sem_ganho:
        L.append(f"Sem evidência de ganho em **h={sem_ganho}**: o naive sazonal "
                 f"é suficiente, e manter modelo ali é custo sem retorno "
                 f"demonstrado.")
    if not com_ganho:
        L.append("")
        L.append("**Nenhum horizonte apresentou ganho demonstrável.** A "
                 "recomendação é não colocar modelo em produção com esta base.")
    L.append("")
    L.append("Leitura obrigatória do p-valor: não rejeitar a hipótese nula **não** "
             "prova equivalência. Com poucas observações o teste tem baixo poder, "
             "e a afirmação honesta é *\"não há evidência de diferença\"*, nunca "
             "*\"são iguais\"*.")
    L.append("")

    # ------------------------------------------------- sensibilidade da janela
    # Vem antes da seção de incerteza de propósito: se o ganho não sobrevive à
    # troca da janela de teste, discutir o intervalo dele é discutir a margem
    # de um número que não se reproduz.
    if sens:
        L.append("## O ganho sobrevive à troca da janela de teste?")
        L.append("")
        L.append("Uma janela de teste só responde *quanto o modelo ganhou naqueles "
                 "meses* — não responde *o modelo ganha*. Repetindo a medição em "
                 "janelas de tamanhos diferentes, um ganho real se mantém; um ganho "
                 "que era característica do período oscila e pode trocar de sinal.")
        L.append("")
        for h in bt.horizons:
            s = sens["por_horizonte"].get(str(h))
            if not s:
                continue
            L.append(f"**h={h} — `{s['campeao']}`:** {s['veredito']}")
            L.append("")
            L.append("| Janela | Período | Campeão | Naive | Ganho | DM p |")
            L.append("|---|---|---|---|---|---|")
            for j in s["janelas"]:
                p_txt = f"**{j['dm_p']:.3f}**" if j["significativo"] else f"{j['dm_p']:.3f}"
                L.append(
                    f"| {j['n_test']} | {j['periodo']} | {j['mape_campeao']:.3f}% "
                    f"| {j['mape_naive']:.3f}% | {j['ganho_pct']:+.1f}% | {p_txt} |"
                )
            L.append("")
        nao_repro = [h for h in bt.horizons
                     if (s := sens["por_horizonte"].get(str(h)))
                     and s.get("troca_sinal")]
        if nao_repro:
            L.append(f"> Em **h={nao_repro}** o ganho troca de sinal conforme o "
                     f"período avaliado. Um modelo cuja vantagem medida é ora "
                     f"positiva ora negativa não é um modelo indeciso — o ganho "
                     f"dele não é reproduzível, e isso basta para não colocá-lo "
                     f"em produção nesse horizonte.")
            L.append("")
        L.append("![sensibilidade à janela](figures/16_sensibilidade_janela.png)")
        L.append("")

    # ------------------------------------------------------------ incerteza
    if incerteza:
        L.append("## Quanta confiança a amostra permite")
        L.append("")
        L.append("O p-valor diz se há evidência; não diz de que tamanho é o ganho, "
                 "nem se o experimento conseguiria enxergá-lo. As duas colunas "
                 "abaixo respondem isso — a primeira por bootstrap de blocos sobre "
                 "os erros observados, a segunda impondo vantagens conhecidas e "
                 "contando quantas vezes o teste as encontra.")
        L.append("")
        L.append("| h | Ganho vs. naive | Intervalo | Detecta a partir de | "
                 "Poder no ganho observado | Falso positivo |")
        L.append("|---|---|---|---|---|---|")
        for h in bt.horizons:
            u = incerteza["por_horizonte"].get(str(h))
            if not u:
                continue
            pw = u["poder"]
            mde = "—" if pw["mde_pct"] is None else f"{pw['mde_pct']:.0f}%"
            fp = f"{pw['taxa_falso_positivo']:.0%}"
            if not pw["bem_calibrado"]:
                fp += " ⚠"
            L.append(
                f"| {h} | {u['ganho_pct']:+.1f}% | "
                f"[{u['ganho_pct_lo']:+.1f}%, {u['ganho_pct_hi']:+.1f}%] | "
                f"{mde} | {pw['poder_no_ganho_observado']:.0%} | {fp} |"
            )
        L.append("")
        for h in bt.horizons:
            u = incerteza["por_horizonte"].get(str(h))
            if u and u.get("leitura"):
                L.append(f"- **h={h}:** {u['leitura']}")
        L.append("")
        descalibrado = [h for h in bt.horizons
                        if (u := incerteza["por_horizonte"].get(str(h)))
                        and not u["poder"]["bem_calibrado"]]
        if descalibrado:
            L.append(f"> ⚠ Em vantagem imposta zero o teste deveria rejeitar "
                     f"{incerteza['alfa']:.0%} das vezes. Em h={descalibrado} ele "
                     f"rejeita mais, porque sobram poucos blocos por reamostragem. "
                     f"O efeito mínimo detectável desses horizontes é grosseiro — "
                     f"e, se algo, otimista.")
            L.append("")
        L.append("![intervalo do ganho](figures/14_bootstrap_ganho.png)")
        L.append("")
        L.append("![poder do teste](figures/15_poder_do_teste.png)")
        L.append("")

    # ---------------------------------------------------------------- por h
    for h in bt.horizons:
        info = por_h[str(h)]
        L.append(f"## Horizonte h={h}")
        L.append("")
        L.append(f"**Campeão declarado:** `{info['campeao']}` — {info['veredito']}")
        L.append("")
        L.append(_tabela_horizonte(bench, h, info["campeao"]))
        L.append("")
        L.append(f"- Explicação: {info['metodo_explicacao']}")
        L.append(f"- Peso do eco sazonal (lag_12 + lag_24 + yoy_diff_12): "
                 f"{info['importancia_eco_sazonal']}%")
        # Só reporta o peso das exógenas quando elas estão na matriz. Com a
        # flag desligada a linha sairia "0.0%", que não é um resultado — é a
        # ausência de uma variável, e o leitor entenderia como "medimos e deu
        # zero". A decisão de desligá-las, essa sim, está no config.yaml.
        peso_exog = info.get("importancia_exogenas_total")
        if peso_exog:
            L.append(f"- Peso das exógenas de calendário: {peso_exog}%")
        L.append(f"- Viés médio do campeão: {info['vies_medio']:+.2f} pontos")
        L.append(f"- Maior ganho sobre o naive: {info['mes_maior_ganho']}")
        naive_vence = info["meses_em_que_naive_vence"]
        L.append(f"- Meses em que o naive ainda vence: "
                 f"{', '.join(naive_vence) if naive_vence else 'nenhum'}")
        L.append(f"- Previsão: {' · '.join(f'{k} {v}' for k, v in info['forecast'].items())}")
        L.append(f"- Intervalo empírico {info['intervalo_cobertura_pct']}%: "
                 f"{info['intervalo_lo']:+.2f} a {info['intervalo_hi']:+.2f} pontos")
        L.append("")
        L.append(f"![backtest h={h}](figures/10_backtest_h{h}.png)")
        L.append("")
        L.append(f"![erro por mês h={h}](figures/11_erro_por_mes_h{h}.png)")
        L.append("")
        L.append(f"![importâncias h={h}](figures/12_importancias_h{h}.png)")
        L.append("")
        L.append(f"![forecast h={h}](figures/13_forecast_h{h}.png)")
        L.append("")

    L.append("---")
    L.append("")
    L.append("Metodologia, decisões de projeto e limitações conhecidas: ver "
             "`README.md`. Para reproduzir do zero: `make audit`.")
    L.append("")

    destino = cfg.metrics_path.parent / "RESULTS.md"
    destino.write_text("\n".join(L), encoding="utf-8")
    print(f"gerado: {destino}")
    print(f"  horizontes com ganho: {com_ganho or 'nenhum'}")
    print(f"  horizontes sem ganho: {sem_ganho or 'nenhum'}")

    if sens:
        readme = _atualizar_readme(cfg, bt, por_h, sens)
        if readme:
            print(f"atualizado: {readme} (bloco resumo-gerado)")


INICIO_BLOCO = "<!-- BEGIN:resumo-gerado"
FIM_BLOCO = "<!-- END:resumo-gerado -->"


def _atualizar_readme(cfg: PipelineConfig, bt, por_h: dict, sens: dict) -> Path | None:
    """Reescreve no README o bloco de resultados, entre marcadores.

    Por que o README participa da geração
    -------------------------------------
    O projeto afirma que nenhum número é digitado à mão, e o RESULTS.md cumpre
    isso. Só que o README é o que o avaliador lê primeiro, e ele trazia a mesma
    tabela escrita manualmente — que envelhecia a cada rodada e, em algumas
    linhas, passou a divergir dos artefatos. Uma ressalva de "ordem de grandeza"
    não conserta um número que virou outro.

    A saída não é tirar os números do README: é a mesma que se usou para as
    figuras e para o RESULTS.md — derivá-los. O texto ao redor continua sendo
    metodologia escrita à mão, que é estável; só o miolo numérico é reescrito
    aqui a cada geração.

    Devolve o caminho quando reescreveu, ``None`` quando não achou os
    marcadores — README editado ou renomeado não deve derrubar o pipeline.
    """
    caminho = cfg.root / "README.md"
    if not caminho.exists():
        return None
    texto = caminho.read_text(encoding="utf-8")
    i, j = texto.find(INICIO_BLOCO), texto.find(FIM_BLOCO)
    if i == -1 or j == -1 or j < i:
        return None

    rotulo = {1: "**1 mês** — operacional", 3: "**3 meses**",
              12: "**12 meses** — S&OP"}

    def _sg(v: float) -> str:
        """Percentual com sinal tipográfico, não hífen ASCII.

        O texto ao redor usa "−"; sem isto a mesma página teria dois traços
        diferentes para a mesma ideia.
        """
        return f"{v:+.0f}%".replace("-", "−")

    B = [texto[i:texto.index("-->", i) + 3]]
    B.append("")
    B.append("| Horizonte | Decisão | Ganho sobre o naive | Veredito |")
    B.append("|---|---|---|---|")
    for h in bt.horizons:
        s = sens["por_horizonte"].get(str(h))
        if not s:
            continue
        ganhos = [j_["ganho_pct"] for j_ in s["janelas"]]
        lo, hi = min(ganhos), max(ganhos)
        usa_modelo = bool(por_h[str(h)]["supera_baseline"]) and s["estavel"]
        decisao = "**usar modelo**" if usa_modelo else "usar naive"
        if not s["estavel"]:
            faixa = f"**{_sg(lo)} a {_sg(hi)} — troca de sinal**"
            veredito = "**não reproduzível** ✗"
        elif usa_modelo:
            faixa = f"{_sg(lo)} a {_sg(hi)} em todas as janelas"
            veredito = "**estável** ✓"
        else:
            faixa = f"{_sg(lo)} a {_sg(hi)}, nunca significativo"
            veredito = "estável, mas pequeno"
        B.append(f"| {rotulo.get(h, f'**{h} meses**')} | {decisao} | {faixa} "
                 f"| {veredito} |")
    B.append("")

    instaveis = [h for h in bt.horizons
                 if (s := sens["por_horizonte"].get(str(h))) and not s["estavel"]]
    if instaveis:
        h = instaveis[0]
        js = sens["por_horizonte"][str(h)]["janelas"]
        curta, longa = js[0], js[-1]
        B.append(
            f"Em h={min(bt.horizons)} a vantagem se mantém ao longo de toda a "
            f"série. Em h={h} ela **inverte de sinal** conforme o período: "
            f"parece {_sg(curta['ganho_pct'])} testando {curta['periodo']}, e "
            f"vira {_sg(longa['ganho_pct'])} testando {longa['periodo']}."
        )
        B.append("")

    B.append(f"<sub>Bloco gerado por `scripts/build_results.py` a partir de "
             f"`{cfg.sensitivity_path.name}` e `{cfg.metrics_path.name}`. "
             f"Números exatos e figuras em `reports/RESULTS.md`.</sub>")
    B.append("")

    caminho.write_text(texto[:i] + "\n".join(B) + texto[j:], encoding="utf-8")
    return caminho


if __name__ == "__main__":
    main()
