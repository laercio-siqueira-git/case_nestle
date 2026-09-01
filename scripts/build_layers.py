"""Materializa as camadas de dados: raw -> refined -> gold.

    python scripts/build_layers.py
    python scripts/build_layers.py --config caminho/outro.yaml

A arquitetura em camadas
------------------------
É a organização que o desenho de produção (seção de arquitetura) pressupõe,
materializada aqui em Parquet para que exista de fato e não só no slide.

    raw/       imutável, exatamente como veio da fonte. Nunca é reescrito.
    refined/   validado, tipado e conformado: uma tabela mensal única, com
               contrato de dados aplicado e as fontes já unidas.
    gold/      pronto para consumo: o que o S&OP e o painel leem, sem precisar
               saber o que é uma defasagem legal.

Por que Parquet, e não CSV
--------------------------
Um CSV perde os tipos: toda leitura precisa readivinhar o que é data, o que é
inteiro, o que é decimal — e erra em silêncio quando o dado muda. Parquet
carrega o esquema dentro do arquivo, é colunar (lê só as colunas pedidas) e
comprime bem.

E é o passo anterior a Delta: uma tabela Delta **é** Parquet com um log de
transações por cima, que acrescenta versionamento e leitura consistente. Migrar
daqui para lá não muda o formato dos dados, só o que existe em volta deles —
que é exatamente o motivo de o projeto parar em Parquet e documentar Delta como
próximo passo, em vez de simular um Delta que não tem infraestrutura por baixo.

Ordem de execução — e por que há dois estágios
----------------------------------------------
`refined` depende só de `raw`, e **tudo o mais depende de refined**: a EDA, o
benchmark e o forecast leem dela. Logo ela precisa existir antes de qualquer
análise.

`gold` é o oposto: depende dos artefatos que o modelo produz, então só pode ser
escrita no fim.

Um script que fizesse as duas coisas de uma vez não caberia nem no começo nem no
fim do pipeline. Daí o `--stage`:

    --stage refined   logo após os testes
    --stage gold      depois de forecast e sensitivity
    --stage all       as duas, para uso manual
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src.config import PipelineConfig
from src.data.calendar_features import build_calendar_frame
from src.data.loader import load_candy_production


def _serie_fred(caminho: Path) -> pd.Series:
    """Lê um CSV do FRED e devolve a série mensal, qualquer que seja o nome da coluna."""
    d = pd.read_csv(caminho)
    d["observation_date"] = pd.to_datetime(d["observation_date"])
    col = next(c for c in d.columns if c != "observation_date")
    return d.set_index("observation_date")[col].astype("float64")


def construir_refined(cfg: PipelineConfig) -> pd.DataFrame:
    """Fato mensal validado: produção + calendário + preços, numa tabela só.

    O contrato de dados de ``load_candy_production`` é a fronteira entre raw e
    refined: o que não passa por ele não vira refined. É isso que dá sentido à
    camada — não é "raw com outro nome", é "raw que provou estar íntegro".
    """
    y = load_candy_production(cfg.data_path, cfg.series_contract())

    tabela = pd.DataFrame({"producao": y})
    tabela.index.name = "data"
    tabela = tabela.join(build_calendar_frame(y.index))

    # Preços entram como colunas opcionais. Antes de 1992 ficam nulos — o que é
    # informação verdadeira, não falha: a fonte não cobre aquele período.
    if cfg.tem_precos_acucar:
        mundo = _serie_fred(cfg.sugar_world_path).rename("acucar_mundial")
        eua = _serie_fred(cfg.sugar_us_path).rename("acucar_eua")
        tabela = tabela.join(mundo).join(eua)
        tabela["acucar_premio"] = tabela["acucar_eua"] - tabela["acucar_mundial"]

    tabela["ano"] = tabela.index.year.astype("int16")
    tabela["mes"] = tabela.index.month.astype("int8")
    return tabela.reset_index()


def construir_gold(cfg: PipelineConfig) -> dict[str, pd.DataFrame]:
    """O que o negócio consome: previsões e desempenho, sem jargão de modelagem."""
    tabelas: dict[str, pd.DataFrame] = {}

    if cfg.metrics_path.exists():
        with cfg.metrics_path.open(encoding="utf-8") as fh:
            metrics = json.load(fh)

        # Duas linhas por mês previsto: a do modelo candidato e a do baseline.
        #
        # A coluna `recomendada` é o ponto da tabela. O relatório conclui que em
        # h=3 e h=12 o modelo não supera o naive; entregar só a previsão do
        # modelo faria a camada de consumo contradizer a própria recomendação —
        # e quem lê a tabela não tem como saber disso. Com as duas presentes,
        # `WHERE recomendada` devolve o número certo e a alternativa continua
        # visível para auditoria.
        linhas = []
        for h, info in metrics.get("por_horizonte", {}).items():
            recomendado = info.get(
                "modelo_recomendado",
                info["campeao"] if info["supera_baseline"] else "seasonal_naive",
            )
            variantes = [(info["campeao"], info.get("forecast", {}),
                          info["intervalo_lo"], info["intervalo_hi"])]
            if info.get("forecast_baseline"):
                variantes.append(("seasonal_naive", info["forecast_baseline"],
                                  info["intervalo_baseline_lo"],
                                  info["intervalo_baseline_hi"]))
            for modelo, previsoes, lo, hi in variantes:
                for data_txt, valor in previsoes.items():
                    linhas.append({
                        "horizonte_meses": int(h),
                        "mes_previsto": data_txt,
                        "previsao": float(valor),
                        "intervalo_inferior": float(valor) + float(lo),
                        "intervalo_superior": float(valor) + float(hi),
                        "cobertura_pct": int(info["intervalo_cobertura_pct"]),
                        "modelo": modelo,
                        "recomendada": modelo == recomendado,
                        "supera_baseline": bool(info["supera_baseline"]),
                    })
        if linhas:
            tabelas["previsoes"] = pd.DataFrame(linhas)

    if cfg.benchmark_path.exists():
        bench = pd.read_csv(cfg.benchmark_path)
        cols = [c for c in ("modelo", "horizonte", "MAPE", "MASE", "MAE", "RMSE",
                            "vies", "p_vs_naive", "q_vs_naive") if c in bench.columns]
        tabelas["desempenho_modelos"] = bench[bench.status == "ok"][cols].copy()

    if cfg.sensitivity_path.exists():
        with cfg.sensitivity_path.open(encoding="utf-8") as fh:
            sens = json.load(fh)
        linhas = []
        for h, info in sens.get("por_horizonte", {}).items():
            for j in info["janelas"]:
                linhas.append({
                    "horizonte_meses": int(h),
                    "modelo": info["campeao"],
                    "janela_meses": int(j["n_test"]),
                    "periodo": j["periodo"],
                    "ganho_pct": float(j["ganho_pct"]),
                    "dm_p": float(j["dm_p"]),
                    "ganho_reproduzivel": bool(info["estavel"]),
                })
        if linhas:
            tabelas["estabilidade_por_janela"] = pd.DataFrame(linhas)

    return tabelas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
    ap.add_argument("--stage", choices=("refined", "gold", "all"), default="all",
                    help="qual camada materializar (padrão: as duas)")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    cfg.preparar_diretorios()
    print(f"configuração: {cfg.source}")

    if args.stage in ("refined", "all"):
        print(f"\n{'=' * 66}\nREFINED — validado, tipado, conformado\n{'=' * 66}")
        refined = construir_refined(cfg)
        destino = cfg.refined_path
        refined.to_parquet(destino, index=False, engine="pyarrow", compression="snappy")
        tam = destino.stat().st_size / 1024
        print(f"   {destino.name:<32} {len(refined):>5} linhas · "
              f"{len(refined.columns):>2} colunas · {tam:6.1f} KB")
        print(f"   período: {refined['data'].min():%Y-%m} a {refined['data'].max():%Y-%m}")
        nulos = refined.columns[refined.isna().any()].tolist()
        if nulos:
            print(f"   colunas com nulos (cobertura parcial da fonte): {nulos}")

    if args.stage in ("gold", "all"):
        print(f"\n{'=' * 66}\nGOLD — pronto para consumo\n{'=' * 66}")
        gold = construir_gold(cfg)
        if not gold:
            print("   nenhum artefato de modelo encontrado — rode forecast antes.")
        for nome, tabela in gold.items():
            destino = cfg.gold_dir / f"{nome}.parquet"
            tabela.to_parquet(destino, index=False, engine="pyarrow",
                              compression="snappy")
            tam = destino.stat().st_size / 1024
            print(f"   {destino.name:<32} {len(tabela):>5} linhas · "
                  f"{len(tabela.columns):>2} colunas · {tam:6.1f} KB")

    print(f"\nraw     -> {cfg.data_path.parent}")
    print(f"refined -> {cfg.refined_dir}")
    print(f"gold    -> {cfg.gold_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
