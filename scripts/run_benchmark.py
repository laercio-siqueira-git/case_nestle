"""
Benchmark completo: todos os modelos, todos os horizontes.

    python scripts/run_benchmark.py              # protocolo correto (embargo)
    python scripts/run_benchmark.py --compare    # também roda o protocolo antigo,
                                                 # para reproduzir o achado da auditoria

XGBoost e LightGBM entram automaticamente se estiverem instalados.
Saída: reports/benchmark.csv
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import pandas as pd

from src.config import PipelineConfig
from src.evaluation.backtest import run_full_benchmark
from src.evaluation.metrics import benjamini_hochberg, diebold_mariano
from src.models.registry import available_boosters, build_model_registry

#: Modelo contra o qual todos os outros são testados. É o piso do problema:
#: nenhum modelo justifica manutenção se não superar repetir o ano anterior.
BASELINE = "seasonal_naive"
ALFA = 0.05


def _p_contra_baseline(tab, resultados, baseline: str):
    """p-valor de Diebold-Mariano de cada linha da tabela contra o baseline.

    Comparação pareada: usa exatamente as mesmas datas previstas pelos dois
    modelos. O próprio baseline recebe NaN — não faz sentido testá-lo contra
    si mesmo.
    """
    ps = []
    for _, linha in tab.iterrows():
        nome, h = linha["modelo"], linha["horizonte"]
        base = resultados.get((baseline, h))
        alvo = resultados.get((nome, h))
        if nome == baseline or base is None or alvo is None:
            ps.append(float("nan"))
            continue
        pa, pb = alvo.predictions, base.predictions
        datas = pa.index.intersection(pb.index)
        _, p = diebold_mariano(
            pa.loc[datas, "y_true"], pa.loc[datas, "y_pred"],
            pb.loc[datas, "y_pred"], horizon=int(h),
        )
        ps.append(p)
    return ps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true",
                    help="roda também o protocolo defeituoso, para comparação")
    ap.add_argument("--config", default=None,
                    help="caminho alternativo para o YAML de configuração")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.config)
    print(f"configuração: {cfg.source}")
    print(f"boosters externos: {available_boosters()}")

    cfg.preparar_diretorios()
    y = cfg.carregar_serie()
    registry = build_model_registry(random_state=cfg.random_state)
    print(f"modelos: {list(registry)}\n")

    protocols = [True, False] if args.compare else [cfg.backtest_config().embargo]
    frames = []
    for embargo in protocols:
        bt = cfg.backtest_config(embargo=embargo)
        tab, res = run_full_benchmark(
            y, registry, bt,
            use_exogenous=cfg.use_exogenous,
            feature_params=cfg.feature_params,
        )
        tab["p_vs_naive"] = _p_contra_baseline(tab, res, BASELINE)
        frames.append(tab)

    out = pd.concat(frames, ignore_index=True)

    # Correção de múltiplos testes sobre TODAS as comparações da rodada. Sem
    # ela, o menor p de um conjunto grande é reportado como se fosse um teste
    # isolado — e a mesma janela de teste que escolheu o campeão também
    # produziu o p-valor dele.
    out["q_vs_naive"] = benjamini_hochberg(out["p_vs_naive"])
    out.to_csv(cfg.benchmark_path, index=False)

    n_testes = int(out["p_vs_naive"].notna().sum())
    sobreviventes = out[out["q_vs_naive"] < ALFA]

    pd.set_option("display.width", 220)
    for h in cfg.backtest_config().horizons:
        print(f"\n===== HORIZONTE h={h} =====")
        sub = out[(out.horizonte == h) & (out.status == "ok")]
        cols = ["modelo", "embargo", "MAPE", "MASE", "MAE", "RMSE", "vies",
                "p_vs_naive", "q_vs_naive"]
        print(sub[cols].sort_values(["embargo", "MASE"]).round(4).to_string(index=False))

    print(f"\n  p_vs_naive: Diebold-Mariano contra '{BASELINE}', bilateral, HAC(h-1)")
    print(f"  q_vs_naive: Benjamini-Hochberg sobre as {n_testes} comparações da rodada")
    print(f"  Use o q, não o p: {n_testes} testes a {ALFA:.0%} produziriam ~"
          f"{n_testes * ALFA:.0f} falso(s) positivo(s) por acaso.")
    if len(sobreviventes):
        resumo = ", ".join(f"{r.modelo} (h={r.horizonte})"
                           for r in sobreviventes.itertuples())
        print(f"  Sobrevivem a q < {ALFA}: {len(sobreviventes)} de {n_testes} — {resumo}")
    else:
        print(f"  Nenhum modelo sobrevive a q < {ALFA}.")
    print(f"\nsalvo em {cfg.benchmark_path}")


if __name__ == "__main__":
    main()
