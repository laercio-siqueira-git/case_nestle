"""O ganho medido sobrevive à troca da janela de teste?

    python scripts/run_sensitivity.py
    python scripts/run_sensitivity.py --config caminho/outro.yaml

Por que este script existe
--------------------------
Todo o resto do pipeline mede o ganho numa janela de teste só. Isso responde
"quanto o modelo ganhou naqueles meses" — não responde "o modelo ganha".

A diferença aparece quando se repete a medição em janelas de tamanhos
diferentes. Um ganho real se mantém. Um ganho que era característica daquele
período específico oscila e, no limite, **troca de sinal**.

Trocar de sinal encerra a discussão: um modelo cuja vantagem medida é ora
positiva ora negativa conforme o período avaliado não é indeciso — o ganho dele
não é reproduzível, e isso basta para não colocá-lo em produção.

A escolha de `n_test` deixa de ser um parâmetro invisível de configuração e
passa a ser uma dimensão auditada do resultado.

Saída: reports/sensitivity.json e a figura 16.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import salvar_json
from src.config import PipelineConfig
from src.evaluation.backtest import BacktestConfig, walk_forward_backtest
from src.evaluation.metrics import diebold_mariano
from src.evaluation.uncertainty import veredito_estabilidade
from src.features.build_features import make_supervised_frame
from src.models.registry import build_model_registry

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
NAVY, ORANGE, GREY, GREEN, RED = "#1f3864", "#e07b39", "#8c8c8c", "#2e7d5b", "#b5342a"
BASELINE = "seasonal_naive"
ALFA = 0.05

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
args = ap.parse_args()

CFG = PipelineConfig.load(args.config)
FIG = CFG.figures_dir
bt_base = CFG.backtest_config()
CFG.preparar_diretorios()
y = CFG.carregar_serie()

print(f"configuração: {CFG.source}")
print(f"janelas declaradas: {list(CFG.janelas_sensibilidade)} (+ máxima disponível)")

out: dict = {"alfa": ALFA, "baseline": BASELINE, "por_horizonte": {}}
dados_fig: dict = {}

for h in bt_base.horizons:
    campeao = CFG.champion_for(h)
    X, yh = make_supervised_frame(y, horizon=h, use_exogenous=CFG.use_exogenous,
                                  **CFG.feature_params)

    # A janela máxima é quantos alvos ainda deixam min_train linhas legais no
    # treino. Calculada, não chutada: depende do horizonte e do aquecimento.
    maxima = 0
    for cutoff in reversed(yh.index):
        origem = cutoff - pd.DateOffset(months=h)
        if int((yh.index <= origem).sum()) < bt_base.min_train:
            break
        maxima += 1

    janelas = sorted({*CFG.janelas_sensibilidade, maxima})
    janelas = [j for j in janelas if 0 < j <= maxima]

    print(f"\n{'=' * 74}\nHORIZONTE h={h} — {campeao} vs {BASELINE}\n{'=' * 74}")
    print(f"   janela máxima disponível: {maxima} pontos")
    print()
    print("   n_test   período              campeão   naive     ganho     DM p")

    linhas, ganhos = [], {}
    for n_test in janelas:
        bt = BacktestConfig(n_test=n_test, horizons=(h,),
                            min_train=bt_base.min_train, season=bt_base.season,
                            embargo=True)
        reg = build_model_registry(random_state=CFG.random_state)
        rc = walk_forward_backtest(reg[campeao], X, yh, bt, h, campeao)
        rb = walk_forward_backtest(reg[BASELINE], X, yh, bt, h, BASELINE)
        pc, pb = rc.predictions, rb.predictions
        datas = pc.index.intersection(pb.index)
        yt = pc.loc[datas, "y_true"].to_numpy()
        pm = pc.loc[datas, "y_pred"].to_numpy()
        pn = pb.loc[datas, "y_pred"].to_numpy()
        _, p = diebold_mariano(yt, pm, pn, horizon=h)
        mae_m, mae_n = np.abs(yt - pm).mean(), np.abs(yt - pn).mean()
        ganho = float((mae_n - mae_m) / mae_n * 100)
        ganhos[len(datas)] = ganho

        periodo = f"{datas.min():%Y-%m} a {datas.max():%Y-%m}"
        linhas.append({
            "n_test": len(datas), "periodo": periodo,
            "mape_campeao": round(float(rc.metrics["MAPE"]), 3),
            "mape_naive": round(float(rb.metrics["MAPE"]), 3),
            "ganho_pct": round(ganho, 2),
            "dm_p": round(float(p), 4),
            "significativo": bool(p < ALFA),
        })
        marca = "  <-- SIGNIF" if p < ALFA else ""
        print(f"   {len(datas):>5}    {periodo:<20} {rc.metrics['MAPE']:6.3f}%  "
              f"{rb.metrics['MAPE']:6.3f}%  {ganho:+6.1f}%  {p:7.4f}{marca}")

    vd = veredito_estabilidade(ganhos)
    print(f"\n   -> {vd['veredito']}")

    out["por_horizonte"][str(h)] = {
        "campeao": campeao, "janela_maxima": maxima,
        "janelas": linhas, **vd,
    }
    dados_fig[h] = {"ganhos": ganhos, "campeao": campeao, "vd": vd}

# ---------------------------------------------------------------- Fig 16
# O eixo x é a janela de teste — o parâmetro que normalmente fica escondido no
# arquivo de configuração. Ver o ganho cruzar o zero conforme ele muda é o
# argumento inteiro numa imagem.
fig, ax = plt.subplots(figsize=(9.0, 3.6))
for (h, cor) in zip(bt_base.horizons, [NAVY, ORANGE, GREEN, RED]):
    d = dados_fig[h]
    xs = sorted(d["ganhos"])
    ys = [d["ganhos"][x] for x in xs]
    estavel = d["vd"]["estavel"]
    ax.plot(xs, ys, marker="o", ms=5, lw=2.0 if estavel else 1.6,
            ls="-" if estavel else "--", color=cor,
            label=f"h={h} · {d['campeao']} — {'estável' if estavel else 'não reproduzível'}")
ax.axhline(0, color=RED, lw=1.4)
ax.text(0.01, 0.02, "abaixo desta linha o modelo é PIOR que o baseline",
        transform=ax.transAxes, fontsize=8, color=RED)
ax.set_xlabel("Tamanho da janela de teste (meses avaliados)")
ax.set_ylabel("Ganho sobre o naive (%)")
ax.legend(frameon=False, fontsize=8.5, loc="upper right")

instaveis = [h for h in bt_base.horizons if not dados_fig[h]["vd"]["estavel"]]
if instaveis:
    sub = (f"Em h={instaveis} o ganho não sobrevive à troca da janela — "
           f"o resultado era característica do período, não do modelo")
else:
    sub = "O ganho se mantém em todas as janelas testadas"
ax.set_title(f"O ganho medido depende de quais meses você testa?\n{sub}",
             loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "16_sensibilidade_janela.png"); plt.close(fig)

salvar_json(out, CFG.sensitivity_path)

print(f"\n{'=' * 74}\nLEITURA\n{'=' * 74}")
for h in bt_base.horizons:
    print(f"  h={h}: {dados_fig[h]['vd']['veredito']}")
print(f"\nfigura em {FIG}\nnúmeros em {CFG.sensitivity_path}")
