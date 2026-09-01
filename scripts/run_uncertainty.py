"""Quanta confiança a amostra permite: intervalo do ganho e cegueira do teste.

    python scripts/run_uncertainty.py
    python scripts/run_uncertainty.py --config caminho/outro.yaml

Duas perguntas que o p-valor sozinho não responde:

1. **De que tamanho é o ganho, com que margem?** Bootstrap de blocos sobre os
   erros observados. Devolve um intervalo na unidade do problema, onde dá para
   *ver* se o zero está dentro.

2. **O teste enxergaria o ganho, se ele existisse?** Impõe vantagens conhecidas
   sobre os erros reais e conta quantas vezes o teste as encontra. Sai daí o
   efeito mínimo detectável — e, com ele, a diferença entre "os modelos são
   iguais" e "esta base não tem tamanho para decidir".

Saída: reports/uncertainty.json e as figuras 14 e 15.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import salvar_json
from src.config import PipelineConfig
from src.evaluation.backtest import walk_forward_backtest
from src.evaluation.metrics import diebold_mariano
from src.evaluation.uncertainty import efeito_minimo_detectavel, intervalo_bootstrap
from src.features.build_features import make_supervised_frame
from src.models.registry import build_model_registry

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
NAVY, ORANGE, GREY, GREEN, RED = "#1f3864", "#e07b39", "#8c8c8c", "#2e7d5b", "#b5342a"
BASELINE = "seasonal_naive"
ALFA = 0.05
PODER_ALVO = 0.80

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
ap.add_argument("--n-bootstrap", type=int, default=5000)
ap.add_argument("--n-simulacoes", type=int, default=600)
args = ap.parse_args()

CFG = PipelineConfig.load(args.config)
FIG = CFG.figures_dir
bt = CFG.backtest_config()
CFG.preparar_diretorios()
y = CFG.carregar_serie()
reg = build_model_registry(random_state=CFG.random_state)

print(f"configuração: {CFG.source}")
print(f"bootstrap: {args.n_bootstrap} reamostragens · "
      f"poder: {args.n_simulacoes} simulações por tamanho de efeito")

out: dict = {"alfa": ALFA, "poder_alvo": PODER_ALVO, "por_horizonte": {}}
dados_fig: dict = {}

for h in bt.horizons:
    campeao = CFG.champion_for(h)
    print(f"\n{'=' * 70}\nHORIZONTE h={h} — {campeao} vs {BASELINE}\n{'=' * 70}")

    X, yh = make_supervised_frame(y, horizon=h, use_exogenous=CFG.use_exogenous,
                                  **CFG.feature_params)
    r_c = walk_forward_backtest(reg[campeao], X, yh, bt, h, campeao)
    r_b = walk_forward_backtest(reg[BASELINE], X, yh, bt, h, BASELINE)
    pc, pb = r_c.predictions, r_b.predictions
    datas = pc.index.intersection(pb.index)
    y_true = pc.loc[datas, "y_true"].to_numpy()
    p_camp = pc.loc[datas, "y_pred"].to_numpy()
    p_base = pb.loc[datas, "y_pred"].to_numpy()

    _, p_dm = diebold_mariano(y_true, p_camp, p_base, horizon=h)
    boot = intervalo_bootstrap(y_true, p_camp, p_base, horizon=h,
                               n_amostras=args.n_bootstrap, alfa=0.10,
                               seed=CFG.random_state)
    poder = efeito_minimo_detectavel(y_true, p_camp, p_base, horizon=h,
                                     n_simulacoes=args.n_simulacoes, alfa=ALFA,
                                     poder_alvo=PODER_ALVO, seed=CFG.random_state)

    print(f"   ganho observado : {boot['ganho_pct']:+.2f}%  "
          f"({boot['ganho_mae']:+.2f} pts de MAE)")
    print(f"   intervalo {boot['cobertura_pct']}%   : "
          f"{boot['ganho_pct_lo']:+.2f}% a {boot['ganho_pct_hi']:+.2f}%"
          f"   — inclui zero: {boot['inclui_zero']}")
    print(f"   blocos por reamostragem: {boot['n_blocos']} "
          f"(de {boot['n_pontos']} pontos)")
    mde_txt = "não atingido" if poder["mde_pct"] is None else f"{poder['mde_pct']:.0f}%"
    print(f"   efeito mínimo detectável (poder {PODER_ALVO:.0%}): {mde_txt}")
    print(f"   poder no ganho observado: {poder['poder_no_ganho_observado']:.0%}")
    print(f"   falso positivo em vantagem zero: {poder['taxa_falso_positivo']:.0%} "
          f"(esperado ~{ALFA:.0%})"
          + ("" if poder["bem_calibrado"] else "   <<< TESTE MAL CALIBRADO AQUI"))

    dados_fig[h] = {"boot": boot, "poder": poder, "campeao": campeao}
    registro = {k: v for k, v in boot.items() if k != "distribuicao_pct"}
    registro.update({"campeao": campeao, "dm_p": round(float(p_dm), 4)})
    registro["poder"] = poder
    out["por_horizonte"][str(h)] = registro

# ---------------------------------------------------------------- Fig 14
# Um intervalo mostra o que um p-valor esconde: o tamanho do ganho e a margem
# em volta dele. Ver o zero dentro da distribuição convence mais do que ler
# "p = 0,40" — e não exige que o leitor saiba o que é um p-valor.
n_h = len(bt.horizons)
fig, axes = plt.subplots(1, n_h, figsize=(3.3 * n_h, 3.2), sharey=False)
axes = np.atleast_1d(axes)
for ax, h in zip(axes, bt.horizons):
    b = dados_fig[h]["boot"]
    dist = b["distribuicao_pct"]
    dist = dist[np.isfinite(dist)]
    ax.hist(dist, bins=45, color=NAVY, alpha=0.75)
    ax.axvline(0, color=RED, lw=1.6, ls="--")
    ax.axvline(b["ganho_pct"], color=ORANGE, lw=2.0)
    ax.axvspan(b["ganho_pct_lo"], b["ganho_pct_hi"], color=ORANGE, alpha=0.15)
    veredito = "zero DENTRO" if b["inclui_zero"] else "zero FORA"
    ax.set_title(f"h={h} · {dados_fig[h]['campeao']}\n"
                 f"{b['ganho_pct']:+.1f}%  [{b['ganho_pct_lo']:+.1f}, "
                 f"{b['ganho_pct_hi']:+.1f}] — {veredito}",
                 fontsize=9, loc="left")
    ax.set_xlabel("Redução de erro vs. naive (%)")
axes[0].set_ylabel("Reamostragens")
cob = dados_fig[bt.horizons[0]]["boot"]["cobertura_pct"]
fig.suptitle(f"Quanto o ganho pode variar — bootstrap de blocos, intervalo {cob}%"
             "  (linha vermelha = ganho zero)", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "14_bootstrap_ganho.png"); plt.close(fig)

# ---------------------------------------------------------------- Fig 15
# "Não deu diferença" pode descrever os modelos ou descrever o experimento.
# A curva de poder separa os dois casos: ela diz qual vantagem esta amostra
# conseguiria enxergar, e onde o ganho observado cai nessa escala.
fig, ax = plt.subplots(figsize=(9.0, 3.8))
cores = [NAVY, ORANGE, GREEN, RED]
descalibrados = []
for (h, cor) in zip(bt.horizons, cores):
    pw = dados_fig[h]["poder"]
    curva = sorted(pw["curva_poder"].items())
    gs = [float(g) for g, _ in curva]
    ps = [float(p) for _, p in curva]
    mde = pw["mde_pct"]
    rotulo_h = (f"h={h}: observado {pw['ganho_observado_pct']:+.0f}% "
                f"→ poder {pw['poder_no_ganho_observado']:.0%}  |  "
                f"detecta a partir de "
                f"{'—' if mde is None else f'{mde:.0f}%'}")
    ax.plot(gs, ps, marker="o", ms=3.5, lw=1.7, color=cor, label=rotulo_h)
    ax.scatter([max(0.0, pw["ganho_observado_pct"])],
               [pw["poder_no_ganho_observado"]], s=75, facecolor="white",
               edgecolor=cor, lw=1.9, zorder=5)
    if not pw["bem_calibrado"]:
        descalibrados.append(f"h={h} ({pw['taxa_falso_positivo']:.0%}, "
                             f"{pw['n_blocos']} blocos)")
ax.axhline(PODER_ALVO, color=GREY, ls="--", lw=1.2)
ax.text(0.5, PODER_ALVO + 0.03, f"poder {PODER_ALVO:.0%}", color=GREY, fontsize=8)
ax.set_xlabel("Vantagem real imposta sobre o naive (%)   ·   círculo = ganho observado")
ax.set_ylabel("Chance de detectar")
ax.set_ylim(0, 1.05)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
# Em vantagem imposta zero a curva deveria partir de alfa. Onde ela parte bem
# acima, o teste inventa diferença — e isso precisa ir no rótulo, não numa nota.
if descalibrados:
    aviso = ("Atenção: em vantagem zero o teste deveria rejeitar "
             f"{ALFA:.0%} das vezes; rejeita mais em {', '.join(descalibrados)}")
else:
    aviso = f"Em vantagem zero o teste rejeita ~{ALFA:.0%}, como esperado"
ax.set_title(f"Este experimento enxergaria um ganho, se ele existisse?\n{aviso}",
             loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(FIG / "15_poder_do_teste.png"); plt.close(fig)

destino = CFG.uncertainty_path

print(f"\n{'=' * 70}\nLEITURA\n{'=' * 70}")
for h in bt.horizons:
    b, pw = dados_fig[h]["boot"], dados_fig[h]["poder"]
    if not b["inclui_zero"]:
        leitura = "ganho sustentado — o intervalo não contém zero."
    elif pw["mde_pct"] is None or pw["ganho_observado_pct"] < pw["mde_pct"]:
        leitura = (f"inconclusivo por TAMANHO DE AMOSTRA. O ganho observado "
                   f"({pw['ganho_observado_pct']:+.1f}%) está abaixo do que esta base "
                   f"consegue detectar ({pw['mde_pct']}%); 'empate' descreve o "
                   f"experimento, não os modelos.")
    else:
        leitura = ("sem ganho — a base detectaria uma vantagem deste tamanho "
                   "e não detectou.")
    print(f"  h={h}: {leitura}")
    out["por_horizonte"][str(h)]["leitura"] = leitura
    if not pw["bem_calibrado"]:
        print(f"        ressalva: com {pw['n_blocos']} blocos por reamostragem o "
              f"teste rejeita {pw['taxa_falso_positivo']:.0%} em vantagem zero; "
              f"leia o número acima como grosseiro.")

salvar_json(out, destino)
print(f"\nfiguras em {FIG}\nnúmeros em {destino}")
