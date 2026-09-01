"""Entrega final: um campeão por horizonte, com diagnóstico e explicação.

    python scripts/run_final.py
    python scripts/run_final.py --config caminho/outro.yaml

Por que um campeão por horizonte
--------------------------------
A previsão direta treina um modelo por horizonte, e a informação disponível
muda com ele. Em ``h=1`` o modelo conhece o mês que acabou de fechar; em
``h=12`` as defasagens legais são apenas 12, 13 e 24 — o mesmo eco sazonal que
o naive já usa. São problemas diferentes, e nada obriga o mesmo vencedor.
``config.yaml`` declara o campeão de cada horizonte.

Explicação segue o modelo, não a moda
-------------------------------------
- Campeão de árvore -> **SHAP** (TreeSHAP): atribuição local, que responde
  "por que ESTE mês foi previsto assim". Útil para quem precisa justificar um
  número de programação de produção.
- Campeão linear -> **coeficiente padronizado**, que já é a explicação exata,
  global e aditiva. Rodar SHAP sobre um modelo linear devolve exatamente
  ``coef_j * (x_j - média_j)``: a mesma informação, aproximada e mais cara.

Em ambos os casos vale o aviso registrado no relatório: com preditores
fortemente colineares (``lag_12`` e ``lag_24`` correlacionam 0,93), qualquer
atribuição reparte crédito entre variáveis quase intercambiáveis. SHAP não
cura isso — é a mesma limitação do coeficiente.

Todo texto de figura é derivado desta rodada. Nenhuma afirmação é fixa.
"""
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

from src.artifacts import proveniencia, salvar_json
from src.config import PipelineConfig
from src.evaluation.backtest import walk_forward_backtest
from src.evaluation.metrics import diebold_mariano
from src.features.build_features import (
    EXOGENOUS_COLS,
    legal_lags_for_horizon,
    make_calendar_features,
    make_lag_features,
    make_supervised_frame,
)
from src.models.registry import build_model_registry

# SHAP é opcional, como XGBoost e LightGBM: se não estiver instalado, a
# explicação cai para importância nativa do modelo e o pipeline segue.
try:
    import shap

    TEM_SHAP = True
except ImportError:  # pragma: no cover
    shap = None
    TEM_SHAP = False

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "figure.facecolor": "white", "axes.facecolor": "white"})
NAVY, ORANGE, GREY, GREEN, RED = "#1f3864", "#e07b39", "#8c8c8c", "#2e7d5b", "#b5342a"
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul",
         "Ago", "Set", "Out", "Nov", "Dez"]
BASELINE = "seasonal_naive"
ALFA = 0.05

ROTULOS = {
    "ridge_fourier": "Ridge + Fourier",
    "gradient_boosting": "Gradient Boosting",
    "hist_gradient_boosting": "HistGradientBoosting",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "seasonal_naive": "Naive sazonal",
    "seasonal_naive_drift": "Naive sazonal + drift",
}


def rotulo(nome: str) -> str:
    return ROTULOS.get(nome, nome.replace("_", " "))


def mes_ano(d) -> str:
    """Rótulo 'Set/2017'. Não usa strftime: %b depende do locale do sistema."""
    return f"{MESES[d.month - 1]}/{d.year}"


# ---------------------------------------------------------------- explicação
def explicar(modelo, X: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """Contribuição de cada variável, pelo método adequado ao modelo.

    Returns
    -------
    (importancia, subtitulo, rotulo_do_eixo)
        ``importancia`` já normalizada para somar 1, em ordem decrescente.
    """
    if hasattr(modelo, "named_steps") and hasattr(
        modelo.named_steps.get("model", None), "coef_"
    ):
        bruto = np.abs(modelo.named_steps["model"].coef_)
        return (
            pd.Series(bruto / bruto.sum(), index=X.columns).sort_values(ascending=False),
            "Magnitude relativa do coeficiente sobre variáveis padronizadas",
            "% da soma dos coeficientes (em módulo)",
        )

    if TEM_SHAP:
        try:
            valores = shap.TreeExplainer(modelo).shap_values(X)
            bruto = np.abs(np.asarray(valores)).mean(axis=0)
            return (
                pd.Series(bruto / bruto.sum(), index=X.columns).sort_values(ascending=False),
                "SHAP: contribuição média absoluta de cada variável (TreeSHAP)",
                "% da atribuição total",
            )
        # Catch amplo de propósito: o SHAP é um explicador opcional e levanta
        # tipos variados conforme o modelo não suportado. Qualquer falha aqui
        # tem de degradar para a importância nativa — nunca derrubar a entrega
        # por causa da figura de explicação.
        except Exception as exc:  # noqa: BLE001
            print(f"      SHAP indisponível para este modelo ({exc}); "
                  f"usando importância nativa.")

    bruto = np.asarray(modelo.feature_importances_, dtype=float)
    return (
        pd.Series(bruto / bruto.sum(), index=X.columns).sort_values(ascending=False),
        "Redução de erro atribuída a cada variável nas divisões das árvores",
        "% da redução total de erro",
    )


# ---------------------------------------------------------------- por horizonte
def rodar_horizonte(h: int, campeao: str, cfg_bt, reg) -> dict:
    """Backtest, diagnóstico, explicação e forecast de um horizonte."""
    print(f"\n{'=' * 70}\nHORIZONTE h={h} — campeão: {campeao}\n{'=' * 70}")
    saida: dict = {"horizonte": h, "campeao": campeao,
                   "campeao_rotulo": rotulo(campeao)}

    X, yh = make_supervised_frame(y, horizon=h, use_exogenous=CFG.use_exogenous, **FEATS)
    avaliar = dict.fromkeys([BASELINE, campeao])
    res = {n: walk_forward_backtest(reg[n], X, yh, cfg_bt, h, n) for n in avaliar}
    for n, r in res.items():
        print(f"   {n:24s} " + str({k: round(v, 3) for k, v in r.metrics.items()}))

    pc = res[campeao].predictions
    pb = res[BASELINE].predictions
    datas = pc.index.intersection(pb.index)
    stat, p = diebold_mariano(pc.loc[datas, "y_true"], pc.loc[datas, "y_pred"],
                              pb.loc[datas, "y_pred"], horizon=h)
    supera = bool(p < ALFA and stat < 0)
    saida["metricas"] = {n: {k: round(v, 3) for k, v in r.metrics.items()}
                         for n, r in res.items()}
    saida["dm_stat"], saida["dm_p"] = round(stat, 3), round(p, 4)
    saida["supera_baseline"] = supera

    # O veredito abaixo é o que decide se o modelo vai a produção. Ele é
    # calculado, não escrito: numa rodada futura em que o ganho desaparecer,
    # a figura passa a dizer que o baseline é suficiente.
    if supera:
        veredito = (f"{rotulo(campeao)} supera o naive sazonal "
                    f"(DM p={p:.3f}) — ganho sustentado por evidência")
    else:
        veredito = (f"Empate técnico com o naive sazonal (DM p={p:.3f}) — "
                    f"sem evidência de ganho neste horizonte")
    saida["veredito"] = veredito
    print(f"   -> {veredito}")

    # ---- Fig 6: realizado vs previsto
    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    ax.plot(pc.index, pc.y_true, color="black", lw=1.8, marker="o", ms=3, label="Realizado")
    ax.plot(pc.index, pc.y_pred, color=ORANGE, lw=1.6, marker="s", ms=3,
            label=f"{rotulo(campeao)} (campeão)")
    ax.plot(pb.index, pb.y_pred, color=GREY, lw=1.2, ls="--", label="Naive sazonal")
    ax.set_title(f"Backtest walk-forward h={h} — {cfg_bt.n_test} meses testados\n{veredito}",
                 loc="left", fontsize=10)
    ax.set_ylabel("Índice"); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / f"10_backtest_h{h}.png"); plt.close(fig)

    # ---- Fig 7: erro absoluto por mês
    ae_c = (pc.y_pred - pc.y_true).abs().groupby(pc.index.month).mean()
    ae_b = (pb.y_pred - pb.y_true).abs().groupby(pb.index.month).mean()
    comp = pd.DataFrame({"campeao": ae_c, "Naive": ae_b}).reindex(range(1, 13))
    ganho = comp["Naive"] - comp["campeao"]
    mes_melhor = MESES[int(ganho.idxmax()) - 1]
    piores = [MESES[m - 1] for m in ganho.index if ganho.loc[m] < 0]
    sub7 = f"Maior ganho em {mes_melhor} ({ganho.max():.1f} pts)"
    if piores:
        sub7 += f"; o naive ainda vence em {', '.join(piores)}"

    fig, ax = plt.subplots(figsize=(8.6, 3.1))
    w, xs = 0.38, np.arange(1, 13)
    ax.bar(xs - w / 2, comp["Naive"], width=w, color=GREY, label="Naive sazonal")
    ax.bar(xs + w / 2, comp["campeao"], width=w, color=ORANGE, label=rotulo(campeao))
    ax.set_xticks(xs); ax.set_xticklabels(MESES)
    ax.set_title(f"Erro absoluto médio por mês (backtest h={h})\n{sub7}",
                 loc="left", fontsize=10)
    ax.set_ylabel("Erro absoluto (pontos)"); ax.legend(frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(FIG / f"11_erro_por_mes_h{h}.png"); plt.close(fig)

    saida["erro_por_mes"] = {MESES[m - 1]: {"campeao": round(float(comp.loc[m, "campeao"]), 2),
                                            "naive": round(float(comp.loc[m, "Naive"]), 2)}
                             for m in range(1, 13)}
    saida["mes_maior_ganho"] = mes_melhor
    saida["meses_em_que_naive_vence"] = piores
    saida["vies_medio"] = round(float((pc.y_pred - pc.y_true).mean()), 2)

    # ---- Fig 8: explicação
    modelo = build_model_registry(random_state=CFG.random_state)[campeao]
    modelo.fit(X, yh)
    imp, sub8, eixo = explicar(modelo, X)
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.barh(top.index, top.values * 100,
            color=[ORANGE if v > 0.05 else NAVY for v in top.values])
    ax.set_title(f"Importância das variáveis — {rotulo(campeao)}, h={h}\n{sub8}",
                 loc="left", fontsize=10)
    ax.set_xlabel(eixo)
    fig.tight_layout(); fig.savefig(FIG / f"12_importancias_h{h}.png"); plt.close(fig)

    saida["metodo_explicacao"] = sub8
    saida["importancias"] = {k: round(float(v) * 100, 1) for k, v in imp.head(12).items()}
    # Importada, não redigitada: se a lista mudar num lugar só, este número
    # passa a somar um grupo diferente do que a matriz de fato contém.
    presentes = [c for c in EXOGENOUS_COLS if c in imp.index]
    saida["importancia_exogenas_total"] = (
        round(float(imp[presentes].sum()) * 100, 1) if presentes else None
    )
    eco = [c for c in ("lag_12", "lag_24", "yoy_diff_12") if c in imp.index]
    saida["importancia_eco_sazonal"] = round(float(imp[eco].sum()) * 100, 1)

    # ---- Fig 9: forecast
    # Um modelo de horizonte h prevê exatamente h meses à frente: para o alvo
    # T+i, a defasagem legal mais recente é y[T+i-h], observada para i <= h.
    futuro = pd.date_range(y.index.max() + pd.DateOffset(months=1), periods=h, freq="MS")
    y_ext = y.reindex(y.index.append(futuro))
    cal_f = make_calendar_features(y_ext.index, n_fourier=FEATS["n_fourier"])
    lag_f = make_lag_features(y_ext, lags=tuple(legal_lags_for_horizon(h, lags=FEATS["lags"])),
                              roll_windows=FEATS["roll_windows"], min_lag=h)
    Xf = pd.concat([cal_f, lag_f], axis=1).loc[futuro]
    assert Xf.notna().all().all(), f"features futuras incompletas em h={h}"
    fc = modelo.predict(Xf[X.columns])

    # O baseline também prevê o futuro, e não só o backtest. Quando ele é o
    # recomendado — h=3 e h=12 — a recomendação precisa existir como número
    # entregável: dizer "use o naive" e publicar apenas a previsão do modelo
    # reprovado deixa a decisão fora da entrega, e quem consome a camada gold
    # pega o número errado sem ter como saber.
    modelo_base = build_model_registry(random_state=CFG.random_state)[BASELINE]
    modelo_base.fit(X, yh)
    fc_base = modelo_base.predict(Xf[X.columns])
    recomendado = campeao if supera else BASELINE
    saida["modelo_recomendado"] = recomendado

    Q_LO, Q_HI = 0.10, 0.90
    cobertura = round((Q_HI - Q_LO) * 100)
    lo_q, hi_q = np.quantile((pc.y_true - pc.y_pred).to_numpy(), [Q_LO, Q_HI])
    # O intervalo de cada modelo sai dos erros DELE no backtest: aplicar a faixa
    # do campeão sobre a previsão do naive descreveria uma incerteza que não é a
    # dele.
    lo_b, hi_b = np.quantile((pb.y_true - pb.y_pred).to_numpy(), [Q_LO, Q_HI])
    saida["intervalo_baseline_lo"] = round(float(lo_b), 2)
    saida["intervalo_baseline_hi"] = round(float(hi_b), 2)
    saida["forecast_baseline"] = {
        mes_ano(d): round(float(v), 1) for d, v in zip(futuro, fc_base)
    }

    fig, ax = plt.subplots(figsize=(9.2, 3.2))
    hist = y[y.index >= y.index.max() - pd.DateOffset(years=5)]
    ax.plot(hist.index, hist.values, color=NAVY, lw=1.5, label="Histórico (5 anos)")
    ax.plot(futuro, fc, color=ORANGE, lw=2.0, marker="o", ms=4,
            label=f"Previsão ({h} {'mês' if h == 1 else 'meses'})")
    ax.fill_between(futuro, fc + lo_q, fc + hi_q, color=ORANGE, alpha=0.20,
                    label=f"Intervalo empírico {cobertura}%")
    ax.axvline(y.index.max(), color=GREY, ls=":", lw=1.2)
    janela = (mes_ano(futuro[0]) if h == 1
              else f"{mes_ano(futuro[0])} a {mes_ano(futuro[-1])}")
    ax.set_title(f"Previsão de {janela} — {rotulo(campeao)}\n"
                 "Faixa construída a partir dos erros reais do backtest",
                 loc="left", fontsize=10)
    ax.set_ylabel("Índice"); ax.legend(frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(FIG / f"13_forecast_h{h}.png"); plt.close(fig)

    saida["intervalo_cobertura_pct"] = cobertura
    saida["intervalo_lo"] = round(float(lo_q), 2)
    saida["intervalo_hi"] = round(float(hi_q), 2)
    saida["forecast"] = {mes_ano(d): round(float(v), 1) for d, v in zip(futuro, fc)}
    return saida


# ---------------------------------------------------------------- execução
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
_args = _ap.parse_args()

CFG = PipelineConfig.load(_args.config)
FIG = CFG.figures_dir
FEATS = CFG.feature_params

CFG.preparar_diretorios()
y = CFG.carregar_serie()
cfg_bt = CFG.backtest_config()
reg = build_model_registry(random_state=CFG.random_state)

print(f"configuração: {CFG.source}")
print(f"SHAP disponível: {TEM_SHAP}")
print(f"campeões por horizonte: {CFG.champions}")

# A proveniência entra no artefato porque semente igual não garante número
# igual: implementações de árvore mudam entre versões de scikit-learn. Sem
# registrar em que ambiente estes valores nasceram, uma divergência de casas
# decimais noutra máquina vira suspeita de erro em vez de informação.
out = {"campeoes": CFG.champions, "alfa": ALFA, "shap_disponivel": TEM_SHAP,
       "proveniencia": proveniencia(CFG.data_path),
       "por_horizonte": {}}
for h in cfg_bt.horizons:
    out["por_horizonte"][str(h)] = rodar_horizonte(h, CFG.champion_for(h), cfg_bt, reg)

# Conclusão agregada — também derivada, para não travar uma narrativa que a
# próxima rodada pode contradizer.
com_ganho = [h for h in cfg_bt.horizons
             if out["por_horizonte"][str(h)]["supera_baseline"]]
sem_ganho = [h for h in cfg_bt.horizons if h not in com_ganho]
out["horizontes_com_ganho"] = com_ganho
out["horizontes_sem_ganho"] = sem_ganho

salvar_json(out, CFG.metrics_path)

print(f"\n{'=' * 70}\nCONCLUSÃO\n{'=' * 70}")
if com_ganho:
    print(f"  Ganho sustentado por evidência em h={com_ganho}: modelo justificado.")
if sem_ganho:
    print(f"  Sem evidência de ganho em h={sem_ganho}: o naive sazonal é suficiente,")
    print("  e manter modelo nesse horizonte é custo sem retorno demonstrado.")
print(f"\nfiguras em {FIG}\nmétricas em {CFG.metrics_path}")
