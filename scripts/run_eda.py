"""EDA + diagnósticos. Gera todas as figuras e o JSON de números do relatório.

    python scripts/run_eda.py
    python scripts/run_eda.py --config caminho/outro.yaml

Caminhos e parâmetros vêm de config/config.yaml, resolvidos contra a raiz do
projeto — o script roda de qualquer diretório de trabalho e em qualquer SO.

Todo texto de figura é derivado dos dados desta rodada: períodos, meses
destacados e vereditos de teste são calculados, nunca escritos à mão. Uma
figura que afirma algo que a rodada não mostra é pior que uma figura sem
texto, porque parece verificada.

Escopo desta exploração — o que NÃO está aqui
---------------------------------------------
A EDA responde às perguntas de que a decisão precisava, e não tenta esgotar a
série. Ficaram de fora, deliberadamente: correlograma (ACF/PACF), testes
formais de estacionariedade, teste formal de quebra estrutural, decomposição
multiplicativa e a investigação individual das 24 anomalias.

O critério foi o mesmo do resto do projeto: uma análise entra quando muda uma
decisão. As quatro primeiras viram pré-requisito no momento em que um modelo
paramétrico (SARIMA, ETS) entrar como concorrente — estão no mesmo item do
roadmap por isso. O detalhamento de cada uma está no README, seção "A
exploração não está esgotada".
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
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.artifacts import formatar_json, salvar_json
from src.config import PipelineConfig
from src.data.calendar_features import (
    NBER_RECESSIONS,
    build_calendar_frame,
    easter_sunday,
)

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.facecolor": "white", "axes.facecolor": "white",
})
NAVY, ORANGE, GREY, GREEN, RED = "#1f3864", "#e07b39", "#8c8c8c", "#2e7d5b", "#b5342a"

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
_args = _ap.parse_args()

CFG = PipelineConfig.load(_args.config)
FIG = CFG.figures_dir

CFG.preparar_diretorios()
y = CFG.carregar_serie()
out = {}
out["n_obs"] = len(y)
out["start"] = str(y.index.min().date())
out["end"] = str(y.index.max().date())
out["min"] = round(float(y.min()), 2)
out["max"] = round(float(y.max()), 2)
out["mean"] = round(float(y.mean()), 2)

df = pd.DataFrame({"y": y})
df["year"], df["month"] = df.index.year, df.index.month

# Anos com os 12 meses observados. Toda estatística anual usa só estes: um ano
# parcial na ponta distorce média e CAGR, e a série quase nunca termina em
# dezembro. Derivar isso — em vez de fixar 2016 — é o que mantém o número certo
# quando a série for estendida no próximo ciclo.
_meses_por_ano = df.groupby("year").size()
ANOS_CHEIOS = _meses_por_ano[_meses_por_ano == 12].index
if len(ANOS_CHEIOS) < 3:
    raise ValueError(f"Apenas {len(ANOS_CHEIOS)} ano(s) completo(s): EDA anual não faz sentido.")
media_anual = df.groupby("year")["y"].mean().loc[ANOS_CHEIOS]
ANO_INI, ANO_FIM = int(ANOS_CHEIOS.min()), int(ANOS_CHEIOS.max())
PERIODO = f"{y.index.min().year}–{y.index.max().year}"

# ---------------------------------------------------------------- Fig 1: série
# Só sombreia as recessões que caem dentro da janela observada, e só menciona
# as faixas na legenda se alguma tiver sido desenhada.
recessoes = [(pd.Timestamp(a), pd.Timestamp(b)) for a, b in NBER_RECESSIONS
             if pd.Timestamp(b) >= y.index.min() and pd.Timestamp(a) <= y.index.max()]
fig, ax = plt.subplots(figsize=(9.2, 3.1))
# As faixas de recessão são estreitas — a de 1980 durou 6 meses, cerca de 1%
# da largura do eixo. Com opacidade baixa elas somem, e a legenda passa a
# prometer um sombreado que o leitor não encontra, o que é pior que não ter.
# Opacidade maior e uma borda fina resolvem sem competir com a série.
for a, b in recessoes:
    ax.axvspan(a, b, color=GREY, alpha=0.32, lw=0.6, edgecolor=GREY)
ax.plot(y.index, y.values, color=NAVY, lw=0.9)
ax.plot(y.index, y.rolling(12, center=True).mean(), color=ORANGE, lw=2.0,
        label="Média móvel 12 meses (tendência)")
sub1 = (f"{len(recessoes)} recessões NBER sombreadas em cinza" if recessoes
        else "Nenhuma recessão NBER catalogada nesta janela")
ax.set_title(f"Índice de produção — açúcar e confeitaria (EUA), {PERIODO}\n{sub1}",
             loc="left", fontsize=10)
ax.set_ylabel("Índice"); ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(FIG / "01_serie.png"); plt.close(fig)

# --------------------------------------------------- tendência e quebras
# O ponto de quebra é o ano de pico observado, não uma data escolhida à mão.
ANO_PICO = int(media_anual.idxmax())
out["periodo"] = PERIODO
out["primeiro_ano_cheio"], out["ultimo_ano_cheio"] = ANO_INI, ANO_FIM
out["peak_year"] = ANO_PICO
out["peak_year_value"] = round(float(media_anual.max()), 2)
out["valor_ultimo_ano_cheio"] = round(float(media_anual.loc[ANO_FIM]), 2)
out["drop_from_peak_pct"] = round(
    (out["valor_ultimo_ano_cheio"] / out["peak_year_value"] - 1) * 100, 2)


def cagr(v_ini: float, v_fim: float, n_anos: int) -> float | None:
    """Taxa composta anual. None quando o intervalo é curto demais para significar algo."""
    if n_anos < 1 or v_ini <= 0:
        return None
    return round(((v_fim / v_ini) ** (1 / n_anos) - 1) * 100, 2)


out["cagr_ate_pico"] = cagr(float(media_anual.loc[ANO_INI]),
                            float(media_anual.loc[ANO_PICO]), ANO_PICO - ANO_INI)
out["cagr_apos_pico"] = cagr(float(media_anual.loc[ANO_PICO]),
                             float(media_anual.loc[ANO_FIM]), ANO_FIM - ANO_PICO)

# ------------------------------------------- Fig 2: decomposição
trend_f = y.rolling(12, center=True).mean()
detr = y - trend_f
seas = detr.groupby(detr.index.month).mean()
seas_full = pd.Series(detr.index.month, index=detr.index).map(seas)
resid = y - trend_f - seas_full

fig, axes = plt.subplots(4, 1, figsize=(9.2, 7.2), sharex=True)
axes[0].plot(y.index, y.values, color=NAVY, lw=0.8); axes[0].set_ylabel("Observado")
axes[1].plot(trend_f.index, trend_f.values, color=ORANGE, lw=1.4); axes[1].set_ylabel("Tendência")
axes[2].plot(seas_full.index, seas_full.values, color=GREEN, lw=0.7); axes[2].set_ylabel("Sazonal")
axes[3].plot(resid.index, resid.values, color=GREY, lw=0.7); axes[3].axhline(0, color="k", lw=0.6)
axes[3].set_ylabel("Resíduo")
axes[0].set_title("Decomposição aditiva: observado = tendência + sazonalidade + resíduo",
                  loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(FIG / "02_decomposicao.png"); plt.close(fig)

var_seas = float(np.nanvar(seas_full)); var_res = float(np.nanvar(resid))
var_trend = float(np.nanvar(trend_f.diff()))
out["var_share_seasonal"] = round(var_seas / (var_seas + var_res) * 100, 1)
out["seasonal_amplitude"] = round(float(seas.max() - seas.min()), 2)
out["resid_std"] = round(float(np.nanstd(resid)), 2)

# ------------------------------------------- Fig 3: perfil sazonal
MESES = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun", "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
fig, ax = plt.subplots(figsize=(8.4, 3.2))
data = [detr[detr.index.month == m].dropna().to_numpy() for m in range(1, 13)]
# Os rótulos são postos com set_xticklabels e não com o argumento `labels=`:
# ele foi renomeado em matplotlib 3.9 e removido em 3.11.
bp = ax.boxplot(data, patch_artist=True, widths=0.6,
                medianprops={"color": "white", "lw": 1.6},
                flierprops={"marker": "o", "ms": 2.5, "mfc": GREY,
                            "mec": "none", "alpha": 0.6})
# O destaque em laranja é medido, não escolhido: são os meses cujo desvio
# sazonal médio fica acima da tendência. Se o perfil da série mudar, o
# destaque e a legenda mudam junto.
meses_altos = [m for m in range(1, 13) if seas[m] > 0]
for i, box in enumerate(bp["boxes"]):
    box.set(facecolor=ORANGE if (i + 1) in meses_altos else NAVY, alpha=0.9, lw=0)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MESES)
ax.axhline(0, color="k", lw=0.8, ls="--")
sub3 = ("Laranja = meses acima da tendência ("
        + ", ".join(MESES[m - 1] for m in meses_altos) + ")") if meses_altos else \
       "Nenhum mês fica acima da tendência"
ax.set_title(f"Desvio sazonal por mês (série sem tendência), {PERIODO}\n{sub3}",
             loc="left", fontsize=10)
ax.set_ylabel("Pontos de índice vs. tendência")
fig.tight_layout(); fig.savefig(FIG / "03_perfil_sazonal.png"); plt.close(fig)

out["seasonal_profile"] = {MESES[m - 1]: round(float(seas[m]), 2) for m in range(1, 13)}
out["max_month"] = MESES[int(seas.idxmax()) - 1]
out["min_month"] = MESES[int(seas.idxmin()) - 1]

# ------------------------------- Fig 4: estabilidade da sazonalidade por período
# Os blocos são fatias iguais do histórico observado, não décadas escritas à
# mão: estender a série em um ano não pode deixar o último bloco truncado nem
# a legenda mentindo sobre o intervalo que cada curva cobre.
N_BLOCOS = 4
blocos = np.array_split(np.array(sorted(df["year"].unique())), N_BLOCOS)
periodos = {f"{int(b[0])}–{int(b[-1])}": (int(b[0]), int(b[-1])) for b in blocos}

prof = {}
for lbl, (a, b) in periodos.items():
    sub = detr[(detr.index.year >= a) & (detr.index.year <= b)]
    prof[lbl] = sub.groupby(sub.index.month).mean()

amp = {k: round(float(v.max() - v.min()), 2) for k, v in prof.items()}
corr = pd.DataFrame(prof).corr()
corr_min = float(corr.to_numpy()[np.triu_indices_from(corr, 1)].min())
rotulos = list(prof)
amp_var_pct = (amp[rotulos[-1]] / amp[rotulos[0]] - 1) * 100
n_anos = y.index.max().year - y.index.min().year

# As duas metades do subtítulo são conclusões medidas: quão parecidos são os
# perfis (correlação mínima entre blocos) e para onde foi a amplitude.
forma = ("o mesmo padrão" if corr_min >= 0.8
         else "padrão parcialmente estável" if corr_min >= 0.5
         else "o padrão MUDOU")
if abs(amp_var_pct) < 5:
    tamanho = "e a amplitude é estável"
else:
    tamanho = f"e a amplitude {'caiu' if amp_var_pct < 0 else 'cresceu'} {abs(amp_var_pct):.0f}%"
sub4 = (f"{forma} há {n_anos} anos (correlação mínima entre blocos "
        f"{corr_min:.2f}) — {tamanho}")
sub4 = sub4[0].upper() + sub4[1:]

fig, ax = plt.subplots(figsize=(8.4, 3.2))
for (lbl, p), c in zip(prof.items(), [NAVY, ORANGE, GREEN, RED]):
    ax.plot(range(1, 13), p.values, marker="o", ms=3.5, color=c, lw=1.6, label=lbl)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MESES)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_title(f"Estabilidade do padrão sazonal ao longo do histórico\n{sub4}",
             loc="left", fontsize=10)
ax.set_ylabel("Pontos de índice"); ax.legend(frameon=False, ncol=N_BLOCOS, fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "04_estabilidade_sazonal.png"); plt.close(fig)

out["amplitude_por_periodo"] = amp
out["corr_sazonal_min"] = round(corr_min, 3)
out["amplitude_variacao_pct"] = round(amp_var_pct, 1)

# ---------------------------- Fig 5: efeito Páscoa (o teste que falhou)
cal = build_calendar_frame(y.index)
d2 = cal.join(y)
d2["year"], d2["month"] = d2.index.year, d2.index.month
# Mesma razão da normalização acima: ano parcial distorce a própria referência.
d2["norm"] = np.where(
    d2["year"].isin(ANOS_CHEIOS),
    d2["production_index"] / d2.groupby("year")["production_index"].transform("mean") * 100,
    np.nan,
)
mar_years = set(d2[(d2.month == 3) & (d2.easter_month == 1)].year)

# O veredito do título vem de um teste de Welch, não de uma frase escrita à
# mão. Se numa rodada futura a Páscoa passar a deslocar a produção, o título
# passa a dizer isso — que é o requisito para esta figura ir a produção.
ALFA = 0.05
teste_pascoa = {}
for mth, nome in [(2, "fev"), (3, "mar"), (4, "abr")]:
    sub = d2[d2.month == mth]
    # `dropna` explícito, e não por descuido: `norm` é nula nos anos parciais,
    # porque normalizar contra a média de um ano incompleto produz referência
    # errada. O ano corrente cai aqui sempre — a série quase nunca termina em
    # dezembro.
    #
    # Sem isso, o padrão do scipy é `nan_policy="propagate"`: UMA observação
    # nula devolve p = NaN para os 46 anos. E como `NaN < 0.05` é False, o
    # veredito sairia "nenhum mês significativo" — a conclusão certa pelo
    # motivo errado, que é o pior desfecho possível para uma hipótese que o
    # relatório apresenta como refutada por teste.
    g1 = sub[sub.year.isin(mar_years)]["norm"].dropna()
    g2 = sub[~sub.year.isin(mar_years)]["norm"].dropna()
    if len(g1) >= 2 and len(g2) >= 2:
        t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
    else:
        t_stat, p_val = np.nan, np.nan
    teste_pascoa[mth] = {"nome": nome, "delta": float(g1.mean() - g2.mean()),
                         "p": float(p_val), "g1": g1, "g2": g2}
    out[f"pascoa_delta_{nome}"] = round(teste_pascoa[mth]["delta"], 2)
    out[f"pascoa_p_{nome}"] = round(float(p_val), 4)
    out[f"pascoa_std_{nome}"] = round(float(sub["norm"].std()), 2)
    # Os n vão para o JSON: um p-valor sem o tamanho dos grupos que o
    # produziram não é auditável, e é justamente onde este bloco falhou antes.
    out[f"pascoa_n_marco_{nome}"] = len(g1)
    out[f"pascoa_n_abril_{nome}"] = len(g2)
out["n_pascoa_marco"] = len(mar_years)
out["pascoa_alfa"] = ALFA

# Guarda de sanidade. `NaN < ALFA` é False, então um teste que não rodou
# entraria silenciosamente na lista dos "não significativos" e viraria a
# conclusão "nenhum deslocamento detectável" — indistinguível de um teste que
# rodou e não achou nada. Falhar aqui é a única forma de essas duas coisas não
# se confundirem no relatório.
sem_p = [t["nome"] for t in teste_pascoa.values() if not np.isfinite(t["p"])]
if sem_p:
    raise ValueError(
        f"Teste de Páscoa não produziu p-valor em {sem_p}. Um p ausente NÃO é "
        f"'sem efeito': é teste que não rodou. Verifique grupos vazios ou nulos "
        f"em `norm` antes de reportar a hipótese como refutada."
    )

sig = [t["nome"] for t in teste_pascoa.values() if t["p"] < ALFA]
out["pascoa_meses_significativos"] = sig
if sig:
    verdito_pascoa = (f"a data móvel DESLOCA a produção em {', '.join(sig)} "
                      f"(Welch, p < {ALFA})")
else:
    p_min = min(t["p"] for t in teste_pascoa.values())
    verdito_pascoa = (f"nenhum deslocamento detectável (Welch, menor p = {p_min:.2f}; "
                      f"n = {len(mar_years)} anos com Páscoa em março)")

fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.1))
for ax_, mth, name in zip(axes, [3, 4], ["Março", "Abril"]):
    t = teste_pascoa[mth]
    ax_.boxplot([t["g1"].to_numpy(), t["g2"].to_numpy()],
                patch_artist=True, widths=0.5,
                boxprops={"facecolor": NAVY, "alpha": 0.85, "lw": 0},
                medianprops={"color": "white", "lw": 1.6})
    ax_.set_xticks([1, 2])
    ax_.set_xticklabels(["Páscoa\nem março", "Páscoa\nem abril"])
    ax_.set_title(f"{name}  (Δ = {t['delta']:+.2f} pts, p = {t['p']:.2f})",
                  fontsize=9.5, loc="left")
    ax_.set_ylabel("Índice normalizado (ano = 100)")
axes[0].figure.suptitle(f"Teste do efeito Páscoa: {verdito_pascoa}",
                        fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "05_teste_pascoa.png"); plt.close(fig)

d2["resid_month"] = d2["norm"] - d2.groupby("month")["norm"].transform("mean")
out["corr_dias_uteis"] = round(float(d2["resid_month"].corr(d2["n_business_days"])), 3)
out["dias_uteis_range"] = [int(d2.n_business_days.min()), int(d2.n_business_days.max())]

# ---------------------------- Fig 6: rampa da temporada
# O perfil sazonal (fig 3) mostra ONDE a produção é alta. Esta mostra QUANDO ela
# sobe — que é a informação de planejamento. O maior salto do ano é o momento em
# que a capacidade já precisa estar contratada.
# A normalização divide cada mês pela média do próprio ano. Num ano incompleto
# essa média cobre só parte da temporada — a série termina em agosto, então a
# média de 2017 ignora o pico de out-dez e infla artificialmente jan-ago. Os
# anos parciais entram como NaN e somem das médias a jusante.
df["norm"] = np.where(
    df["year"].isin(ANOS_CHEIOS),
    df["y"] / df.groupby("year")["y"].transform("mean") * 100,
    np.nan,
)
share = df.groupby("month")["norm"].mean()
salto = pd.Series({m: share[m] - share[12 if m == 1 else m - 1] for m in range(1, 13)})
mes_salto = int(salto.idxmax())
mes_queda = int(salto.idxmin())
mes_pico_nivel = int(share.idxmax())
tot_share = float(share.sum())
share_temporada = float(share[[9, 10, 11, 12]].sum()) / tot_share * 100

sub6 = (f"Maior salto: {MESES[mes_salto - 1]} ({salto.max():+.1f} p.p. da média "
        f"anual) — o pico de nível é {MESES[mes_pico_nivel - 1]}")

fig, ax = plt.subplots(figsize=(9.2, 3.3))
cores = [ORANGE if v > 0 else NAVY for v in salto.to_numpy()]
ax.bar(range(1, 13), salto.values, color=cores, alpha=0.9, width=0.62,
       label="Variação vs. mês anterior")
ax2 = ax.twinx()
ax2.plot(range(1, 13), share.values, color=GREY, lw=1.8, marker="o", ms=4,
         label="Nível (% da média anual)")
ax2.set_ylabel("% da média anual", color=GREY)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(MESES)
ax.set_ylabel("Variação (p.p.)")
ax.set_title(f"Rampa da temporada — quando a produção sobe, não quando é alta\n{sub6}",
             loc="left", fontsize=10)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, frameon=False, ncol=2, loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "06_rampa_temporada.png"); plt.close(fig)

out["rampa_por_mes"] = {MESES[m - 1]: round(float(salto[m]), 2) for m in range(1, 13)}
out["mes_maior_salto"] = MESES[mes_salto - 1]
out["mes_maior_queda"] = MESES[mes_queda - 1]
out["mes_pico_nivel"] = MESES[mes_pico_nivel - 1]
out["share_set_dez_pct"] = round(share_temporada, 1)

# ---------------------------- Fig 7: Páscoa, teste contínuo
# A fig 5 compara dois grupos (Páscoa em março vs abril). Esta usa a data exata:
# se a Páscoa móvel deslocasse a produção, o dia do ano em que ela cai teria de
# explicar parte da variação mensal. É o teste mais sensível dos dois.
d2["pascoa_doy"] = [easter_sunday(a).timetuple().tm_yday for a in d2["year"]]
reg_pascoa = {}
for mth in (2, 3, 4):
    sub = d2[d2.month == mth].dropna(subset=["norm", "pascoa_doy"])
    r = stats.linregress(sub["pascoa_doy"], sub["norm"])
    reg_pascoa[mth] = {"sub": sub, "r": r}
    out[f"pascoa_reg_coef_{MESES[mth-1].lower()}"] = round(float(r.slope), 4)
    out[f"pascoa_reg_r2_{MESES[mth-1].lower()}"] = round(float(r.rvalue ** 2), 4)
    out[f"pascoa_reg_p_{MESES[mth-1].lower()}"] = round(float(r.pvalue), 4)

sig_reg = [MESES[m - 1] for m, v in reg_pascoa.items() if v["r"].pvalue < ALFA]
amplitude_doy = int(d2["pascoa_doy"].max() - d2["pascoa_doy"].min())
if sig_reg:
    sub7 = (f"A data da Páscoa explica a produção em {', '.join(sig_reg)} "
            f"(regressão, p < {ALFA})")
else:
    r2_max = max(v["r"].rvalue ** 2 for v in reg_pascoa.values())
    sub7 = (f"{amplitude_doy} dias de variação na data não explicam nada: "
            f"maior R² = {r2_max:.3f}, nenhum p < {ALFA}")
out["pascoa_reg_meses_significativos"] = sig_reg
out["pascoa_amplitude_dias"] = amplitude_doy

fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharey=True)
for ax_, mth in zip(axes, (2, 3, 4)):
    sub, r = reg_pascoa[mth]["sub"], reg_pascoa[mth]["r"]
    ax_.scatter(sub["pascoa_doy"], sub["norm"], s=18, color=NAVY, alpha=0.65)
    xs = np.array([sub["pascoa_doy"].min(), sub["pascoa_doy"].max()])
    cor_linha = RED if r.pvalue < ALFA else GREY
    ax_.plot(xs, r.intercept + r.slope * xs, color=cor_linha, lw=1.8)
    ax_.set_title(f"{MESES[mth-1]}  (R² = {r.rvalue**2:.3f}, p = {r.pvalue:.2f})",
                  fontsize=9.5, loc="left")
    ax_.set_xlabel("Dia do ano da Páscoa")
axes[0].set_ylabel("Índice normalizado (ano = 100)")
axes[0].figure.suptitle(f"Efeito Páscoa, teste contínuo: {sub7}",
                        fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "07_pascoa_continuo.png"); plt.close(fig)

# ---------------------------- Fig 8: anomalias
# Removendo tendência E o efeito do mês, o que sobra é evento — choque que o
# calendário não explica. Onde esses choques se concentram é onde o S&OP corre
# risco, porque é onde a previsão erra mais e o erro custa mais.
Z_CORTE = 2.0
resid_evt = (detr - pd.Series(detr.index.month, index=detr.index).map(seas)).dropna()
z = resid_evt / resid_evt.std()
graves = z[z.abs() > Z_CORTE]
cont_mes = graves.groupby(graves.index.month).size().reindex(range(1, 13), fill_value=0)
top_meses = cont_mes.sort_values(ascending=False)
top3 = [int(m) for m in top_meses.index[:3] if top_meses[m] > 0]
share_pico = (cont_mes[[9, 10, 11, 12]].sum() / len(graves) * 100) if len(graves) else 0.0

sub8 = (f"{len(graves)} meses fora de ±{Z_CORTE:.0f} desvios "
        f"({len(graves)/len(z)*100:.1f}% da série); "
        f"{share_pico:.0f}% deles caem entre Set e Dez")

fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2),
                         gridspec_kw={"width_ratios": [2.4, 1]})
for a, b in recessoes:
    axes[0].axvspan(a, b, color=GREY, alpha=0.18, lw=0)
axes[0].plot(z.index, z.values, color=NAVY, lw=0.7)
axes[0].scatter(graves.index, graves.values, s=22, color=RED, zorder=3)
for lim in (-Z_CORTE, Z_CORTE):
    axes[0].axhline(lim, color=RED, ls="--", lw=0.9, alpha=0.7)
axes[0].axhline(0, color="k", lw=0.6)
axes[0].set_ylabel("Desvios-padrão")
axes[0].set_title("Choques não explicados por tendência nem por sazonalidade",
                  fontsize=9.5, loc="left")

cores_mes = [ORANGE if m in top3 else NAVY for m in range(1, 13)]
axes[1].bar(range(1, 13), cont_mes.values, color=cores_mes, alpha=0.9)
axes[1].set_xticks(range(1, 13))
axes[1].set_xticklabels([m[0] for m in MESES], fontsize=7.5)
axes[1].set_title("Concentração por mês", fontsize=9.5, loc="left")
axes[1].set_ylabel("nº de anomalias")
axes[0].figure.suptitle(f"Anomalias: {sub8}", fontsize=10, x=0.01, ha="left")
fig.tight_layout(); fig.savefig(FIG / "08_anomalias.png"); plt.close(fig)

# ---------------------------- Fig 9: o programa açucareiro explica a estagnação?
# HIPÓTESE TESTADA E REFUTADA. O contexto histórico é real — Kraft moveu a
# produção de Life Savers para o Canadá em 2002, Spangler moveu metade para o
# México em 2003, ambos citando o preço do açúcar. A tentação é concluir que o
# prêmio do açúcar americano explica a estagnação do pico da série.
#
# Os dados de preço dizem o contrário, e este bloco existe para registrar isso
# em vez de escondê-lo. Vale mais um teste honesto que refuta uma hipótese
# própria do que uma narrativa que só cita jornal.
if CFG.tem_precos_acucar:
    def _serie_fred(caminho):
        d = pd.read_csv(caminho)
        d["observation_date"] = pd.to_datetime(d["observation_date"])
        col = next(c for c in d.columns if c != "observation_date")
        return d.set_index("observation_date")[col].astype(float)

    p_mundo = _serie_fred(CFG.sugar_world_path)
    p_eua = _serie_fred(CFG.sugar_us_path)
    precos = pd.DataFrame({"mundo": p_mundo, "eua": p_eua}).dropna()
    precos["premio"] = precos["eua"] - precos["mundo"]
    precos["razao"] = precos["eua"] / precos["mundo"]
    precos["ano"] = precos.index.year

    # Comparação anual: a produção tem 32 pontos de amplitude sazonal e o preço
    # não tem nenhuma. Correlacionar mês a mês mediria sazonalidade, não relação.
    anual_preco = precos.groupby("ano").agg(
        eua=("eua", "mean"), mundo=("mundo", "mean"),
        premio=("premio", "mean"), razao=("razao", "mean"))
    comp = anual_preco.join(media_anual.rename("producao"), how="inner").dropna()

    correls = {}
    for nome, col in [("premio", "premio"), ("razao", "razao"),
                      ("preco_eua", "eua"), ("preco_mundial", "mundo")]:
        r = stats.linregress(comp[col], comp["producao"])
        correls[nome] = {"r": float(r.rvalue), "p": float(r.pvalue)}

    r_premio = correls["premio"]["r"]
    r_mundo = correls["preco_mundial"]["r"]
    # A hipótese previa correlação NEGATIVA entre prêmio e produção: açúcar
    # doméstico caro empurraria a fabricação para fora. Sinal positivo a refuta.
    hipotese_confirmada = r_premio < 0 and correls["premio"]["p"] < ALFA
    if hipotese_confirmada:
        sub9 = (f"Sim — prêmio maior acompanha produção menor "
                f"(r = {r_premio:+.2f})")
    else:
        sub9 = (f"NÃO — prêmio maior acompanha produção MAIOR (r = {r_premio:+.2f});\n"
                f"quem se correlaciona com a produção é o preço MUNDIAL "
                f"(r = {r_mundo:+.2f}), não a política doméstica")

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.7),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    axes[0].plot(precos.index, precos["eua"], color=NAVY, lw=1.4, label="EUA (No. 16)")
    axes[0].plot(precos.index, precos["mundo"], color=ORANGE, lw=1.4,
                 label="mundial (No. 11)")
    axes[0].fill_between(precos.index, precos["mundo"], precos["eua"],
                         color=GREY, alpha=0.18)
    axes[0].axvspan(pd.Timestamp("2002-01-01"), pd.Timestamp("2003-12-31"),
                    color=RED, alpha=0.15, lw=0)
    axes[0].text(pd.Timestamp("2003-06-01"), precos["eua"].max() * 0.97,
                 "fábricas saem", fontsize=7.5, color=RED, ha="center")
    # A série de preço vai além do fim da produção. Marcar onde ela termina evita
    # que alguém leia a parte direita do gráfico como se houvesse produção ali.
    axes[0].axvline(y.index.max(), color=NAVY, ls=":", lw=1.2)
    axes[0].text(y.index.max(), precos["eua"].min(), " fim da série\n de produção",
                 fontsize=7, color=NAVY, va="bottom")
    axes[0].set_ylabel("centavos de dólar por libra")
    axes[0].set_title(f"Prêmio máximo: {anual_preco['razao'].max():.1f}× em "
                      f"{int(anual_preco['razao'].idxmax())}",
                      fontsize=9.5, loc="left")
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    axes[1].scatter(comp["premio"], comp["producao"], s=26, color=NAVY, alpha=0.75)
    rr = stats.linregress(comp["premio"], comp["producao"])
    xs = np.array([comp["premio"].min(), comp["premio"].max()])
    axes[1].plot(xs, rr.intercept + rr.slope * xs, color=RED, lw=1.7)
    axes[1].set_xlabel("prêmio médio no ano (EUA − mundial)")
    axes[1].set_ylabel("produção média no ano")
    axes[1].set_title(f"r = {r_premio:+.2f}  (n = {len(comp)} anos)",
                      fontsize=9.5, loc="left")

    axes[0].figure.suptitle(f"O programa açucareiro explica a estagnação? {sub9}",
                            fontsize=10, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(FIG / "09_precos_acucar.png"); plt.close(fig)

    out["acucar_hipotese_confirmada"] = bool(hipotese_confirmada)
    out["acucar_correlacoes"] = {k: {"r": round(v["r"], 3), "p": round(v["p"], 4)}
                                 for k, v in correls.items()}
    out["acucar_anos_comuns"] = len(comp)
    out["acucar_periodo"] = f"{int(comp.index.min())}-{int(comp.index.max())}"
    out["acucar_razao_max"] = round(float(anual_preco["razao"].max()), 2)
    out["acucar_ano_razao_max"] = int(anual_preco["razao"].idxmax())
else:
    print("séries de preço do açúcar ausentes — seção pulada (figura 09)")
    out["acucar_hipotese_confirmada"] = None

out["anomalia_z_corte"] = Z_CORTE
out["anomalia_desvio_padrao"] = round(float(resid_evt.std()), 2)
out["anomalias_total"] = len(graves)
out["anomalias_pct_serie"] = round(len(graves) / len(z) * 100, 1)
out["anomalias_share_set_dez_pct"] = round(float(share_pico), 1)
out["anomalias_por_mes"] = {MESES[m - 1]: int(cont_mes[m]) for m in range(1, 13)}
out["anomalias_maiores"] = {
    f"{d:%Y-%m}": round(float(v), 2)
    for d, v in z.reindex(z.abs().sort_values(ascending=False).index).head(10).items()
}

salvar_json(out, CFG.eda_path)
print(formatar_json(out))
print(f"\nfiguras em {FIG}\nnúmeros em {CFG.eda_path}")
