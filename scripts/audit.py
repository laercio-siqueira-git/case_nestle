"""
Auditoria independente do pipeline.

Este script existe para que ninguém precise confiar na documentação. Ele
reproduz, do zero, as quatro verificações que sustentam as afirmações do
relatório, e imprime PASSOU ou FALHOU para cada uma.

    python scripts/audit.py

Se algum bloco falhar, uma afirmação do relatório deixou de ser verdadeira.

Blocos
------
1. Proveniência das features — descobre, por perturbação, de quais defasagens
   de ``y`` cada coluna realmente depende, em cada horizonte.
2. Efeito do embargo — quantifica o vazamento do protocolo antigo e roda o
   experimento de controle que separa "recência" de "volume de dados".
3. Forecast não recursivo — prova que os 12 meses futuros não consomem
   nenhuma previsão como insumo.
4. Comportamento do MAPE — mostra por que ele subpesa o erro no pico.

Tempo de execução: ~3 a 6 minutos (o bloco 2 retreina modelos muitas vezes).
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.config import PipelineConfig
from src.evaluation.backtest import train_mask_for
from src.evaluation.metrics import evaluate
from src.features.build_features import (
    legal_lags_for_horizon,
    make_calendar_features,
    make_lag_features,
    make_supervised_frame,
)
from src.models.registry import available_boosters, build_model_registry

# Toda a parametrização vem de config/config.yaml — a auditoria tem de
# verificar o pipeline tal como ele está configurado, não uma cópia dos
# números que pode divergir em silêncio.
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--config", default=None, help="caminho alternativo para o YAML")
_args = _ap.parse_args()

CFG = PipelineConfig.load(_args.config)
BT = CFG.backtest_config()
FEATS = CFG.feature_params
N_TEST, MIN_TRAIN = BT.n_test, BT.min_train
results: dict[str, bool] = {}


def header(txt: str) -> None:
    print(f"\n{'=' * 78}\n{txt}\n{'=' * 78}")


def verdict(name: str, passed: bool, detail: str = "") -> None:
    results[name] = passed
    print(f"\n  >>> {name}: {'PASSOU' if passed else 'FALHOU'} {detail}")


# ---------------------------------------------------------------------------
def audit_feature_provenance(y: pd.Series) -> None:
    """Bloco 1 — de quais defasagens de y cada coluna depende, de fato.

    Método: perturbar ``y[t-k]`` isoladamente e observar quais colunas da
    linha ``t`` mudam. Isso mapeia a dependência real do código executado, e
    não a dependência que a documentação afirma existir.
    """
    header("BLOCO 1 — PROVENIÊNCIA DAS FEATURES (teste de perturbação)")
    all_ok = True

    for horizon in BT.horizons:
        X, _ = make_supervised_frame(
            y, horizon=horizon, use_exogenous=CFG.use_exogenous, **FEATS
        )
        target = X.index[-1]
        base = X.loc[target]
        violations = []

        print(f"\n  h={horizon:2d} | {X.shape[1]} colunas | linha auditada: {target.date()}")
        for col in X.columns:
            deps = []
            for k in range(30):
                date_k = target - pd.DateOffset(months=k)
                if date_k not in y.index:
                    continue
                perturbed = y.copy()
                perturbed.loc[date_k] += 1000.0
                Xp, _ = make_supervised_frame(
                    perturbed, horizon=horizon, use_exogenous=CFG.use_exogenous, **FEATS
                )
                if abs(float(Xp.loc[target, col]) - float(base[col])) > 1e-9:
                    deps.append(k)
            if deps and min(deps) < horizon:
                violations.append((col, min(deps)))
                print(f"      {col:26s} depende de k={min(deps)} < h={horizon}  <<< VIOLAÇÃO")

        if violations:
            all_ok = False
            print(f"      {len(violations)} violação(ões)")
        else:
            print("      nenhuma coluna depende de defasagem menor que o horizonte")

    verdict("Bloco 1 — proveniência", all_ok)


# ---------------------------------------------------------------------------
def audit_embargo(y: pd.Series) -> None:
    """Bloco 2 — quanto o protocolo antigo inflava o resultado.

    Roda o mesmo modelo em quatro configurações. As duas últimas são o
    controle que separa duas explicações concorrentes: o protocolo antigo
    ganhava por ter *mais linhas* de treino, ou por ter linhas *mais
    recentes*? Se for volume, remover 11 linhas quaisquer deve doer igual.
    """
    header("BLOCO 2 — EFEITO DO EMBARGO (com experimento de controle)")
    horizon = max(BT.horizons)
    X, yh = make_supervised_frame(
        y, horizon=horizon, use_exogenous=CFG.use_exogenous, **FEATS
    )
    model = build_model_registry(
        random_state=CFG.random_state, include_optional=False
    )["gradient_boosting"]
    test_index = yh.index[-N_TEST:]
    y_train_ref = yh[yh.index < test_index[0]].to_numpy()
    rng = np.random.default_rng(7)

    def run(mask_fn, label: str) -> dict[str, float]:
        rows = []
        for cutoff in test_index:
            mask = mask_fn(cutoff)
            if mask.sum() < MIN_TRAIN:
                continue
            est = clone(model)
            est.fit(X.loc[mask], yh.loc[mask])
            rows.append(
                (float(yh.loc[cutoff]), float(np.ravel(est.predict(X.loc[[cutoff]]))[0]),
                 int(mask.sum()))
            )
        arr = np.array([r[:2] for r in rows])
        met = evaluate(arr[:, 0], arr[:, 1], y_train=y_train_ref, season=BT.season)
        print(f"    {label:48s} MAPE={met['MAPE']:6.3f}%  MASE={met['MASE']:.3f}  n={rows[-1][2]}")
        return met

    print()
    m_leaky = run(lambda c: train_mask_for(yh.index, c, horizon, embargo=False),
                  "A) protocolo antigo (alvo < cutoff)")
    m_clean = run(lambda c: train_mask_for(yh.index, c, horizon, embargo=True),
                  "B) embargo correto (alvo <= origem)")

    # O embargo remove exatamente h-1 linhas. Os controles removem a mesma
    # quantidade, mas de posições diferentes: é isso que separa "recência"
    # de "volume de dados".
    n_removed = horizon - 1
    if n_removed < 1:
        # Em h=1 os dois protocolos coincidem: não há linha a remover, os
        # controles viram no-op e o bloco reprovaria sem explicar o motivo.
        print("\n    h=1: embargo e protocolo antigo são idênticos por definição "
              "(a origem é o mês anterior).")
        print("    Não há vazamento a medir — bloco não se aplica.")
        verdict("Bloco 2 — embargo", True, "(não se aplica em h=1)")
        return

    def ctrl_random(cutoff):
        mask = train_mask_for(yh.index, cutoff, horizon, embargo=False).copy()
        idx = np.where(mask)[0]
        mask[rng.choice(idx[:-n_removed], size=n_removed, replace=False)] = False
        return mask

    def ctrl_oldest(cutoff):
        mask = train_mask_for(yh.index, cutoff, horizon, embargo=False).copy()
        mask[np.where(mask)[0][:n_removed]] = False
        return mask

    m_rand = run(ctrl_random, f"C) controle: -{n_removed} linhas aleatórias do miolo")
    m_old = run(ctrl_oldest, f"D) controle: -{n_removed} linhas mais antigas")

    inflation = m_clean["MAPE"] - m_leaky["MAPE"]
    control_gap = max(m_rand["MAPE"], m_old["MAPE"]) - m_leaky["MAPE"]
    print(f"\n    Inflação atribuída ao vazamento: {inflation:+.3f} p.p.")
    print(f"    Efeito de perder {n_removed} linhas quaisquer: {control_gap:+.3f} p.p.")
    print("    -> se o primeiro é muito maior, a causa é a recência, não o volume.")

    passed = inflation > 1.0 and control_gap < 0.5
    verdict("Bloco 2 — embargo", passed,
            f"(vazamento inflava {abs(inflation):.2f} p.p.)")


# ---------------------------------------------------------------------------
def audit_forecast_is_direct(y: pd.Series) -> None:
    """Bloco 3 — o forecast de 12 meses é mesmo direto, sem recursão."""
    header("BLOCO 3 — FORECAST NÃO RECURSIVO")
    horizon = max(BT.horizons)
    future = pd.date_range(y.index.max() + pd.DateOffset(months=1), periods=horizon, freq="MS")
    y_ext = y.reindex(y.index.append(future))

    all_future_nan = bool(y_ext.loc[future].isna().all())
    cal = make_calendar_features(y_ext.index, n_fourier=FEATS["n_fourier"])
    lags = make_lag_features(
        y_ext,
        lags=tuple(legal_lags_for_horizon(horizon, lags=FEATS["lags"])),
        roll_windows=FEATS["roll_windows"],
        min_lag=horizon,
    )
    Xf = pd.concat([cal, lags], axis=1).loc[future]
    no_nan = not bool(Xf.isna().any().any())

    lag_col = f"lag_{horizon}"
    lag12_ok = all(
        abs(float(y.loc[d - pd.DateOffset(months=horizon)]) - float(Xf.loc[d, lag_col])) < 1e-9
        for d in future
    )

    print(f"\n    y dos {horizon} meses futuros é todo NaN.............. {all_future_nan}")
    print(f"    nenhuma feature futura é NaN................... {no_nan}")
    print(f"    {lag_col} futuro == y observado {horizon} meses antes.... {lag12_ok}")
    print(f"    origem dos lags: {(future[0] - pd.DateOffset(months=horizon)):%b/%Y}"
          f" a {(future[-1] - pd.DateOffset(months=horizon)):%b/%Y} (observados)")
    print("    -> se as três linhas acima são True, nenhuma previsão alimenta outra.")

    verdict("Bloco 3 — forecast direto", all_future_nan and no_nan and lag12_ok)


# ---------------------------------------------------------------------------
def audit_mape_behaviour(y: pd.Series) -> None:
    """Bloco 4 — por que o MAPE subpesa o erro no pico."""
    header("BLOCO 4 — COMPORTAMENTO DO MAPE NESTA SÉRIE")
    df = pd.DataFrame({"y": y})
    df["month"] = df.index.month
    peak = df[df.month.isin([10, 11, 12])].y.mean()
    trough = df[df.month.isin([4, 5, 6])].y.mean()

    print(f"\n    faixa do índice: {y.min():.1f} a {y.max():.1f} -> longe de zero, MAPE estável")
    print(f"    média meses de pico:  {peak:.1f}")
    print(f"    média meses de vale:  {trough:.1f}   (razão {peak / trough:.2f})")
    print(f"    um erro de 5 pontos vale {5 / trough * 100:.2f}% no vale"
          f" e {5 / peak * 100:.2f}% no pico")
    print("    -> o mesmo erro absoluto 'pesa' menos no pico, que é onde o S&OP sofre.")
    print("    -> use MASE para decidir; MAPE para comunicar, sempre ao lado do naive.")

    verdict("Bloco 4 — MAPE", y.min() > 10, "(diagnóstico, não teste binário)")


# ---------------------------------------------------------------------------
def main() -> int:
    print("AUDITORIA DO PIPELINE")
    print(f"configuração: {CFG.source}")
    print(f"boosters externos instalados: {available_boosters()}")

    # A auditoria não escreve artefato nenhum: ela só lê a série da camada
    # refined e reproduz as verificações. Por isso não prepara diretórios.
    y = CFG.carregar_serie()
    print(f"série carregada: {len(y)} observações, "
          f"{y.index.min():%b/%Y} a {y.index.max():%b/%Y}")

    audit_feature_provenance(y)
    audit_embargo(y)
    audit_forecast_is_direct(y)
    audit_mape_behaviour(y)

    header("RESUMO")
    for name, passed in results.items():
        print(f"  {'PASSOU' if passed else 'FALHOU'}  {name}")
    failed = [n for n, p in results.items() if not p]
    print()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
