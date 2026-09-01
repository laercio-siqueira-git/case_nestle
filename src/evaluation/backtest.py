"""
Backtest walk-forward (janela expansível) para previsão direta.

Por que não usar validação cruzada k-fold
-----------------------------------------
K-fold embaralha as linhas. Em série temporal isso coloca dados de 2015 no
treino e de 2013 no teste — o modelo "vê o futuro". A métrica resultante é
otimista e não tem relação com o desempenho em produção.

O embargo: a correção mais importante deste módulo
--------------------------------------------------
Evitar k-fold não basta. Em previsão *direta* existe uma segunda fonte de
vazamento, mais sutil, e que a primeira versão deste código continha.

Para prever o mês ``t`` com horizonte ``h``, a previsão é feita na **origem**
``o = t - h``. Uma linha de treino ``(X_s, y_s)`` só está disponível em ``o``
se o seu alvo já foi observado, isto é, se ``s <= o``.

A versão anterior treinava com ``s < t``, o que incluía os alvos de ``o+1``
até ``t-1`` — ``h-1`` meses que ainda não existem no momento da previsão.

Não é detalhe. Medido nesta série com ``h=12``, o Gradient Boosting passava de
5,98% para 3,34% de MAPE só por causa desses 11 meses. Um experimento de
controle confirmou que a causa é a *recência* dos alvos e não o volume:
remover 11 linhas aleatórias do miolo mantém o MAPE em 3,36%, enquanto remover
as 11 recentes o leva a 5,98%.

A regra correta está em :func:`train_mask_for` e o comportamento está travado
por ``tests/test_pipeline.py::TestEmbargo``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone

from src.evaluation.metrics import evaluate
from src.features.build_features import make_supervised_frame

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "run_full_benchmark",
    "train_mask_for",
    "walk_forward_backtest",
]


@dataclass(frozen=True)
class BacktestConfig:
    """Parâmetros do protocolo de validação.

    Attributes
    ----------
    n_test : int
        Meses avaliados no final da série. 36 cobre três ciclos sazonais.
    horizons : tuple[int, ...]
        Horizontes avaliados separadamente. Avaliar só ``h=1`` é a armadilha
        clássica: o S&OP planeja 12 meses.
    min_train : int
        Tamanho mínimo da janela de treino, em meses.
    season : int
        Período sazonal, usado no MASE.
    embargo : bool
        Se ``True`` (padrão e única opção correta), o treino de cada ponto usa
        apenas alvos observáveis na origem da previsão. ``False`` reproduz o
        protocolo defeituoso e existe **somente** para o script de auditoria
        quantificar o efeito do vazamento. Nunca use ``False`` para reportar.
    """

    n_test: int = 36
    horizons: tuple[int, ...] = (1, 3, 12)
    min_train: int = 240
    season: int = 12
    embargo: bool = True


@dataclass
class BacktestResult:
    """Saída de um backtest: previsões ponto a ponto + métricas agregadas."""

    model_name: str
    horizon: int
    predictions: pd.DataFrame = field(repr=False)
    metrics: dict[str, float] = field(default_factory=dict)
    embargo: bool = True


def train_mask_for(
    index: pd.DatetimeIndex,
    cutoff: pd.Timestamp,
    horizon: int,
    embargo: bool = True,
) -> np.ndarray:
    """Máscara booleana das linhas de treino legítimas para prever ``cutoff``.

    Com ``embargo=True``, a origem da previsão é ``cutoff - horizon`` meses e
    só alvos até essa data estão observados.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Índice dos alvos (uma data por linha da matriz supervisionada).
    cutoff : pd.Timestamp
        Data do alvo que será previsto.
    horizon : int
        Horizonte de previsão, em meses.
    embargo : bool
        ``False`` reproduz o protocolo defeituoso. Só para auditoria.

    Returns
    -------
    np.ndarray
        Vetor booleano do mesmo tamanho de ``index``.

    Examples
    --------
    Prevendo jan/2015 com 12 meses de antecedência, a origem é jan/2014 —
    dez/2014 não pode entrar no treino:

    >>> idx = pd.date_range("2014-01-01", "2015-01-01", freq="MS")
    >>> m = train_mask_for(idx, pd.Timestamp("2015-01-01"), horizon=12)
    >>> idx[m].max()
    Timestamp('2014-01-01 00:00:00')
    """
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    if not embargo:
        return np.asarray(index < cutoff)
    origin = cutoff - pd.DateOffset(months=horizon)
    return np.asarray(index <= origin)


def walk_forward_backtest(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    config: BacktestConfig,
    horizon: int,
    model_name: str = "model",
) -> BacktestResult:
    """Executa o walk-forward de um modelo em um horizonte.

    Para cada mês da janela de teste: monta a máscara respeitando o embargo,
    clona o estimador (nenhum estado sobrevive entre iterações), treina, prevê
    aquele mês e guarda o par previsto/realizado.

    Parameters
    ----------
    model : BaseEstimator
        Estimador **não treinado**. É clonado a cada iteração.
    X, y : pd.DataFrame, pd.Series
        Matriz supervisionada já alinhada ao horizonte.
    config : BacktestConfig
    horizon : int
        Governa o embargo — não é apenas um rótulo.
    model_name : str

    Returns
    -------
    BacktestResult
    """
    if len(X) <= config.min_train:
        raise ValueError(
            f"Amostra insuficiente: {len(X)} linhas para min_train={config.min_train}."
        )

    test_index = y.index[-config.n_test:]
    records = []

    for cutoff in test_index:
        mask = train_mask_for(y.index, cutoff, horizon, embargo=config.embargo)
        if mask.sum() < config.min_train:
            continue

        est = clone(model)
        est.fit(X.loc[mask], y.loc[mask])
        pred = float(np.ravel(est.predict(X.loc[[cutoff]]))[0])
        records.append(
            {
                "date": cutoff,
                "y_true": float(y.loc[cutoff]),
                "y_pred": pred,
                "n_train": int(mask.sum()),
            }
        )

    if not records:
        raise ValueError(
            f"Nenhum ponto avaliado para h={horizon}. min_train="
            f"{config.min_train} é alto demais para {len(X)} linhas."
        )

    preds = pd.DataFrame(records).set_index("date")
    y_train_full = y[y.index < test_index[0]].to_numpy()
    metrics = evaluate(
        preds["y_true"].to_numpy(),
        preds["y_pred"].to_numpy(),
        y_train=y_train_full,
        season=config.season,
    )
    metrics["vies"] = float((preds["y_pred"] - preds["y_true"]).mean())
    return BacktestResult(model_name, horizon, preds, metrics, config.embargo)


def run_full_benchmark(
    y: pd.Series,
    registry: dict[str, BaseEstimator],
    config: BacktestConfig,
    use_exogenous: bool = True,
    feature_params: dict | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, int], BacktestResult]]:
    """Roda todos os modelos em todos os horizontes.

    A matriz ``X`` é reconstruída por horizonte, porque as defasagens legais
    mudam: o modelo de ``h=12`` não tem acesso à coluna ``lag_1``, e essa
    restrição é estrutural, não uma convenção.

    Parameters
    ----------
    use_exogenous : bool, default True
        Chave do teste de ablação. Fica separada de ``feature_params`` de
        propósito, para que ligar ou desligar as exógenas seja sempre
        visível no ponto de chamada.
    feature_params : dict, optional
        Demais argumentos de :func:`make_supervised_frame` — ``n_fourier``,
        ``lags``, ``roll_windows``. Vêm de ``config/config.yaml`` via
        ``PipelineConfig.feature_params``.
    """
    rows, results = [], {}
    feature_params = dict(feature_params or {})
    feature_params.pop("use_exogenous", None)  # governado pelo argumento explícito

    for horizon in config.horizons:
        X, y_h = make_supervised_frame(
            y, horizon=horizon, use_exogenous=use_exogenous, **feature_params
        )
        for name, model in registry.items():
            try:
                res = walk_forward_backtest(
                    model, X, y_h, config, horizon=horizon, model_name=name
                )
            except ValueError as exc:
                rows.append(
                    {"modelo": name, "horizonte": horizon, "status": f"pulado: {exc}"}
                )
                continue
            results[(name, horizon)] = res
            rows.append(
                {
                    "modelo": name,
                    "horizonte": horizon,
                    "status": "ok",
                    "embargo": config.embargo,
                    **res.metrics,
                }
            )

    return pd.DataFrame(rows), results
