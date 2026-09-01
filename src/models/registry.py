"""
Catálogo de modelos avaliados, do trivial ao não linear.

Regra de ouro adotada: **nenhum modelo entra em produção sem bater o
baseline mais burro que resolve o problema**. Em séries fortemente sazonais
esse baseline é o naive sazonal, e ele é surpreendentemente forte. Ignorá-lo
é o erro mais comum em projetos de forecasting.

Todos os estimadores expõem a interface `fit`/`predict` do scikit-learn, o
que permite tratá-los de forma intercambiável no backtest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

__all__ = [
    "SeasonalNaive",
    "SeasonalNaiveDrift",
    "available_boosters",
    "build_model_registry",
]

# ---------------------------------------------------------------------------
# Boosters opcionais
# ---------------------------------------------------------------------------
# XGBoost e LightGBM são importados de forma tolerante: o pipeline roda sem
# eles. A razão é que nenhum dos dois é necessário para o resultado — em ~500
# linhas de treino, as vantagens deles (velocidade em volume, GPU, tratamento
# nativo de ausentes) não mordem. Eles entram no benchmark para que a
# afirmação "trocar de biblioteca não resolve" seja verificada, e não assumida.
try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    LGBMRegressor = None
    HAS_LIGHTGBM = False


def available_boosters() -> dict[str, bool]:
    """Quais boosters externos estão instalados neste ambiente."""
    return {"xgboost": HAS_XGBOOST, "lightgbm": HAS_LIGHTGBM}


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Repete o valor observado 12 meses antes.

    Implementado como estimador sklearn para viver no mesmo backtest dos
    demais — assim a comparação é exatamente sobre os mesmos pontos, nas
    mesmas datas, sem código paralelo que possa divergir.

    A previsão é simplesmente a coluna ``lag_{season}`` da matriz de features.
    Não há parâmetro aprendido: ``fit`` só registra o nome da coluna.

    Parameters
    ----------
    season : int, default 12
        Período sazonal em meses.
    """

    def __init__(self, season: int = 12):
        self.season = season

    def fit(self, X: pd.DataFrame, y=None):
        col = f"lag_{self.season}"
        if col not in X.columns:
            raise ValueError(
                f"SeasonalNaive precisa da coluna '{col}'. "
                f"Para horizonte > {self.season} ela não é gerada."
            )
        self.lag_col_ = col
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.lag_col_].to_numpy()


class SeasonalNaiveDrift(BaseEstimator, RegressorMixin):
    """Naive sazonal corrigido pela tendência recente (*drift*).

    Melhoria mínima sobre o naive puro: soma ao valor de 12 meses atrás a
    variação ano-contra-ano observada, amortecida por ``damping``.

    ``previsao = lag_12 + damping * (lag_12 - lag_24)``

    Serve para responder a uma objeção natural do time de negócio: "repetir
    o ano passado ignora que a fábrica cresceu". Com um parâmetro só, o
    baseline passa a acompanhar tendência — e frequentemente já resolve boa
    parte do problema.

    Parameters
    ----------
    season : int, default 12
    damping : float, default 0.5
        0 reduz ao naive puro; 1 extrapola a variação anual inteira. Valores
        intermediários evitam amplificar ruído de um único ano atípico.
    """

    def __init__(self, season: int = 12, damping: float = 0.5):
        self.season = season
        self.damping = damping

    def fit(self, X: pd.DataFrame, y=None):
        self.lag_col_ = f"lag_{self.season}"
        self.drift_col_ = "yoy_diff_12"
        for c in (self.lag_col_, self.drift_col_):
            if c not in X.columns:
                raise ValueError(f"Coluna obrigatória ausente: '{c}'")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = X[self.lag_col_].to_numpy()
        drift = X[self.drift_col_].to_numpy()
        return base + self.damping * drift


def build_model_registry(
    random_state: int = 42, include_optional: bool = True
) -> dict[str, BaseEstimator]:
    """Devolve o dicionário nome -> estimador avaliado no backtest.

    Racional de cada entrada:

    ``seasonal_naive``
        Piso de referência. Custo zero, zero manutenção, zero risco.

    ``seasonal_naive_drift``
        Piso corrigido por tendência. Ainda explicável em uma frase.

    ``ridge_fourier``
        Regressão linear regularizada sobre calendário + Fourier + exógenas.
        Totalmente interpretável (coeficientes lidos como "efeito em pontos
        de índice"). A regularização L2 controla a colinearidade natural
        entre termos de Fourier e dummies de estação. Padronização é
        obrigatória aqui porque as escalas variam de 0/1 a ~140.

    ``random_forest``
        Não linear, robusto a outliers, praticamente sem tuning. Média de
        árvores independentes — reduz variância, mas não extrapola tendência
        para fora da faixa vista no treino (limitação relevante em série com
        crescimento estrutural).

    ``gradient_boosting``
        Árvores sequenciais, cada uma corrigindo o resíduo da anterior.
        Costuma ser o mais preciso da lista em dados tabulares; em troca é
        mais sensível a hiperparâmetros e também não extrapola.

    Notas de configuração
    ---------------------
    ``hist_gradient_boosting``
        Booster por histograma — o mesmo algoritmo do LightGBM e do modo
        ``hist`` do XGBoost, mas nativo do scikit-learn. Serve de controle:
        se ele reproduzir o resultado do XGBoost, fica demonstrado que a
        biblioteca não é a variável relevante.

    ``xgboost`` / ``lightgbm``
        Só entram se instalados. Configurados de forma equivalente aos demais
        (mesma profundidade, mesma taxa de aprendizado, mesma semente) para
        que a comparação isole a implementação, e não os hiperparâmetros.

    Notas de configuração
    ---------------------
    ``max_depth`` baixo (3) no boosting é deliberado: com ~500 linhas de
    treino, árvores profundas decoram o histórico. ``subsample=0.8``
    adiciona estocasticidade e funciona como regularização.

    Parameters
    ----------
    random_state : int
        Semente. Fixa por requisito de auditoria, não por preferência.
    include_optional : bool
        Se ``False``, omite XGBoost e LightGBM mesmo se instalados. Útil para
        reproduzir exatamente o benchmark do relatório.
    """
    registry: dict[str, BaseEstimator] = {
        "seasonal_naive": SeasonalNaive(season=12),
        "seasonal_naive_drift": SeasonalNaiveDrift(season=12, damping=0.5),
        "ridge_fourier": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=None)),
            ]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            random_state=random_state,
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=3,
            l2_regularization=1.0,
            random_state=random_state,
        ),
    }

    if include_optional and HAS_XGBOOST:
        registry["xgboost"] = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
            verbosity=0,
        )

    if include_optional and HAS_LIGHTGBM:
        registry["lightgbm"] = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=3,
            num_leaves=8,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=10,
            n_jobs=-1,
            random_state=random_state,
            verbose=-1,
        )

    return registry
