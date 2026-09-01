"""
Engenharia de atributos para previsão *direta* multi-horizonte.

O ponto central deste módulo
----------------------------
Existem duas formas de prever 12 meses à frente com um modelo tabular:

**Recursiva** — treina um modelo de 1 passo e realimenta a própria previsão
como se fosse observação real, 12 vezes. Simples, mas o erro do passo 1
contamina o passo 2, que contamina o passo 3, e assim por diante. A
degradação é composta e difícil de quantificar.

**Direta** — treina um modelo *por horizonte*. O modelo do horizonte 12
aprende a mapear "o que eu sei hoje" em "o que vai acontecer em 12 meses",
sem nunca consumir uma previsão como entrada. Custa 12 modelos em vez de 1,
mas cada previsão é honesta e o erro de cada horizonte é medido diretamente.

Escolhemos a estratégia **direta**. A consequência prática está na função
`make_supervised_frame`: para prever o alvo em ``t + h``, só podemos usar
defasagens ``lag_k`` com ``k >= h``. Se usássemos ``lag_1`` para prever 12
meses à frente, estaríamos assumindo conhecer o valor de 11 meses no futuro
— vazamento clássico, que infla a métrica de validação e explode em produção.

A função impõe essa regra em código, não em disciplina do analista.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.calendar_features import build_calendar_frame

__all__ = [
    "EXOGENOUS_COLS",
    "LAGS",
    "ROLL_WINDOWS",
    "legal_lags_for_horizon",
    "make_calendar_features",
    "make_lag_features",
    "make_supervised_frame",
]

#: Defasagens candidatas, em meses. 12 e 24 capturam o eco sazonal;
#: 1--6 capturam o nível/momento recente (utilizáveis só em horizontes curtos).
LAGS: tuple[int, ...] = (1, 2, 3, 6, 12, 13, 24)

#: Janelas de estatística móvel (média e desvio-padrão).
ROLL_WINDOWS: tuple[int, ...] = (3, 12)

#: As oito colunas que a flag ``use_exogenous`` liga e desliga.
#:
#: A fronteira é **informação sobre o mundo** contra **codificação do
#: calendário**, e não "tudo que veio do módulo de calendário".
#:
#: Entram aqui: quantos dias úteis o mês teve, em que mês caiu a Páscoa, quantos
#: dias separam Thanksgiving do Natal. São fatos sobre o calendário civil de
#: cada ano — variam de um ano para o outro e carregam informação externa.
#:
#: Não entram: ``quarter``, ``time_trend``, os termos de Fourier e
#: ``is_holiday_peak_season`` (que é ``mês ∈ {9,10,11,12}``). Essas são funções
#: determinísticas da posição no ano — a sazonalidade escrita de outro jeito.
#: Removê-las não seria "desligar exógenas", seria tirar do modelo a noção de
#: que dezembro é diferente de abril, o que nenhum resultado deste projeto
#: sustenta.
#:
#: Vive aqui, e não numa lista solta em cada script, porque ela é usada em dois
#: lugares com propósitos diferentes — montar a matriz e somar a importância
#: atribuída ao grupo. Duas cópias que precisam concordar acabam divergindo, e a
#: divergência apareceria como um número de relatório levemente errado, que é o
#: tipo de erro que ninguém percebe.
EXOGENOUS_COLS: tuple[str, ...] = (
    "n_days_month",
    "n_business_days",
    "n_saturdays",
    "easter_month",
    "easter_lead_1",
    "easter_lead_2",
    "easter_day_of_year",
    "thanksgiving_to_xmas_days",
)


def make_calendar_features(index: pd.DatetimeIndex, n_fourier: int = 2) -> pd.DataFrame:
    """Bloco determinístico: calendário civil + sazonalidade harmônica.

    Combina duas famílias complementares:

    - **Exógenas de calendário** (`build_calendar_frame`): Páscoa móvel, dias
      úteis, janela Thanksgiving--Natal. Explicam por que dois meses de março
      diferentes não são equivalentes.
    - **Termos de Fourier**: pares seno/cosseno de período anual. Codificam a
      posição no ciclo de forma *contínua e cíclica* — dezembro e janeiro
      ficam próximos no espaço de features, o que não acontece com a
      codificação ingênua ``mes = 12`` vs ``mes = 1``. Com ``n_fourier=2``
      temos 4 colunas que descrevem uma sazonalidade suave, contra 11 dummies
      de mês; menos parâmetros para o mesmo sinal.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Grade mensal alvo.
    n_fourier : int, default 2
        Número de harmônicos. Mais harmônicos = curva sazonal mais "pontuda";
        acima de 3 tende a sobreajustar em séries mensais.

    Returns
    -------
    pd.DataFrame
        Features indexadas por data. Todas conhecidas a priori.
    """
    cal = build_calendar_frame(index).copy()

    month = index.month.to_numpy()
    for k in range(1, n_fourier + 1):
        cal[f"fourier_sin_{k}"] = np.sin(2 * np.pi * k * month / 12)
        cal[f"fourier_cos_{k}"] = np.cos(2 * np.pi * k * month / 12)

    # Tendência determinística: índice temporal em anos desde o início.
    cal["time_trend"] = (index.year - index.year.min()) + (index.month - 1) / 12
    cal["quarter"] = index.quarter
    return cal


def make_lag_features(
    y: pd.Series,
    lags: tuple[int, ...] = LAGS,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
    min_lag: int = 1,
) -> pd.DataFrame:
    """Bloco autoregressivo: defasagens e estatísticas móveis do próprio alvo.

    Convenção crítica: ``lag_k`` na linha de data ``t`` contém ``y[t - k]``.

    ``min_lag`` é a defasagem mais recente **legalmente observável** — igual
    ao horizonte de previsão. Toda estatística móvel usa uma janela que
    termina em ``t - min_lag``, e não em ``t - 1``.

    Essa distinção é a fonte de vazamento mais silenciosa de todo o pipeline.
    Uma média móvel de 3 meses deslocada apenas um período contém
    ``y[t-1], y[t-2], y[t-3]``. Num modelo de horizonte 12, esses três valores
    ainda não existem no momento da previsão — a média móvel entregaria ao
    modelo o nível quase atual da série. O resultado é uma métrica de
    validação excelente e um desempenho medíocre em produção.

    Returns
    -------
    pd.DataFrame
        Colunas ``lag_{k}``, ``roll_mean_{w}``, ``roll_std_{w}``,
        ``yoy_diff_12`` (variação contra o mesmo mês do ano anterior).
    """
    if min_lag < 1:
        raise ValueError("min_lag deve ser >= 1")

    out = pd.DataFrame(index=y.index)
    for k in lags:
        if k < min_lag:
            raise ValueError(
                f"lag_{k} é ilegal para min_lag={min_lag}: exigiria conhecer "
                f"o valor de {min_lag - k} mês(es) no futuro."
            )
        out[f"lag_{k}"] = y.shift(k)

    shifted = y.shift(min_lag)          # último ponto legalmente observável
    for w in roll_windows:
        out[f"roll_mean_{w}"] = shifted.rolling(w).mean()
        out[f"roll_std_{w}"] = shifted.rolling(w).std()

    # Variação ano-contra-ano: sempre uma diferença de 12 meses, mas ancorada
    # na defasagem mais recente que for legal. Para min_lag <= 12 a âncora é 12
    # e a coluna se chama `yoy_diff_12`; acima disso a âncora muda, e o nome
    # muda junto. Manter o sufixo 12 com a fórmula usando outra defasagem daria
    # uma coluna cujo nome mente — e nomes de feature acabam em gráfico de
    # importância, onde ninguém confere a fórmula.
    ancora = max(12, min_lag)
    out[f"yoy_diff_{ancora}"] = y.shift(ancora) - y.shift(ancora + 12)
    return out


def legal_lags_for_horizon(horizon: int, lags: tuple[int, ...] = LAGS) -> list[int]:
    """Defasagens utilizáveis para prever ``horizon`` meses à frente.

    Em previsão direta, ao prever ``y[t+h]`` a informação mais recente
    disponível é ``y[t]``, que em relação ao alvo é a defasagem ``h``.
    Portanto só ``lag_k`` com ``k >= h`` é legítimo.
    """
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    return [k for k in lags if k >= horizon]


def make_supervised_frame(
    y: pd.Series,
    horizon: int,
    n_fourier: int = 2,
    use_exogenous: bool = True,
    lags: tuple[int, ...] = LAGS,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
) -> tuple[pd.DataFrame, pd.Series]:
    """Monta a matriz (X, y) para previsão direta no horizonte informado.

    O alvo de cada linha é o valor observado naquela data; as features são
    todas construídas de modo a estarem disponíveis ``horizon`` meses antes.

    Parameters
    ----------
    y : pd.Series
        Série alvo, mensal, sem buracos.
    horizon : int
        Horizonte de previsão em meses. Governa quais defasagens são legais.
    n_fourier : int, default 2
        Harmônicos de Fourier.
    use_exogenous : bool, default True
        Se ``False``, remove as exógenas de calendário (Páscoa, dias úteis,
        Thanksgiving) e mantém só Fourier + tendência + autoregressivos.
        Existe para permitir o *teste de ablação* que quantifica o ganho
        atribuível às exógenas — sem esse controle, não é possível afirmar
        que elas ajudaram.
    lags : tuple[int, ...], default LAGS
        Defasagens candidatas. São filtradas por ``legal_lags_for_horizon``,
        então passar uma defasagem menor que ``horizon`` é inofensivo: ela
        simplesmente não entra na matriz.
    roll_windows : tuple[int, ...], default ROLL_WINDOWS
        Janelas das estatísticas móveis, todas encerradas em ``t - horizon``.

    Returns
    -------
    (X, y_aligned) : tuple[pd.DataFrame, pd.Series]
        Linhas com qualquer NaN são descartadas — o aquecimento inicial da
        série (primeiros 24 meses) não tem defasagens completas.
    """
    cal = make_calendar_features(y.index, n_fourier=n_fourier)
    if not use_exogenous:
        cal = cal.drop(columns=[c for c in EXOGENOUS_COLS if c in cal.columns])

    legal = legal_lags_for_horizon(horizon, lags=tuple(lags))
    lag_block = make_lag_features(
        y,
        lags=tuple(legal),
        roll_windows=tuple(roll_windows),
        min_lag=horizon,
    )

    X = pd.concat([cal, lag_block], axis=1)
    frame = X.join(y.rename("__target__")).dropna()
    return frame.drop(columns="__target__"), frame["__target__"]
