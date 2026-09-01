"""
Calendário civil e comercial dos EUA como fonte de dados exógena.

Por que este módulo existe
--------------------------
A série IPG3113N é produção de confeitaria nos EUA, *não ajustada
sazonalmente*. Boa parte da sua variação mensal é explicada por dois
fenômenos de calendário que o índice do mês, sozinho, NÃO captura:

1. **Páscoa móvel.** A Páscoa cai entre 22/mar e 25/abr. A produção de
   ovos e coelhos de chocolate acontece semanas antes da data. Logo, se a
   Páscoa cai em março, o pico de produção migra de março para fevereiro;
   se cai em abril, migra para março. Um modelo que só conhece "mês = 3"
   não distingue esses dois regimes e trata a diferença como ruído.

2. **Composição do mês.** Um mês com 23 dias úteis produz mais que um mês
   com 19 dias úteis, tudo o mais constante. Como a série é mensal e não
   ajustada, esse efeito entra direto no índice.

Decisão de engenharia
---------------------
Estas variáveis são *calculadas*, não baixadas. As regras são definidas em
lei (5 U.S.C. § 6103 para feriados federais) e por computus eclesiástico
(Páscoa). Isso significa:

- zero dependência de API externa em produção (sem risco de indisponibilidade);
- valores conhecidos com anos de antecedência — condição obrigatória para
  serem usados como regressores em previsão futura;
- reprodutibilidade total: o mesmo código gera o mesmo valor sempre.

Esse é o motivo de não usarmos a biblioteca `holidays`: para o escopo
necessário (feriados federais fixos + Páscoa), o código abaixo é curto,
auditável e elimina uma dependência de terceiros do caminho crítico.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import numpy as np
import pandas as pd

__all__ = [
    "NBER_RECESSIONS",
    "build_calendar_frame",
    "easter_sunday",
    "us_federal_holidays",
]


# ---------------------------------------------------------------------------
# 1. Páscoa
# ---------------------------------------------------------------------------
@lru_cache(maxsize=512)
def easter_sunday(year: int) -> dt.date:
    """Domingo de Páscoa (calendário gregoriano).

    Implementa o algoritmo *Anonymous Gregorian* (também chamado
    Meeus/Jones/Butcher). É aritmética pura sobre o ano: não há tabela,
    não há chamada de rede, e o resultado é exato para 1583--4099.

    A intuição do algoritmo: a Páscoa é o primeiro domingo após a primeira
    lua cheia eclesiástica que ocorre em ou depois de 21 de março. As
    variáveis abaixo reconstroem essa regra em passos inteiros.

    Parameters
    ----------
    year : int
        Ano gregoriano.

    Returns
    -------
    datetime.date
        Data do Domingo de Páscoa.

    Examples
    --------
    >>> easter_sunday(2024)
    datetime.date(2024, 3, 31)
    >>> easter_sunday(2025)
    datetime.date(2025, 4, 20)
    """
    a = year % 19            # posição no ciclo metônico de 19 anos
    b, c = divmod(year, 100)  # século e ano dentro do século
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30   # epacta: idade da lua em 22/mar
    i, k = divmod(c, 4)
    lam = (32 + 2 * e + 2 * i - h - k) % 7  # dias até o domingo seguinte
    m = (a + 11 * h + 22 * lam) // 451       # correção de casos-limite
    month, day = divmod(h + lam - 7 * m + 114, 31)
    return dt.date(year, month, day + 1)


# ---------------------------------------------------------------------------
# 2. Feriados federais dos EUA
# ---------------------------------------------------------------------------
def _nth_weekday(year: int, month: int, weekday: int, n: int) -> dt.date:
    """N-ésima ocorrência de um dia da semana no mês (weekday: 0=segunda)."""
    first = dt.date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + dt.timedelta(days=offset + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> dt.date:
    """Última ocorrência de um dia da semana no mês."""
    last_day = pd.Timestamp(year=year, month=month, day=1).days_in_month
    last = dt.date(year, month, last_day)
    return last - dt.timedelta(days=(last.weekday() - weekday) % 7)


@lru_cache(maxsize=256)
def us_federal_holidays(year: int) -> tuple[dt.date, ...]:
    """Feriados federais dos EUA observados no ano (regra de 5 U.S.C. § 6103).

    Inclui a regra de *observância*: feriado de data fixa que cai no sábado é
    observado na sexta anterior; no domingo, na segunda seguinte. Isso importa
    porque a contagem de dias úteis do mês depende do dia observado, não da
    data nominal.

    Juneteenth (19/jun) só passa a valer a partir de 2021 — a função respeita
    isso para não introduzir anacronismo no histórico de 1972 em diante.
    """
    fixed = [
        dt.date(year, 1, 1),    # Ano-Novo
        dt.date(year, 7, 4),    # Independência
        dt.date(year, 11, 11),  # Veterans Day
        dt.date(year, 12, 25),  # Natal
    ]
    if year >= 2021:
        fixed.append(dt.date(year, 6, 19))  # Juneteenth

    def observed(d: dt.date) -> dt.date:
        if d.weekday() == 5:                      # sábado -> sexta
            return d - dt.timedelta(days=1)
        if d.weekday() == 6:                      # domingo -> segunda
            return d + dt.timedelta(days=1)
        return d

    floating = [
        _nth_weekday(year, 1, 0, 3),    # MLK Day: 3ª segunda de janeiro
        _nth_weekday(year, 2, 0, 3),    # Presidents Day: 3ª segunda de fevereiro
        _last_weekday(year, 5, 0),      # Memorial Day: última segunda de maio
        _nth_weekday(year, 9, 0, 1),    # Labor Day: 1ª segunda de setembro
        _nth_weekday(year, 10, 0, 2),   # Columbus Day: 2ª segunda de outubro
        _nth_weekday(year, 11, 3, 4),   # Thanksgiving: 4ª quinta de novembro
    ]
    return tuple(sorted([observed(d) for d in fixed] + floating))


# ---------------------------------------------------------------------------
# 3. Recessões NBER (usadas apenas para diagnóstico, nunca como feature)
# ---------------------------------------------------------------------------
#: Datas oficiais de pico->vale do National Bureau of Economic Research.
#: NÃO entram como variável preditiva: o NBER declara uma recessão com
#: meses de atraso, então em produção esse rótulo não estaria disponível no
#: momento da previsão. Usamos só para anotar gráficos e interpretar erros.
NBER_RECESSIONS: tuple[tuple[str, str], ...] = (
    ("1973-11-01", "1975-03-01"),
    ("1980-01-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01"),
    ("1990-07-01", "1991-03-01"),
    ("2001-03-01", "2001-11-01"),
    ("2007-12-01", "2009-06-01"),
)


# ---------------------------------------------------------------------------
# 4. Montagem do frame mensal de calendário
# ---------------------------------------------------------------------------
def build_calendar_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Constrói o bloco de features de calendário para um índice mensal.

    Todas as colunas são conhecidas com antecedência arbitrária — nenhuma
    depende de dado observado. Essa é a propriedade que permite usá-las em
    forecast de 12 meses sem vazamento.

    Colunas geradas
    ---------------
    n_days_month : int
        Dias corridos no mês (28--31).
    n_business_days : int
        Dias úteis (seg--sex) descontando feriados federais observados.
    n_saturdays : int
        Sábados no mês — proxy de turno extra em pico sazonal.
    easter_month : int
        1 se o Domingo de Páscoa cai neste mês.
    easter_lead_1 : int
        1 se a Páscoa cai no mês *seguinte* (janela principal de produção).
    easter_lead_2 : int
        1 se a Páscoa cai daqui a dois meses.
    easter_day_of_year : int
        Dia do ano da Páscoa — mede *quão tarde* ela cai, o que desloca a
        produção continuamente em vez de em degraus.
    thanksgiving_to_xmas_days : int
        Dias entre Thanksgiving e o Natal (26--32). Janela curta comprime a
        temporada de vendas de fim de ano; só é diferente de zero em nov/dez.
    is_holiday_peak_season : int
        1 para setembro--dezembro (Halloween + Natal).
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("`index` deve ser um pd.DatetimeIndex mensal.")

    rows = []
    for ts in index:
        year, month = ts.year, ts.month
        n_days = ts.days_in_month
        month_start = np.datetime64(dt.date(year, month, 1))
        month_end = np.datetime64(dt.date(year, month, n_days) + dt.timedelta(days=1))

        hols = np.array(
            [np.datetime64(h) for h in us_federal_holidays(year)],
            dtype="datetime64[D]",
        )
        n_bus = int(
            np.busday_count(month_start, month_end, holidays=hols)
        )
        n_sat = int(np.busday_count(month_start, month_end, weekmask="Sat"))

        easter = easter_sunday(year)
        easter_ord = (easter.year - year) * 12 + easter.month  # mês da Páscoa
        months_until_easter = easter_ord - month

        thanksgiving = _nth_weekday(year, 11, 3, 4)
        tg_to_xmas = (dt.date(year, 12, 25) - thanksgiving).days

        rows.append(
            {
                "date": ts,
                "n_days_month": n_days,
                "n_business_days": n_bus,
                "n_saturdays": n_sat,
                "easter_month": int(months_until_easter == 0),
                "easter_lead_1": int(months_until_easter == 1),
                "easter_lead_2": int(months_until_easter == 2),
                "easter_day_of_year": easter.timetuple().tm_yday,
                "thanksgiving_to_xmas_days": tg_to_xmas if month in (11, 12) else 0,
                "is_holiday_peak_season": int(month in (9, 10, 11, 12)),
            }
        )

    return pd.DataFrame(rows).set_index("date")
