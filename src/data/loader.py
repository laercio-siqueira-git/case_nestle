"""
Ingestão e validação da série IPG3113N.

Princípio: *falhar cedo e alto*. Todo dado que entra no pipeline passa por um
contrato explícito. Se o contrato quebra, o pipeline para com uma mensagem
que diz o que quebrou — em vez de propagar silenciosamente um NaN até o
forecast e produzir um número errado que ninguém questiona.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

__all__ = ["SeriesContract", "load_candy_production", "load_refined_series"]


@dataclass(frozen=True)
class SeriesContract:
    """Contrato esperado da série de entrada.

    Attributes
    ----------
    date_col, value_col : str
        Nomes das colunas no arquivo bruto.
    freq : str
        Frequência pandas esperada ("MS" = início de mês).
    min_value, max_value : float
        Faixa plausível do índice. Serve de *sanity check*: um índice de
        produção com base 100 fora de [10, 500] indica erro de leitura,
        troca de unidade ou arquivo corrompido.
    min_observations : int
        Mínimo de pontos para que o backtest sazonal faça sentido.
    """

    date_col: str = "observation_date"
    value_col: str = "IPG3113N"
    freq: str = "MS"
    min_value: float = 10.0
    max_value: float = 500.0
    min_observations: int = 120


def load_candy_production(
    path: str | Path, contract: SeriesContract | None = None
) -> pd.Series:
    """Lê o CSV do FRED e devolve uma Series mensal validada.

    Validações aplicadas, nesta ordem:

    1. **Colunas presentes** — protege contra mudança de schema na origem.
    2. **Sem duplicatas de data** — duplicata silenciosa distorce médias móveis.
    3. **Grade temporal completa** — reindexa para a grade mensal teórica e
       verifica que não sobrou buraco. Um mês faltante desalinha *todos* os
       lags sazonais a jusante, que é uma das falhas mais difíceis de
       diagnosticar depois.
    4. **Faixa de valores plausível**.
    5. **Volume mínimo de histórico**.

    Parameters
    ----------
    path : str or Path
        Caminho do CSV bruto.
    contract : SeriesContract, optional
        Contrato a aplicar. Usa o padrão do FRED se omitido.

    Returns
    -------
    pd.Series
        Série indexada por data, frequência "MS", nome ``production_index``.

    Raises
    ------
    ValueError
        Se qualquer cláusula do contrato for violada.
    """
    contract = contract or SeriesContract()
    df = pd.read_csv(path)

    missing = {contract.date_col, contract.value_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no arquivo {path}: {sorted(missing)}. "
            f"Encontradas: {sorted(df.columns)}"
        )

    df[contract.date_col] = pd.to_datetime(df[contract.date_col])
    if df[contract.date_col].duplicated().any():
        dups = df.loc[df[contract.date_col].duplicated(), contract.date_col]
        raise ValueError(f"Datas duplicadas na origem: {dups.tolist()[:5]}")

    s = (
        df.set_index(contract.date_col)[contract.value_col]
        .sort_index()
        .astype("float64")
        .rename("production_index")
    )

    full_grid = pd.date_range(s.index.min(), s.index.max(), freq=contract.freq)
    reindexed = s.reindex(full_grid)
    if reindexed.isna().any():
        gaps = reindexed.index[reindexed.isna()]
        raise ValueError(
            f"{len(gaps)} mês(es) ausente(s) na grade temporal. "
            f"Primeiros: {[str(g.date()) for g in gaps[:5]]}"
        )
    s = reindexed
    s.index.name = "date"

    out_of_range = s[(s < contract.min_value) | (s > contract.max_value)]
    if not out_of_range.empty:
        raise ValueError(
            f"{len(out_of_range)} valor(es) fora da faixa plausível "
            f"[{contract.min_value}, {contract.max_value}]: "
            f"{out_of_range.head().to_dict()}"
        )

    if len(s) < contract.min_observations:
        raise ValueError(
            f"Histórico insuficiente: {len(s)} observações, "
            f"mínimo {contract.min_observations}."
        )

    return s


def load_refined_series(
    refined_path: str | Path,
    raw_path: str | Path | None = None,
    value_col: str = "producao",
    date_col: str = "data",
) -> pd.Series:
    """Lê a série da camada refined, já validada na construção dela.

    Por que existe uma segunda função de leitura
    -------------------------------------------
    ``load_candy_production`` aplica o contrato a cada leitura. Isso é correto
    na fronteira entre ``raw`` e ``refined``, e desperdício depois dela: o dado
    já provou estar íntegro quando virou ``refined``. Validar uma vez, no ponto
    de entrada, é o que dá sentido à camada.

    O risco que isso cria, e a defesa
    ---------------------------------
    Se ``refined`` for mais antiga que ``raw``, todo o pipeline roda sobre dado
    velho **sem avisar** — a mesma classe de falha silenciosa que o projeto
    combate em outros pontos. Por isso a função compara as datas de modificação
    e **falha alto** quando a origem é mais recente, em vez de devolver o dado
    desatualizado.

    Parameters
    ----------
    refined_path : str or Path
        Parquet da camada refined.
    raw_path : str or Path, optional
        Origem, para a verificação de atualidade. Sem ela a verificação é
        pulada — útil em teste, arriscado em produção.
    value_col, date_col : str
        Colunas a extrair.

    Returns
    -------
    pd.Series
        Série mensal indexada por data, nome ``production_index`` — mesma
        forma que ``load_candy_production`` devolve, para que quem consome não
        precise saber de qual camada veio.
    """
    refined_path = Path(refined_path)
    if not refined_path.exists():
        raise FileNotFoundError(
            f"Camada refined ausente: {refined_path}. "
            f"Rode `python scripts/build_layers.py --stage refined` antes."
        )

    if raw_path is not None:
        raw_path = Path(raw_path)
        if raw_path.exists() and raw_path.stat().st_mtime > refined_path.stat().st_mtime:
            raise ValueError(
                f"Camada refined desatualizada: {raw_path.name} é mais recente "
                f"que {refined_path.name}. Reconstrua com "
                f"`python scripts/build_layers.py --stage refined`."
            )

    df = pd.read_parquet(refined_path, columns=[date_col, value_col])
    s = (
        df.set_index(date_col)[value_col]
        .astype("float64")
        .rename("production_index")
        .sort_index()
    )
    s.index.name = "date"
    return s
