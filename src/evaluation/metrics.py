"""
Métricas de erro para previsão.

Usamos quatro métricas porque cada uma responde a uma pergunta diferente, e
apresentar só uma é a forma mais fácil de contar uma meia-verdade.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

__all__ = [
    "benjamini_hochberg",
    "diebold_mariano",
    "evaluate",
    "mae",
    "mape",
    "mase",
    "rmse",
]


def benjamini_hochberg(p_values) -> np.ndarray:
    """q-valores de Benjamini-Hochberg para um conjunto de testes.

    Por que isso é necessário
    -------------------------
    Comparar 8 modelos em 3 horizontes são 21 testes. A 5% cada, esperar-se-ia
    ~1 "descoberta" só por acaso. Reportar o menor p-valor de um conjunto grande
    como se fosse um teste isolado é a mesma classe de erro que escolher o
    campeão na janela de teste: a conclusão fica otimista por seleção.

    Bonferroni resolve dividindo o limiar pelo número de testes, mas é
    conservador a ponto de descartar achados reais. Benjamini-Hochberg controla
    a **proporção esperada de falsos positivos entre os rejeitados** (FDR) em
    vez de eliminá-los, o que é a pergunta certa aqui: não queremos zero erro,
    queremos saber quanto do que sobrou é confiável.

    Leitura: ``q < 0,05`` significa "entre os testes rejeitados a este limiar,
    espera-se no máximo 5% de falsos positivos".

    Parameters
    ----------
    p_values : array-like
        p-valores. ``NaN`` é preservado e ignorado no cálculo — serve para o
        baseline, que não é testado contra si mesmo.

    Returns
    -------
    np.ndarray
        q-valores, na mesma ordem da entrada.
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    validos = np.flatnonzero(~np.isnan(p))
    if validos.size == 0:
        return q

    pv = p[validos]
    m = pv.size
    ordem = np.argsort(pv)
    ranks = np.arange(1, m + 1)
    # Sobe da maior p para a menor tomando o mínimo acumulado: garante que o
    # q-valor seja monótono, isto é, que um p menor nunca receba q maior.
    ajustado = np.minimum.accumulate((pv[ordem] * m / ranks)[::-1])[::-1]
    q_ordenado = np.empty(m)
    q_ordenado[ordem] = np.minimum(ajustado, 1.0)
    q[validos] = q_ordenado
    return q


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro Absoluto Médio — na unidade do índice.

    Interpretação de negócio: "erramos, em média, X pontos de índice".
    Trata todos os erros linearmente; não pune outliers de forma especial.
    """
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Raiz do Erro Quadrático Médio.

    Penaliza erros grandes de forma quadrática. Se RMSE >> MAE, existem
    poucos meses com erro muito grande puxando a média — sinal de que o
    modelo falha em eventos específicos, não de forma difusa.
    """
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e**2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro Percentual Absoluto Médio (%).

    É a métrica que o público não técnico entende sem tradução. Cuidado
    conhecido: é instável quando ``y_true`` se aproxima de zero — não é o
    caso aqui, pois o índice opera na faixa 50--140.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season: int = 12
) -> float:
    """Erro Absoluto Escalado Médio.

    Divide o MAE do modelo pelo MAE de um naive sazonal calculado *no treino*.
    Leitura direta:

    - ``MASE < 1`` -> o modelo é melhor que repetir o ano passado;
    - ``MASE = 1`` -> empata com repetir o ano passado;
    - ``MASE > 1`` -> repetir o ano passado seria melhor.

    É a métrica mais honesta para séries sazonais, porque incorpora o
    baseline na própria definição. Um MAPE de 3% parece ótimo até você
    descobrir que o naive sazonal faz 2,8%.
    """
    y_train = np.asarray(y_train, dtype=float)
    denom = np.mean(np.abs(y_train[season:] - y_train[:-season]))
    if denom == 0:
        return float("nan")
    return mae(y_true, y_pred) / denom


def diebold_mariano(
    y_true, pred_a, pred_b, horizon: int, power: int = 1
) -> tuple[float, float]:
    """Testa se dois modelos têm acurácia *estatisticamente* diferente.

    Por que isso é necessário
    -------------------------
    Um MAPE menor não é evidência de um modelo melhor. Com 36 pontos de
    teste, uma diferença de meio ponto percentual cabe folgadamente dentro
    do ruído amostral. Reportar "o modelo A venceu" sem testar é a mesma
    classe de erro que o vazamento corrigido em :mod:`src.evaluation.backtest`:
    uma afirmação que a evidência não sustenta.

    Como funciona
    -------------
    Sobre o diferencial de perda ``d_t = |e_A|^p - |e_B|^p``:

    - ``H0``: os dois modelos têm a mesma acurácia (``E[d] = 0``);
    - ``d < 0`` indica A melhor que B.

    Duas correções obrigatórias em previsão multi-horizonte:

    1. **Variância HAC.** Previsões de ``h`` passos feitas em origens
       consecutivas compartilham informação, então ``d_t`` é autocorrelacionado
       até a defasagem ``h-1``. Usar a variância simples subestimaria o erro
       padrão e inflaria a significância. Somamos as autocovariâncias até
       ``h-1`` (Newey-West com truncamento em ``h-1``).
       As autocovariâncias entram com **peso de Bartlett** ``1 - k/h``. O peso
       não é cosmético: somá-las sem peso permite que a variância estimada
       saia negativa quando o diferencial tem autocorrelação negativa forte,
       e o teste devolve ``nan`` justamente nos casos em que os dois modelos
       mais divergem. Bartlett garante variância não negativa.

    2. **Correção de amostra pequena (Harvey-Leybourne-Newbold, 1997).** Com
       poucas observações o DM original rejeita demais. Aplicamos o fator de
       correção e comparamos contra uma ``t`` com ``n-1`` graus de liberdade,
       em vez da normal.

    Parameters
    ----------
    y_true : array-like
        Valores realizados.
    pred_a, pred_b : array-like
        Previsões dos dois modelos, **nas mesmas datas** de ``y_true``.
    horizon : int
        Horizonte da previsão, em meses. Governa o truncamento HAC.
    power : int, default 1
        1 compara erro absoluto (par do MAE/MAPE); 2 compara erro quadrático.

    Returns
    -------
    (stat, p_value) : tuple[float, float]
        ``stat`` negativo = ``pred_a`` melhor. ``p_value`` bilateral.
        ``(nan, nan)`` quando a variância estimada não é positiva — acontece
        com pouquíssimos pontos e deve ser reportado como "inconclusivo",
        nunca como empate.

    Notes
    -----
    Não rejeitar ``H0`` **não** prova que os modelos são equivalentes: com
    ``n`` pequeno o teste tem pouco poder. A leitura honesta é "não há
    evidência de diferença", e não "são iguais".
    """
    if horizon < 1:
        raise ValueError("horizon deve ser >= 1")
    y_true = np.asarray(y_true, dtype=float)
    pred_a = np.asarray(pred_a, dtype=float)
    pred_b = np.asarray(pred_b, dtype=float)
    if not (len(pred_a) == len(pred_b) == len(y_true)):
        raise ValueError(
            f"y_true, pred_a e pred_b devem ter o mesmo tamanho; recebi "
            f"{len(y_true)}, {len(pred_a)} e {len(pred_b)}."
        )

    d = np.abs(y_true - pred_a) ** power - np.abs(y_true - pred_b) ** power
    n = len(d)
    if n < 3:
        return float("nan"), float("nan")

    d_bar = float(d.mean())
    gamma_0 = float(np.mean((d - d_bar) ** 2))
    # Previsões idênticas: o diferencial é identicamente nulo. A variância é
    # zero e a razão seria 0/0, mas a resposta correta não é "inconclusivo" —
    # é "nenhuma diferença", com p=1.
    if gamma_0 == 0.0 and d_bar == 0.0:
        return 0.0, 1.0
    soma_hac = 0.0
    for k in range(1, min(horizon, n)):
        gamma_k = float(np.mean((d[k:] - d_bar) * (d[:-k] - d_bar)))
        soma_hac += (1 - k / horizon) * gamma_k      # peso de Bartlett
    var_d = (gamma_0 + 2 * soma_hac) / n
    if var_d <= 0:
        return float("nan"), float("nan")

    stat = d_bar / np.sqrt(var_d)
    correcao = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    if correcao <= 0:          # h grande demais para n: correção indefinida
        return float("nan"), float("nan")
    stat *= correcao
    p_value = 2 * (1 - stats.t.cdf(abs(stat), df=n - 1))
    return float(stat), float(p_value)


def evaluate(
    y_true, y_pred, y_train=None, season: int = 12
) -> dict[str, float]:
    """Calcula o conjunto de métricas de uma vez."""
    res = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
    if y_train is not None:
        res["MASE"] = mase(y_true, y_pred, y_train, season=season)
    return res
