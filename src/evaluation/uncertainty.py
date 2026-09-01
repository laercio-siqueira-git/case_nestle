"""
Quanta confiança o tamanho da amostra permite ter.

O Diebold-Mariano em :mod:`src.evaluation.metrics` responde *"há evidência de
diferença?"*. Ele não responde as duas perguntas que vêm logo depois, e que
decidem se um resultado nulo significa algo:

1. **De que tamanho é a diferença, com que margem?** Um p-valor esconde isso.
   ``intervalo_bootstrap`` devolve o ganho observado e um intervalo em torno
   dele, na unidade do problema. Ver o zero dentro do intervalo é mais
   informativo — e mais convincente para quem decide — do que ler "p = 0,40".

2. **O teste enxergaria a diferença, se ela existisse?** Não rejeitar a
   hipótese nula com amostra pequena não é evidência de equivalência: pode ser
   cegueira do experimento. ``efeito_minimo_detectavel`` mede essa cegueira
   diretamente, impondo vantagens conhecidas e contando quantas vezes o teste
   as encontra.

Dependência temporal
--------------------
Previsões de ``h`` passos feitas em origens consecutivas se sobrepõem: elas
compartilham informação, e o diferencial de perda fica autocorrelacionado até
a defasagem ``h-1``. Reamostrar mês a mês trataria observações dependentes
como independentes, produziria intervalos estreitos demais e declararia
significância onde não há. Por isso tudo aqui usa **bootstrap de blocos**, com
blocos do tamanho do horizonte.

O preço disso é honesto e precisa ser reportado: com 36 meses e ``h=12``
sobram 3 blocos por reamostragem. O intervalo sai largo porque a informação
disponível é pouca, não porque o método é ruim.
"""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import diebold_mariano

__all__ = [
    "efeito_minimo_detectavel",
    "intervalo_bootstrap",
    "veredito_estabilidade",
]


def veredito_estabilidade(
    ganhos: dict[int, float], limiar_amplitude: float = 15.0
) -> dict:
    """Classifica um ganho medido em várias janelas de teste.

    Por que isso é necessário
    -------------------------
    Uma única janela de teste responde "quanto o modelo ganhou naqueles meses".
    Não responde "o modelo ganha". A diferença aparece quando se repete a
    medição em janelas diferentes: um ganho real se mantém; um ganho que era
    característica daquele período específico oscila, e às vezes **troca de
    sinal**.

    Trocar de sinal é o achado mais forte que esta função detecta. Um modelo
    cuja vantagem medida é ora positiva ora negativa conforme o período avaliado
    não é um modelo indeciso — é um modelo cujo ganho não é reproduzível, e isso
    basta para não colocá-lo em produção.

    Parameters
    ----------
    ganhos : dict[int, float]
        ``{tamanho_da_janela: ganho_percentual}``. Positivo = campeão melhor.
    limiar_amplitude : float, default 15.0
        Amplitude, em pontos percentuais, acima da qual o ganho é considerado
        instável mesmo sem trocar de sinal.

    Returns
    -------
    dict
        ``troca_sinal``, ``amplitude``, ``ganho_min``, ``ganho_max``,
        ``estavel`` e ``veredito`` — este último já redigido para ir a
        relatório ou título de figura.
    """
    if len(ganhos) < 2:
        raise ValueError(
            f"São necessárias ao menos 2 janelas para julgar estabilidade; "
            f"recebi {len(ganhos)}."
        )

    valores = list(ganhos.values())
    g_min, g_max = float(min(valores)), float(max(valores))
    amplitude = g_max - g_min
    troca_sinal = bool(g_min < 0 < g_max)

    if troca_sinal:
        veredito = (f"NÃO REPRODUZÍVEL — o ganho troca de sinal entre janelas "
                    f"({g_min:+.1f}% a {g_max:+.1f}%)")
        estavel = False
    elif amplitude > limiar_amplitude:
        veredito = (f"INSTÁVEL — o ganho varia {amplitude:.1f} p.p. entre "
                    f"janelas ({g_min:+.1f}% a {g_max:+.1f}%)")
        estavel = False
    else:
        veredito = (f"ESTÁVEL — o ganho se mantém entre {g_min:+.1f}% e "
                    f"{g_max:+.1f}% em todas as janelas")
        estavel = True

    return {
        "troca_sinal": troca_sinal,
        "amplitude": round(amplitude, 2),
        "ganho_min": round(g_min, 2),
        "ganho_max": round(g_max, 2),
        "estavel": estavel,
        "veredito": veredito,
        "n_janelas": len(ganhos),
    }


def _indices_em_blocos(
    n: int, comprimento: int, n_amostras: int, rng: np.random.Generator
) -> np.ndarray:
    """Índices de *moving block bootstrap*, no formato ``(n_amostras, n)``.

    Sorteia blocos de posições consecutivas, com reposição, e os concatena até
    completar ``n`` observações. Blocos consecutivos preservam a
    autocorrelação que a reamostragem ponto a ponto destruiria.
    """
    comprimento = max(1, min(comprimento, n))
    n_blocos = int(np.ceil(n / comprimento))
    inicios = rng.integers(0, n - comprimento + 1, size=(n_amostras, n_blocos))
    deslocamento = np.arange(comprimento)
    idx = (inicios[:, :, None] + deslocamento[None, None, :]).reshape(n_amostras, -1)
    return idx[:, :n]


def intervalo_bootstrap(
    y_true,
    pred_a,
    pred_b,
    horizon: int,
    n_amostras: int = 5000,
    alfa: float = 0.10,
    seed: int = 42,
) -> dict:
    """Intervalo de confiança para a vantagem de ``pred_a`` sobre ``pred_b``.

    Parameters
    ----------
    y_true, pred_a, pred_b : array-like
        Realizados e as duas previsões, nas mesmas datas.
    horizon : int
        Horizonte em meses. Define o comprimento do bloco.
    n_amostras : int, default 5000
        Reamostragens.
    alfa : float, default 0.10
        Nível do intervalo: 0,10 devolve um intervalo de 90%.
    seed : int
        Semente, por requisito de auditoria.

    Returns
    -------
    dict
        ``ganho_mae`` em pontos do índice (positivo = ``pred_a`` melhor),
        ``ganho_pct`` em % de redução do erro, os limites de ambos,
        ``inclui_zero`` e ``n_blocos`` — este último para que a largura do
        intervalo possa ser lida junto com a informação que a produziu.
    """
    y_true = np.asarray(y_true, dtype=float)
    e_a = np.abs(y_true - np.asarray(pred_a, dtype=float))
    e_b = np.abs(y_true - np.asarray(pred_b, dtype=float))
    n = len(e_a)
    if n < 3:
        raise ValueError(f"Amostra insuficiente para bootstrap: {n} pontos.")

    rng = np.random.default_rng(seed)
    idx = _indices_em_blocos(n, horizon, n_amostras, rng)

    mae_a, mae_b = e_a[idx].mean(axis=1), e_b[idx].mean(axis=1)
    ganho_mae = mae_b - mae_a                       # positivo = pred_a melhor
    with np.errstate(divide="ignore", invalid="ignore"):
        ganho_pct = np.where(mae_b > 0, ganho_mae / mae_b * 100, np.nan)

    q = [alfa / 2 * 100, (1 - alfa / 2) * 100]
    lo_mae, hi_mae = np.percentile(ganho_mae, q)
    lo_pct, hi_pct = np.nanpercentile(ganho_pct, q)

    return {
        "ganho_mae": float(e_b.mean() - e_a.mean()),
        "ganho_mae_lo": float(lo_mae),
        "ganho_mae_hi": float(hi_mae),
        "ganho_pct": float((e_b.mean() - e_a.mean()) / e_b.mean() * 100),
        "ganho_pct_lo": float(lo_pct),
        "ganho_pct_hi": float(hi_pct),
        "inclui_zero": bool(lo_mae <= 0 <= hi_mae),
        "cobertura_pct": round((1 - alfa) * 100),
        "n_pontos": n,
        "n_blocos": int(np.ceil(n / max(1, min(horizon, n)))),
        "amostras_bootstrap": n_amostras,
        "distribuicao_pct": ganho_pct,
    }


def efeito_minimo_detectavel(
    y_true,
    pred_a,
    pred_b,
    horizon: int,
    ganhos=None,
    n_simulacoes: int = 600,
    alfa: float = 0.05,
    poder_alvo: float = 0.80,
    seed: int = 42,
) -> dict:
    """Menor vantagem real que este experimento conseguiria detectar.

    Método
    ------
    Os erros observados dos dois modelos são o material — preservam a
    distribuição real e a correlação entre eles, que um gerador sintético não
    reproduziria. Sobre esse material impomos uma vantagem **conhecida**:
    os erros de ``pred_a`` são reescalados até que seu MAE seja exatamente
    ``(1 - g)`` vezes o de ``pred_b``.

    Para cada ``g``, reamostramos em blocos e rodamos o Diebold-Mariano.
    A fração de reamostragens em que o teste rejeita é o **poder** para
    aquele tamanho de efeito. O menor ``g`` cujo poder alcança ``poder_alvo``
    é o efeito mínimo detectável.

    A leitura que isso permite: se o ganho observado for menor que o efeito
    mínimo detectável, "não houve diferença" descreve o **experimento**, não
    os modelos.

    Returns
    -------
    dict
        ``curva_poder`` (ganho imposto -> poder), ``mde_pct``,
        ``poder_no_ganho_observado`` e ``ganho_observado_pct``.
        ``mde_pct`` é ``None`` quando nem o maior ganho testado atinge o alvo.
    """
    y_true = np.asarray(y_true, dtype=float)
    e_a = y_true - np.asarray(pred_a, dtype=float)
    e_b = y_true - np.asarray(pred_b, dtype=float)
    n = len(e_a)
    if n < 3:
        raise ValueError(f"Amostra insuficiente para análise de poder: {n} pontos.")

    mae_a, mae_b = np.abs(e_a).mean(), np.abs(e_b).mean()
    if mae_a <= 0 or mae_b <= 0:
        raise ValueError("Erro médio nulo: não há o que escalar.")
    ganho_observado = (mae_b - mae_a) / mae_b * 100

    if ganhos is None:
        ganhos = np.round(np.arange(0.0, 0.55, 0.05), 3)
    ganhos = [float(g) for g in ganhos]

    rng = np.random.default_rng(seed)
    idx = _indices_em_blocos(n, horizon, n_simulacoes, rng)
    zeros = np.zeros(n)

    curva = {}
    for g in ganhos:
        # Reescala os erros de A até que seu MAE seja (1-g) vezes o de B.
        fator = (1 - g) * mae_b / mae_a
        rejeicoes = 0
        for linha in idx:
            ea, eb = e_a[linha] * fator, e_b[linha]
            # y_true=0 e previsões negadas fazem (y_true - pred) devolver os
            # próprios erros — evita reconstruir realizados só para reusar o DM.
            _, p = diebold_mariano(zeros, -ea, -eb, horizon=horizon)
            if np.isfinite(p) and p < alfa:
                rejeicoes += 1
        curva[g] = rejeicoes / len(idx)

    atingem = [g for g, poder in curva.items() if poder >= poder_alvo]
    mde = min(atingem) if atingem else None

    # Taxa de rejeição quando a vantagem imposta é ZERO. Deveria ficar em torno
    # de ``alfa``: é a chance de o teste inventar uma diferença que não existe.
    # Muito acima disso, o teste está mal calibrado neste regime — o que
    # costuma acontecer quando sobram poucos blocos por reamostragem — e tanto
    # o poder quanto o efeito mínimo detectável devem ser lidos como grosseiros.
    falso_positivo = curva.get(0.0, float("nan"))

    # Poder no tamanho de efeito realmente observado, por interpolação.
    gs = np.array(sorted(curva))
    poderes = np.array([curva[g] for g in gs])
    alvo = max(0.0, ganho_observado / 100)
    poder_observado = float(np.interp(alvo, gs, poderes))

    return {
        "curva_poder": {round(g * 100, 1): round(p, 3) for g, p in curva.items()},
        "mde_pct": None if mde is None else round(mde * 100, 1),
        "poder_alvo": poder_alvo,
        "alfa": alfa,
        "ganho_observado_pct": round(float(ganho_observado), 2),
        "poder_no_ganho_observado": round(poder_observado, 3),
        "taxa_falso_positivo": round(float(falso_positivo), 3),
        "bem_calibrado": bool(falso_positivo <= 2 * alfa),
        "n_pontos": n,
        "n_blocos": int(np.ceil(n / max(1, min(horizon, n)))),
        "n_simulacoes": n_simulacoes,
    }
