"""Roda o pipeline inteiro, na ordem correta, em qualquer sistema operacional.

    python scripts/run_all.py
    python scripts/run_all.py --skip audit          # pula a etapa mais lenta
    python scripts/run_all.py --only eda benchmark
    python scripts/run_all.py --config outro.yaml

Por que existe, tendo Makefile
------------------------------
O ``Makefile`` cobre Linux e macOS, mas ``make`` não vem instalado no Windows —
onde este projeto é desenvolvido. Um pipeline que só roda na máquina de quem
tem a ferramenta certa não é reprodutível, e reprodutibilidade é a premissa do
repositório inteiro. Este script não depende de nada além do Python que já é
obrigatório.

Ele usa ``sys.executable``, ou seja, o **mesmo interpretador** que o invocou.
Rodando pelo Python do ambiente virtual, todas as etapas herdam esse ambiente —
sem risco de o ``python`` do PATH ser outro, que é a origem clássica de
"funciona na minha máquina".

A ordem não é cosmética
-----------------------
Ela segue o fluxo do dado, e as duas pontas são camadas:

``refined`` vem logo depois dos testes porque **toda análise lê dela** — é a
tabela conformada em Parquet que a EDA, o benchmark e o forecast consomem. Se
ela não existir, nada roda; se estiver velha, a leitura falha alto em vez de
devolver dado defasado em silêncio.

``gold`` vem no fim pelo motivo simétrico: ela é feita dos artefatos que o
modelo produz, então não há o que materializar antes de eles existirem.

No meio, ``build_results.py`` lê o que ``benchmark``, ``forecast`` e
``uncertainty`` produzem. Rodar fora de ordem geraria um relatório novo sobre
números velhos — exatamente o problema que o resto do projeto se esforça para
impedir. Por isso a sequência é fixa e a execução para no primeiro erro: um
pipeline que segue depois de uma falha entrega resultado parcial com cara de
completo.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

#: (nome, argumentos, aceita --config). A ordem desta lista é a ordem de execução.
ETAPAS: list[tuple[str, list[str], bool]] = [
    ("lint", ["-m", "ruff", "check", "src/", "scripts/", "tests/"], False),
    ("test", ["-m", "pytest", "tests/", "-q"], False),
    ("refined", ["scripts/build_layers.py", "--stage", "refined"], True),
    ("audit", ["scripts/audit.py"], True),
    ("eda", ["scripts/run_eda.py"], True),
    ("benchmark", ["scripts/run_benchmark.py"], True),
    ("forecast", ["scripts/run_final.py"], True),
    ("uncertainty", ["scripts/run_uncertainty.py"], True),
    ("sensitivity", ["scripts/run_sensitivity.py"], True),
    ("gold", ["scripts/build_layers.py", "--stage", "gold"], True),
    ("results", ["scripts/build_results.py"], True),
]
NOMES = [nome for nome, _, _ in ETAPAS]


def _selecionar(args) -> list[tuple[str, list[str], bool]]:
    escolhidas = ETAPAS
    if args.only:
        escolhidas = [e for e in escolhidas if e[0] in args.only]
    if args.skip:
        escolhidas = [e for e in escolhidas if e[0] not in args.skip]
    if not escolhidas:
        raise SystemExit("Nenhuma etapa selecionada.")
    return escolhidas


def _formatar(segundos: float) -> str:
    return f"{segundos:5.1f}s" if segundos < 60 else f"{segundos / 60:5.1f}min"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None,
                    help="YAML alternativo, repassado às etapas que o aceitam")
    ap.add_argument("--only", nargs="+", choices=NOMES, help="roda apenas estas etapas")
    ap.add_argument("--skip", nargs="+", choices=NOMES, help="pula estas etapas")
    ap.add_argument("--quiet", action="store_true",
                    help="esconde a saída das etapas; mostra só em caso de falha")
    args = ap.parse_args()

    etapas = _selecionar(args)
    print(f"pipeline: {' -> '.join(nome for nome, _, _ in etapas)}")
    print(f"python  : {sys.executable}")
    if args.config:
        print(f"config  : {args.config}")

    tempos: list[tuple[str, float, bool]] = []
    inicio_geral = time.perf_counter()

    for i, (nome, argumentos, aceita_config) in enumerate(etapas, start=1):
        comando = [sys.executable, *argumentos]
        if aceita_config and args.config:
            comando += ["--config", args.config]

        # flush explícito: fora de um terminal o Python bufferiza a saída, e o
        # progresso de um pipeline de 15 minutos só apareceria no fim — que é
        # justamente quando ele não serve mais para nada.
        print(f"\n{'=' * 70}", flush=True)
        print(f"[{i}/{len(etapas)}] {nome}", flush=True)
        print("=" * 70, flush=True)

        t0 = time.perf_counter()
        # check=False: o código de saída é tratado logo abaixo, com mensagem
        # própria dizendo qual etapa quebrou e o que deixou de rodar.
        proc = subprocess.run(
            comando, cwd=RAIZ, check=False,
            capture_output=args.quiet, text=True, encoding="utf-8", errors="replace",
        )
        decorrido = time.perf_counter() - t0
        ok = proc.returncode == 0
        tempos.append((nome, decorrido, ok))

        if args.quiet and ok:
            print(f"ok ({_formatar(decorrido)})")

        if not ok:
            # Em modo silencioso a saída foi capturada: só faz sentido mostrá-la
            # agora, que é quando ela explica alguma coisa.
            if args.quiet:
                print(proc.stdout or "", end="")
                print(proc.stderr or "", file=sys.stderr, end="")
            print(f"\n{'=' * 70}")
            print(f"FALHOU em '{nome}' (código {proc.returncode}) após "
                  f"{_formatar(decorrido)}.")
            print("Etapas seguintes NÃO foram executadas: um relatório parcial "
                  "com cara de completo é pior que nenhum relatório.")
            print("=" * 70)
            _resumo(tempos, time.perf_counter() - inicio_geral)
            return proc.returncode

    print(f"\n{'=' * 70}")
    print("PIPELINE COMPLETO")
    print("=" * 70)
    _resumo(tempos, time.perf_counter() - inicio_geral)
    return 0


def _resumo(tempos: list[tuple[str, float, bool]], total: float) -> None:
    print()
    for nome, seg, ok in tempos:
        print(f"  {'ok    ' if ok else 'FALHOU'}  {nome:14s} {_formatar(seg)}")
    print(f"  {'':6s}  {'total':14s} {_formatar(total)}")


if __name__ == "__main__":
    raise SystemExit(main())
