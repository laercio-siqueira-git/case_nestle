"""Escrita dos artefatos JSON do pipeline, num lugar só.

Por que este módulo existe
--------------------------
``json.dump`` do Python escreve ``NaN``, ``Infinity`` e ``-Infinity`` por
padrão. Nenhum dos três é JSON válido: a especificação (RFC 8259) só admite
números finitos. O arquivo *parece* certo, abre no editor, e quebra na hora de
ser lido por qualquer parser estrito — ``JSON.parse`` do JavaScript, o leitor
do Spark, a maioria dos clientes de API.

Isso não é hipótese neste projeto: o ``eda.json`` saía com ``NaN`` literal e não
passava em parser estrito. Como um artefato que o Spark não lê é exatamente o
que a arquitetura de produção proposta iria consumir, a correção fica na
fronteira de escrita — nulo é o jeito de JSON dizer "sem valor", e é o que o
Parquet e o Spark leem como nulo do outro lado.

Uma ressalva sobre o que isto **não** conserta: converter para nulo trata a
representação, nunca a causa. Naquele caso a causa era um teste estatístico que
não rodava, e o nulo o tornava indistinguível de um teste que rodou sem achar
nada. Quem produz o valor é responsável por decidir se a ausência é legítima;
este módulo só garante que ela seja escrita de forma legível.

O segundo problema, mais silencioso
-----------------------------------
``numpy.float64``, ``numpy.int64`` e ``numpy.bool_`` não são serializáveis por
``json.dump``. Como quase todo número deste pipeline sai do numpy ou do pandas,
qualquer campo novo é um ``TypeError`` em potencial, descoberto no fim de uma
rodada de sete minutos. A conversão vai junto, pelo mesmo motivo de estar aqui:
quem escreve artefato não deveria precisar lembrar disso.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["formatar_json", "proveniencia", "salvar_json", "sanitizar_json"]

#: Bibliotecas cuja versão muda resultado numérico. Não é a lista de
#: dependências: é a lista do que altera um número entre duas máquinas.
_LIBS = ("numpy", "pandas", "scikit-learn", "scipy",
         "xgboost", "lightgbm")


def _versao(nome: str) -> str | None:
    try:
        return importlib.metadata.version(nome)
    except importlib.metadata.PackageNotFoundError:
        return None


def _commit() -> str | None:
    """Hash do commit, quando o projeto está num repositório git."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None


def hash_arquivo(caminho: str | Path) -> str | None:
    """SHA-256 dos primeiros 12 caracteres — identifica a versão do dado.

    Curto de propósito: serve para responder "é o mesmo arquivo?", não para
    garantia criptográfica.
    """
    p = Path(caminho)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for bloco in iter(lambda: fh.read(1 << 20), b""):
            h.update(bloco)
    return h.hexdigest()[:12]


def proveniencia(dado: str | Path | None = None) -> dict[str, Any]:
    """Com o que este artefato foi produzido: código, dado e bibliotecas.

    Por que isto existe
    -------------------
    O projeto promete que qualquer pessoa reproduz os números. Só que "mesmo
    código, mesmo dado, mesma semente" **não** basta: modelos de árvore mudam
    de resultado entre versões de scikit-learn, e o pandas alterou o
    comportamento de tipos entre majors. Rodar este repositório noutra máquina
    devolve os mesmos vereditos, e não exatamente as mesmas casas decimais.

    Isso não é defeito enquanto estiver **declarado**. O que transformava em
    defeito era o artefato não dizer em que ambiente nasceu: aí a divergência
    vira suspeita de erro, e não informação. Gravar versões, commit e hash do
    dado bruto converte "não reproduziu" em "reproduziu noutro ambiente, eis
    qual" — que é uma afirmação verificável.

    Parameters
    ----------
    dado : str or Path, optional
        Arquivo de dado bruto a identificar por hash.
    """
    return {
        "python": platform.python_version(),
        "plataforma": platform.system(),
        "commit": _commit(),
        "hash_dado_bruto": hash_arquivo(dado) if dado is not None else None,
        "bibliotecas": {n: v for n in _LIBS if (v := _versao(n)) is not None},
    }


def sanitizar_json(valor: Any) -> Any:
    """Converte um objeto para tipos que o JSON representa de verdade.

    Regras, na ordem em que são aplicadas:

    - ``NaN``, ``inf`` e ``-inf`` viram ``None``. É a tradução honesta: JSON
      não tem como escrever "indefinido" a não ser com nulo.
    - Escalares do numpy viram os equivalentes do Python.
    - ``dict`` e sequências são percorridos recursivamente; chaves viram
      ``str``, porque JSON não admite outra coisa.

    O resto passa intacto — inclusive tipos que ``json`` não conhece, que
    seguem para estourar em ``salvar_json`` com a mensagem própria do módulo
    ``json``, em vez de sumirem convertidos em texto por engano.
    """
    if isinstance(valor, (np.floating, float)):
        f = float(valor)
        return None if math.isnan(f) or math.isinf(f) else f
    if isinstance(valor, (np.bool_, bool)):
        return bool(valor)
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.ndarray):
        return [sanitizar_json(v) for v in valor.tolist()]
    if isinstance(valor, dict):
        return {str(k): sanitizar_json(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple, set)):
        return [sanitizar_json(v) for v in valor]
    return valor


def salvar_json(dados: Any, caminho: str | Path, *, indent: int = 2) -> Path:
    """Grava um artefato JSON válido e devolve o caminho escrito.

    ``allow_nan=False`` é cinto de segurança, não a correção: a sanitização
    já removeu o que não é finito. Se ainda assim algo escapar — um tipo novo,
    um caminho não previsto —, a escrita **falha** em vez de produzir um
    arquivo que só vai quebrar semanas depois, na máquina de outra pessoa.
    """
    destino = Path(caminho)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8") as fh:
        json.dump(sanitizar_json(dados), fh, indent=indent,
                  ensure_ascii=False, allow_nan=False)
    return destino


def formatar_json(dados: Any, *, indent: int = 2) -> str:
    """Mesma serialização, como texto — para imprimir o que foi gravado.

    Existe para que a saída no terminal e o arquivo em disco não possam
    divergir: os dois passam pela mesma sanitização.
    """
    return json.dumps(sanitizar_json(dados), indent=indent,
                      ensure_ascii=False, allow_nan=False)
