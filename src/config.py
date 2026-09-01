"""
Leitura da configuração única do pipeline (``config/config.yaml``).

O arquivo YAML deixou de ser documentação e passou a ser a fonte dos
parâmetros: este módulo o lê e monta os objetos tipados que o resto do
pipeline consome — ``SeriesContract``, ``BacktestConfig``, os parâmetros de
engenharia de atributos e os caminhos de saída.

Posição na arquitetura
----------------------
A regra de dependência do projeto é ``data -> features -> models ->
evaluation``. Este módulo fica *acima* de todas elas: importa de ``data`` e
de ``evaluation`` para montar os objetos, e **nenhum módulo de ``src/`` o
importa**. Quem o usa são os ``scripts/``. Não há ciclo, e cada camada
continua testável sem precisar carregar configuração de disco.

Resolução de caminhos
---------------------
Todo caminho declarado no YAML é relativo à raiz do projeto e é resolvido
aqui, contra ``ROOT``. Assim os scripts rodam de qualquer diretório de
trabalho e em qualquer sistema operacional — que era exatamente o que os
caminhos absolutos anteriores impediam.

Chaves ausentes, chaves erradas
------------------------------
São coisas diferentes e recebem tratamento diferente.

Chave **ausente** degrada para um default: um YAML incompleto continua rodando
com o comportamento documentado, em vez de quebrar.

Chave **presente e inválida** falha na leitura, não no uso. ``validar()`` roda
dentro de ``load()`` e checa estrutura, tipos e coerência entre seções — um
horizonte sem campeão declarado, por exemplo, é erro de configuração e aparece
em milissegundos, não depois de sete minutos de benchmark. É o mesmo princípio
do contrato de dados em :mod:`src.data.loader`: falhar cedo e alto.

Sem efeito colateral em leitura
-------------------------------
Os acessores de caminho **não criam diretórios**. Ler configuração não deveria
mexer no disco — um ``print(cfg.figures_dir)`` num teste não pode materializar
pastas. Quem vai escrever chama ``preparar_diretorios()`` uma vez, no início.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from src.data.loader import SeriesContract
from src.evaluation.backtest import BacktestConfig

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

__all__ = ["DEFAULT_CONFIG_PATH", "ROOT", "PipelineConfig"]

#: Raiz do projeto — o diretório que contém ``src/``, ``config/`` e ``data/``.
ROOT: Path = Path(__file__).resolve().parents[1]

#: Caminho padrão do arquivo de configuração.
DEFAULT_CONFIG_PATH: Path = ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class PipelineConfig:
    """Configuração do pipeline, lida de ``config/config.yaml``.

    Attributes
    ----------
    conteudo : dict
        Conteúdo do YAML. Exposto para inspeção e para chaves que ainda não
        tenham acessor dedicado. Não se chama ``raw`` de propósito: neste
        projeto ``raw`` é o nome da camada de dados de entrada, e reaproveitar
        o termo para "o dicionário do YAML" confundiria duas coisas sem
        relação.
    root : Path
        Raiz usada para resolver caminhos relativos.
    source : Path
        Arquivo efetivamente lido — útil em mensagem de erro e log.
    """

    conteudo: dict[str, Any] = field(default_factory=dict)
    root: Path = ROOT
    source: Path = DEFAULT_CONFIG_PATH

    # -- construção ---------------------------------------------------------
    @classmethod
    def load(
        cls,
        path: str | Path | None = None,
        root: str | Path | None = None,
    ) -> PipelineConfig:
        """Lê o YAML e devolve a configuração.

        Parameters
        ----------
        path : str or Path, optional
            Arquivo a ler. Usa ``config/config.yaml`` da raiz se omitido.
        root : str or Path, optional
            Raiz para resolver caminhos relativos. Usa a raiz do projeto
            se omitida.

        Raises
        ------
        FileNotFoundError
            Se o arquivo não existir. Falhar aqui é preferível a rodar com
            parâmetros silenciosamente diferentes dos documentados.
        """
        cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Configuração não encontrada: {cfg_path}. "
                f"O pipeline depende dela para não ter número mágico no código."
            )
        with cfg_path.open(encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        # ValueError, e não TypeError: o que está errado é o *conteúdo* do
        # arquivo, não o tipo de um argumento — mesma convenção do contrato
        # de dados em `src/data/loader.py`.
        if not isinstance(loaded, dict):
            raise ValueError(  # noqa: TRY004
                f"{cfg_path} não contém um mapeamento YAML no topo."
            )
        cfg = cls(
            conteudo=loaded,
            root=Path(root).resolve() if root is not None else ROOT,
            source=cfg_path,
        )
        cfg.validar()
        return cfg

    # -- validação ----------------------------------------------------------
    def validar(self) -> None:
        """Checa estrutura, tipos e coerência entre seções. Falha alto.

        Roda dentro de ``load()``. O objetivo é que erro de configuração custe
        milissegundos, e não sete minutos de benchmark antes de estourar no
        meio de um script.

        Valida apenas o que foi **declarado**: chave ausente continua caindo no
        default, porque degradar graciosamente na ausência e reclamar na
        presença de valor inválido são comportamentos diferentes e ambos
        desejáveis.

        Raises
        ------
        ValueError
            Com a chave exata e o valor recebido, para que a correção não
            exija adivinhação.
        """
        erros: list[str] = []

        for secao in ("data", "contract", "features", "backtest", "model",
                      "output", "validation"):
            valor = self.conteudo.get(secao)
            if valor is not None and not isinstance(valor, dict):
                erros.append(f"'{secao}' deveria ser um mapeamento, veio {type(valor).__name__}")

        horizontes = self._get("backtest", "horizons", None)
        if horizontes is not None:
            if not isinstance(horizontes, (list, tuple)) or not horizontes:
                erros.append(f"backtest.horizons deveria ser uma lista não vazia, veio {horizontes!r}")
            elif any(not isinstance(h, int) or h < 1 for h in horizontes):
                erros.append(f"backtest.horizons só aceita inteiros >= 1, veio {horizontes!r}")

        for secao, chave in (("backtest", "n_test"), ("backtest", "min_train"),
                             ("backtest", "season"), ("features", "n_fourier")):
            valor = self._get(secao, chave, None)
            if valor is not None and (not isinstance(valor, int) or valor < 1):
                erros.append(f"{secao}.{chave} deveria ser inteiro >= 1, veio {valor!r}")

        for chave in ("lags", "roll_windows"):
            valor = self._get("features", chave, None)
            if valor is not None and (
                not isinstance(valor, (list, tuple))
                or any(not isinstance(v, int) or v < 1 for v in valor)
            ):
                erros.append(f"features.{chave} só aceita inteiros >= 1, veio {valor!r}")

        # Coerência entre seções: todo horizonte avaliado precisa de campeão.
        # É o erro mais provável ao editar o YAML, e o mais caro de descobrir
        # tarde — ele só apareceria no meio do run_final.
        campeao = self._get("model", "champion", None)
        if isinstance(campeao, dict) and horizontes:
            declarados = {int(h) for h in campeao}
            faltando = sorted(set(horizontes) - declarados)
            if faltando:
                erros.append(
                    f"model.champion não declara campeão para o(s) horizonte(s) "
                    f"{faltando}; declarados: {sorted(declarados)}"
                )

        if erros:
            detalhe = "\n  - ".join(erros)
            raise ValueError(f"Configuração inválida em {self.source}:\n  - {detalhe}")

    # -- utilitários internos ----------------------------------------------
    def _get(self, section: str, key: str, default: Any) -> Any:
        secao = self.conteudo.get(section)
        if not isinstance(secao, dict):
            return default
        return secao.get(key, default)

    def _resolve(self, value: str | Path) -> Path:
        """Torna absoluto um caminho declarado no YAML."""
        p = Path(value)
        return p if p.is_absolute() else self.root / p

    # -- caminhos -----------------------------------------------------------
    @property
    def data_path(self) -> Path:
        """CSV bruto da série."""
        return self._resolve(
            self._get("data", "raw_path", "data/raw/candy_production.csv")
        )

    def _caminho_opcional(self, chave: str) -> Path | None:
        """Caminho declarado no YAML que pode legitimamente não existir.

        Devolve ``None`` quando a chave não está declarada **ou** quando o
        arquivo não está no disco. Quem chama decide o que fazer com a ausência
        — no caso das séries de preço, pular a seção correspondente da EDA em
        vez de quebrar o pipeline.
        """
        valor = self._get("data", chave, None)
        if not valor:
            return None
        caminho = self._resolve(valor)
        return caminho if caminho.exists() else None

    @property
    def sugar_world_path(self) -> Path | None:
        """CSV do preço mundial do açúcar (No. 11), se disponível."""
        return self._caminho_opcional("sugar_world_path")

    @property
    def sugar_us_path(self) -> Path | None:
        """CSV do preço do açúcar nos EUA (No. 16), se disponível."""
        return self._caminho_opcional("sugar_us_path")

    @property
    def tem_precos_acucar(self) -> bool:
        """Se as duas séries de preço estão presentes — a análise exige ambas."""
        return self.sugar_world_path is not None and self.sugar_us_path is not None

    @property
    def refined_dir(self) -> Path:
        """Diretório da camada validada e conformada."""
        return self._resolve(self._get("data", "refined_dir", "data/refined"))

    @property
    def refined_path(self) -> Path:
        """Parquet da série conformada — o que o pipeline efetivamente lê."""
        return self.refined_dir / "producao_mensal.parquet"

    @property
    def gold_dir(self) -> Path:
        """Diretório da camada pronta para consumo."""
        return self._resolve(self._get("data", "gold_dir", "data/gold"))

    @property
    def figures_dir(self) -> Path:
        """Diretório das figuras."""
        return self._resolve(self._get("output", "figures_dir", "reports/figures"))

    def preparar_diretorios(self) -> None:
        """Cria os diretórios de saída. Chamar uma vez, por quem vai escrever.

        Existe para que os acessores de caminho permaneçam **puros**: ler uma
        configuração não deveria materializar pastas no disco. Um teste que
        inspeciona ``cfg.figures_dir`` não pode deixar rastro.
        """
        for d in (self.refined_dir, self.gold_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)
        for p in (self.metrics_path, self.eda_path, self.benchmark_path,
                  self.uncertainty_path, self.sensitivity_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    @property
    def metrics_path(self) -> Path:
        """JSON com métricas e forecast do modelo campeão."""
        return self._output_file("metrics_path", "reports/metrics.json")

    @property
    def eda_path(self) -> Path:
        """JSON com os números da análise exploratória."""
        return self._output_file("eda_path", "reports/eda.json")

    @property
    def benchmark_path(self) -> Path:
        """CSV com a tabela completa do benchmark."""
        return self._output_file("benchmark_path", "reports/benchmark.csv")

    @property
    def uncertainty_path(self) -> Path:
        """JSON com o intervalo do ganho e o poder do teste."""
        return self._output_file("uncertainty_path", "reports/uncertainty.json")

    @property
    def sensitivity_path(self) -> Path:
        """JSON com o ganho medido em cada janela de teste."""
        return self._output_file("sensitivity_path", "reports/sensitivity.json")

    @property
    def janelas_sensibilidade(self) -> tuple[int, ...]:
        """Tamanhos de janela de teste comparados na análise de sensibilidade.

        A janela máxima disponível é acrescentada pelo script, porque depende
        do tamanho da matriz supervisionada e do ``min_train``.
        """
        return tuple(
            int(j) for j in self._get("backtest", "janelas_sensibilidade", (36, 72, 144))
        )

    def _output_file(self, key: str, default: str) -> Path:
        return self._resolve(self._get("output", key, default))

    # -- acesso ao dado -----------------------------------------------------
    def carregar_serie(self) -> pd.Series:
        """Devolve a série alvo, lendo da camada correta.

        Centraliza a decisão de qual camada consumir. Hoje é ``refined`` — o
        contrato de dados já foi aplicado na construção dela, e revalidar a
        cada leitura seria desperdício. A verificação de atualidade contra
        ``raw`` vai junto: refined mais velha que a origem falha alto, em vez
        de devolver dado desatualizado em silêncio.

        Ter isso num lugar só evita que cada script decida por conta própria de
        onde ler — que é como camadas de dados costumam virar decoração.
        """
        from src.data.loader import load_refined_series

        return load_refined_series(self.refined_path, raw_path=self.data_path)

    # -- objetos do pipeline -----------------------------------------------
    def series_contract(self) -> SeriesContract:
        """Contrato de dados aplicado na ingestão."""
        return SeriesContract(
            date_col=self._get("data", "date_col", "observation_date"),
            value_col=self._get("data", "value_col", "IPG3113N"),
            freq=self._get("data", "freq", "MS"),
            min_value=float(self._get("contract", "min_value", 10.0)),
            max_value=float(self._get("contract", "max_value", 500.0)),
            min_observations=int(self._get("contract", "min_observations", 120)),
        )

    def backtest_config(
        self, horizons: tuple[int, ...] | None = None, embargo: bool | None = None
    ) -> BacktestConfig:
        """Protocolo de validação.

        Parameters
        ----------
        horizons : tuple[int, ...], optional
            Sobrescreve os horizontes do YAML. Usado por scripts que
            avaliam um horizonte só.
        embargo : bool, optional
            Sobrescreve ``validation.embargo``. Existe apenas para o
            script de auditoria comparar os dois protocolos; reportar
            resultado com ``False`` é sempre errado.
        """
        return BacktestConfig(
            n_test=int(self._get("backtest", "n_test", 36)),
            horizons=tuple(
                horizons
                if horizons is not None
                else self._get("backtest", "horizons", (1, 3, 12))
            ),
            min_train=int(self._get("backtest", "min_train", 240)),
            season=int(self._get("backtest", "season", 12)),
            embargo=bool(
                embargo
                if embargo is not None
                else self._get("validation", "embargo", True)
            ),
        )

    # -- features e modelo --------------------------------------------------
    @property
    def feature_params(self) -> dict[str, Any]:
        """Parâmetros de ``make_supervised_frame``, prontos para ``**kwargs``.

        ``use_exogenous`` fica de fora de propósito: é a chave do teste de
        ablação e cada script a controla explicitamente, para que ligar ou
        desligar as exógenas seja sempre visível no ponto de chamada.
        """
        return {
            "n_fourier": int(self._get("features", "n_fourier", 2)),
            "lags": tuple(self._get("features", "lags", (1, 2, 3, 6, 12, 13, 24))),
            "roll_windows": tuple(self._get("features", "roll_windows", (3, 12))),
        }

    @property
    def use_exogenous(self) -> bool:
        """Se as exógenas de calendário entram na matriz de atributos."""
        return bool(self._get("features", "use_exogenous", True))

    @property
    def champions(self) -> dict[int, str]:
        """Campeão de cada horizonte: ``{horizonte: nome_no_registry}``.

        O YAML aceita duas formas:

        - **mapa** ``{1: xgboost, 12: ridge_fourier}`` — um campeão por
          horizonte, que é a forma correta. A previsão direta treina um modelo
          por horizonte, e a informação disponível muda com ele: em ``h=1`` há
          defasagens recentes que as árvores exploram; em ``h=12`` sobra o eco
          sazonal, que o baseline já usa. Nada obriga o mesmo vencedor.
        - **string única** ``ridge_fourier`` — aplicada a todos os horizontes.
          Mantida para configurações antigas e para o caso em que a escolha
          realmente não depende do horizonte.
        """
        bruto = self._get("model", "champion", "ridge_fourier")
        horizontes = self.backtest_config().horizons
        if isinstance(bruto, dict):
            mapa = {int(h): str(n) for h, n in bruto.items()}
            faltando = [h for h in horizontes if h not in mapa]
            if faltando:
                raise ValueError(
                    f"model.champion não declara campeão para o(s) horizonte(s) "
                    f"{faltando}. Horizontes avaliados: {list(horizontes)}."
                )
            return mapa
        return dict.fromkeys(horizontes, str(bruto))

    def champion_for(self, horizon: int) -> str:
        """Campeão de um horizonte específico."""
        try:
            return self.champions[int(horizon)]
        except KeyError:
            raise ValueError(
                f"Nenhum campeão declarado para h={horizon} em model.champion."
            ) from None

    @property
    def random_state(self) -> int:
        """Semente única. Fixa por requisito de auditoria."""
        return int(self._get("model", "random_state", 42))
