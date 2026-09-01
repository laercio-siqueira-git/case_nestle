"""
Testes do pipeline.

Filosofia: testar as *garantias*, não a implementação. Cada teste abaixo
corresponde a uma afirmação que o documento de insights faz sobre o modelo.
Se um teste quebra, uma afirmação do relatório deixou de ser verdadeira.

O teste mais importante do arquivo é `test_no_leakage_at_horizon_12`. Ele
existe porque o vazamento descrito na seção 5.2 do relatório passou
despercebido na primeira versão — e um bug que já aconteceu uma vez merece
um teste permanente.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifacts import formatar_json, salvar_json, sanitizar_json
from src.config import PipelineConfig
from src.data.calendar_features import (
    build_calendar_frame,
    easter_sunday,
    us_federal_holidays,
)
from src.data.loader import SeriesContract, load_candy_production
from src.evaluation.backtest import BacktestConfig, train_mask_for
from src.evaluation.metrics import (
    benjamini_hochberg,
    diebold_mariano,
    mae,
    mape,
    mase,
    rmse,
)
from src.evaluation.uncertainty import (
    efeito_minimo_detectavel,
    intervalo_bootstrap,
    veredito_estabilidade,
)
from src.features.build_features import (
    EXOGENOUS_COLS,
    legal_lags_for_horizon,
    make_calendar_features,
    make_lag_features,
    make_supervised_frame,
)
from src.models.registry import SeasonalNaive, build_model_registry


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def toy_series() -> pd.Series:
    """Série sintética com sazonalidade conhecida e tendência linear."""
    idx = pd.date_range("2000-01-01", periods=120, freq="MS")
    seasonal = 10 * np.sin(2 * np.pi * idx.month / 12)
    trend = np.linspace(100, 120, len(idx))
    return pd.Series(seasonal + trend, index=idx, name="production_index")


# ------------------------------------------------------ calendário
class TestCalendar:
    """A Páscoa é a única exógena com regra não trivial: vale testar a fundo."""

    @pytest.mark.parametrize(
        "year,expected",
        [
            (1972, "1972-04-02"),
            (2000, "2000-04-23"),
            (2017, "2017-04-16"),
            (2024, "2024-03-31"),
            (2025, "2025-04-20"),
        ],
    )
    def test_easter_matches_known_dates(self, year, expected):
        assert str(easter_sunday(year)) == expected

    def test_easter_always_between_march_22_and_april_25(self):
        """Limite teórico do computus. Se sair disso, o algoritmo quebrou."""
        for year in range(1972, 2051):
            e = easter_sunday(year)
            assert dt.date(year, 3, 22) <= e <= dt.date(year, 4, 25)

    def test_easter_always_sunday(self):
        assert all(easter_sunday(y).weekday() == 6 for y in range(1972, 2051))

    def test_juneteenth_only_from_2021(self):
        """Evita anacronismo: o feriado não existia no histórico de 1972.

        A verificação é por contagem, e não pela data nominal, porque a regra
        de observância pode deslocar o feriado: 19/jun/2021 caiu num sábado e
        foi observado na sexta, 18/jun.
        """
        assert len(us_federal_holidays(2020)) == 10
        assert len(us_federal_holidays(2023)) == 11
        assert dt.date(2023, 6, 19) in us_federal_holidays(2023)

    def test_business_days_in_plausible_range(self):
        idx = pd.date_range("1972-01-01", "2017-08-01", freq="MS")
        cal = build_calendar_frame(idx)
        assert cal["n_business_days"].between(17, 23).all()

    def test_calendar_is_deterministic(self):
        idx = pd.date_range("2020-01-01", periods=24, freq="MS")
        pd.testing.assert_frame_equal(build_calendar_frame(idx), build_calendar_frame(idx))


# ------------------------------------------------------ contrato de dados
class TestLoader:
    def test_rejects_missing_column(self, tmp_path):
        p = tmp_path / "bad.csv"
        p.write_text("data,valor\n1972-01-01,85.0\n")
        with pytest.raises(ValueError, match="Colunas ausentes"):
            load_candy_production(p)

    def test_rejects_gap_in_grid(self, tmp_path):
        """Um mês faltante desalinha todos os lags sazonais a jusante."""
        p = tmp_path / "gap.csv"
        rows = ["observation_date,IPG3113N"]
        for m in [1, 2, 4, 5]:  # março ausente
            rows.append(f"1972-0{m}-01,100.0")
        p.write_text("\n".join(rows))
        with pytest.raises(ValueError, match="ausente"):
            load_candy_production(p, SeriesContract(min_observations=1))

    def test_rejects_out_of_range_value(self, tmp_path):
        p = tmp_path / "wild.csv"
        p.write_text("observation_date,IPG3113N\n1972-01-01,99999.0\n")
        with pytest.raises(ValueError, match="fora da faixa"):
            load_candy_production(p, SeriesContract(min_observations=1))


# ------------------------------------------------------ configuração
class TestConfig:
    """O YAML governa o pipeline: se ele deixar de ser lido, tudo volta a ser
    número mágico sem que nenhum outro teste perceba."""

    def _escrever(self, tmp_path, texto: str):
        p = tmp_path / "cfg.yaml"
        p.write_text(texto, encoding="utf-8")
        return p

    def test_le_o_yaml_do_projeto(self):
        """O config versionado tem de carregar e apontar para dados que existem."""
        cfg = PipelineConfig.load()
        assert cfg.data_path.exists()
        assert cfg.backtest_config().embargo is True

    def test_todo_horizonte_avaliado_tem_campeao_registrado(self):
        """Campeão declarado no YAML tem de existir no registry, em todo horizonte.

        Sem isto, um erro de digitação em `model.champion` só apareceria no meio
        de `run_final.py`, depois de minutos de backtest.
        """
        cfg = PipelineConfig.load()
        registry = build_model_registry()
        for h in cfg.backtest_config().horizons:
            nome = cfg.champion_for(h)
            assert nome in registry, f"campeão '{nome}' de h={h} não está no registry"

    def test_campeoes_nao_dependem_de_biblioteca_opcional(self):
        """Regressão: o campeão declarado tem de existir sem XGBoost/LightGBM.

        O README promete que o pipeline roda sem os boosters externos. Declarar
        um deles como campeão quebra essa promessa em silêncio — e só apareceria
        num ambiente limpo, provavelmente o de quem está avaliando o projeto.
        """
        cfg = PipelineConfig.load()
        essenciais = build_model_registry(include_optional=False)
        for h in cfg.backtest_config().horizons:
            nome = cfg.champion_for(h)
            assert nome in essenciais, (
                f"campeão de h={h} é '{nome}', que só existe com biblioteca "
                f"opcional instalada"
            )

    def test_valores_do_yaml_chegam_aos_objetos(self, tmp_path):
        cfg = PipelineConfig.load(self._escrever(tmp_path, """
data: {raw_path: x.csv, date_col: d, value_col: v, freq: MS}
contract: {min_value: 1.0, max_value: 9.0, min_observations: 7}
features: {n_fourier: 3, lags: [6, 12], roll_windows: [4], use_exogenous: false}
backtest: {n_test: 11, horizons: [2, 5], min_train: 99, season: 6}
model: {champion: random_forest, random_state: 7}
validation: {embargo: true}
"""), root=tmp_path)
        contrato = cfg.series_contract()
        assert (contrato.min_value, contrato.max_value) == (1.0, 9.0)
        assert contrato.min_observations == 7

        bt = cfg.backtest_config()
        assert (bt.n_test, bt.horizons, bt.min_train, bt.season) == (11, (2, 5), 99, 6)

        assert cfg.feature_params == {"n_fourier": 3, "lags": (6, 12), "roll_windows": (4,)}
        assert cfg.use_exogenous is False
        assert cfg.random_state == 7
        # string única: mesmo campeão em todo horizonte declarado
        assert cfg.champions == {2: "random_forest", 5: "random_forest"}
        assert cfg.champion_for(5) == "random_forest"

    def test_campeao_por_horizonte(self, tmp_path):
        """A forma de mapa permite um campeão diferente por horizonte — que é o
        ponto: h=1 e h=12 não são o mesmo problema."""
        cfg = PipelineConfig.load(self._escrever(tmp_path, """
backtest: {horizons: [1, 12]}
model:
  champion:
    1: xgboost
    12: ridge_fourier
"""), root=tmp_path)
        assert cfg.champions == {1: "xgboost", 12: "ridge_fourier"}
        assert cfg.champion_for(1) == "xgboost"
        assert cfg.champion_for(12) == "ridge_fourier"

    def test_horizonte_sem_campeao_declarado_falha_cedo(self, tmp_path):
        """Falta de campeão estoura ao ler o config, não no meio do run.

        A mensagem precisa nomear o horizonte faltante: `[3]` é a diferença
        entre corrigir o YAML em dez segundos e reler o arquivo inteiro.
        """
        with pytest.raises(ValueError, match=r"não declara campeão.*\[3\]"):
            PipelineConfig.load(self._escrever(tmp_path, """
backtest: {horizons: [1, 3, 12]}
model:
  champion:
    1: xgboost
    12: ridge_fourier
"""), root=tmp_path)

    def test_ler_caminho_nao_toca_no_disco(self, tmp_path):
        """Ler configuração é leitura: nenhum acessor pode criar diretório.

        Um getter que faz `mkdir` transforma um `print(cfg.figures_dir)` num
        efeito colateral, e um teste que só inspeciona caminhos passa a deixar
        rastro. Criar é decisão de quem escreve, e tem método próprio.
        """
        cfg = PipelineConfig.load(self._escrever(tmp_path, """
data: {refined_dir: camadas/prata, gold_dir: camadas/ouro}
output: {figures_dir: saida/figuras}
"""), root=tmp_path)
        assert cfg.refined_dir == tmp_path / "camadas" / "prata"
        assert cfg.gold_dir == tmp_path / "camadas" / "ouro"
        assert not cfg.refined_dir.exists()
        assert not cfg.gold_dir.exists()
        assert not cfg.figures_dir.exists()

    def test_preparar_diretorios_cria_tudo_que_o_pipeline_escreve(self, tmp_path):
        """O método é o ponto único onde a estrutura de saída nasce."""
        cfg = PipelineConfig.load(self._escrever(tmp_path, """
data: {refined_dir: camadas/prata, gold_dir: camadas/ouro}
output: {figures_dir: saida/figuras, metrics_path: saida/rel/metrics.json}
"""), root=tmp_path)
        cfg.preparar_diretorios()
        assert cfg.refined_dir.is_dir()
        assert cfg.gold_dir.is_dir()
        assert cfg.figures_dir.is_dir()
        assert cfg.metrics_path.parent.is_dir()
        # Idempotente: rodar o pipeline duas vezes não pode quebrar na segunda.
        cfg.preparar_diretorios()

    def test_config_invalido_falha_na_leitura(self, tmp_path):
        """Chave presente e inválida estoura em `load`, não sete minutos depois.

        Distinção deliberada: chave *ausente* cai no default e o pipeline segue;
        chave *declarada com valor impossível* é erro humano de edição do YAML e
        precisa aparecer no primeiro milissegundo.
        """
        with pytest.raises(ValueError, match="backtest.horizons"):
            PipelineConfig.load(self._escrever(tmp_path, """
backtest: {horizons: [0, 3]}
"""), root=tmp_path)

        with pytest.raises(ValueError, match="não declara campeão"):
            PipelineConfig.load(self._escrever(tmp_path, """
backtest: {horizons: [1, 3, 12]}
model:
  champion: {1: ridge_fourier, 12: ridge_fourier}
"""), root=tmp_path)

        with pytest.raises(ValueError, match="deveria ser inteiro"):
            PipelineConfig.load(self._escrever(tmp_path, """
backtest: {n_test: -5}
"""), root=tmp_path)

    def test_precos_opcionais_ausentes_nao_quebram(self, tmp_path):
        """Arquivo declarado mas inexistente conta como ausente, não como erro.

        É o que permite clonar o repositório sem as séries de preço e ainda
        rodar a EDA — a seção correspondente simplesmente não é gerada.
        """
        cfg = PipelineConfig.load(self._escrever(tmp_path, """
data: {sugar_world_path: nao/existe.csv, sugar_us_path: tambem/nao.csv}
"""), root=tmp_path)
        assert cfg.sugar_world_path is None
        assert cfg.sugar_us_path is None
        assert cfg.tem_precos_acucar is False

    def test_caminho_relativo_resolve_contra_a_raiz(self, tmp_path):
        """Caminho do YAML é relativo ao projeto, não ao diretório de trabalho."""
        cfg = PipelineConfig.load(
            self._escrever(tmp_path, "data: {raw_path: data/raw/x.csv}\n"), root=tmp_path
        )
        assert cfg.data_path == tmp_path / "data" / "raw" / "x.csv"
        assert cfg.data_path.is_absolute()

    def test_caminho_absoluto_e_preservado(self, tmp_path):
        absoluto = (tmp_path / "outro" / "x.csv").as_posix()
        cfg = PipelineConfig.load(
            self._escrever(tmp_path, f"data: {{raw_path: {absoluto}}}\n"), root=tmp_path
        )
        assert cfg.data_path == Path(absoluto)

    def test_yaml_ausente_falha_alto(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="não encontrada"):
            PipelineConfig.load(tmp_path / "nao_existe.yaml")

    def test_yaml_que_nao_e_mapeamento_falha(self, tmp_path):
        with pytest.raises(ValueError, match="mapeamento"):
            PipelineConfig.load(self._escrever(tmp_path, "- a\n- b\n"))

    def test_chave_ausente_cai_no_default(self, tmp_path):
        """YAML incompleto degrada para o comportamento anterior, não quebra."""
        cfg = PipelineConfig.load(self._escrever(tmp_path, "model: {champion: ridge_fourier}\n"),
                                  root=tmp_path)
        assert cfg.backtest_config() == BacktestConfig()
        assert cfg.use_exogenous is True

    def test_embargo_pode_ser_sobrescrito_so_para_auditoria(self, tmp_path):
        cfg = PipelineConfig.load(self._escrever(tmp_path, "validation: {embargo: true}\n"),
                                  root=tmp_path)
        assert cfg.backtest_config().embargo is True
        assert cfg.backtest_config(embargo=False).embargo is False


# ------------------------------------------------------ VAZAMENTO
class TestNoLeakage:
    """O grupo de testes mais importante do repositório."""

    def test_legal_lags_exclude_recent(self):
        assert legal_lags_for_horizon(12) == [12, 13, 24]
        assert 1 not in legal_lags_for_horizon(3)

    def test_illegal_lag_raises(self, toy_series):
        with pytest.raises(ValueError, match="ilegal"):
            make_lag_features(toy_series, lags=(1,), min_lag=12)

    def test_no_leakage_at_horizon_12(self, toy_series):
        """Nenhuma feature pode depender de valor posterior a t-12.

        Prova por perturbação: alteramos os últimos 11 valores da série e
        verificamos que a linha de features de uma data anterior não muda.
        Se alguma feature olhasse para frente, ela mudaria.
        """
        X_before, _ = make_supervised_frame(toy_series, horizon=12)
        target_date = X_before.index[-1]

        perturbed = toy_series.copy()
        window_start = target_date - pd.DateOffset(months=11)
        perturbed.loc[window_start:] += 999.0

        X_after, _ = make_supervised_frame(perturbed, horizon=12)
        pd.testing.assert_series_equal(
            X_before.loc[target_date], X_after.loc[target_date], check_names=False
        )

    def test_nome_do_yoy_acompanha_a_ancora(self, toy_series):
        """O sufixo da coluna tem de bater com a defasagem que ela usa.

        Nomes de feature terminam em gráfico de importância, onde ninguém
        confere a fórmula: uma coluna chamada `yoy_diff_12` calculada sobre
        outra defasagem seria uma mentira difícil de pegar.
        """
        b12 = make_lag_features(toy_series, lags=(12,), min_lag=12)
        assert "yoy_diff_12" in b12.columns
        pd.testing.assert_series_equal(
            b12["yoy_diff_12"], toy_series.shift(12) - toy_series.shift(24),
            check_names=False,
        )

        b18 = make_lag_features(toy_series, lags=(18,), min_lag=18)
        assert "yoy_diff_18" in b18.columns, "âncora mudou, nome tem de mudar"
        assert "yoy_diff_12" not in b18.columns
        pd.testing.assert_series_equal(
            b18["yoy_diff_18"], toy_series.shift(18) - toy_series.shift(30),
            check_names=False,
        )

    def test_rolling_window_respects_horizon(self, toy_series):
        """A média móvel de h=12 deve terminar em t-12, não em t-1."""
        lags = make_lag_features(toy_series, lags=(12,), min_lag=12)
        expected = toy_series.shift(12).rolling(3).mean()
        pd.testing.assert_series_equal(
            lags["roll_mean_3"], expected, check_names=False
        )


# ------------------------------------------------------ métricas
class TestMetrics:
    def test_perfect_prediction_is_zero(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == 0 and rmse(y, y) == 0 and mape(y, y) == 0

    def test_rmse_penalises_outliers_more_than_mae(self):
        y = np.zeros(10)
        p = np.zeros(10)
        p[0] = 10.0
        assert rmse(y + 1, p + 1) > mae(y + 1, p + 1)

    def test_mase_is_one_when_model_matches_naive(self):
        """Âncora conceitual do MASE.

        Construímos uma série cujo erro sazonal de treino é conhecido e um
        modelo que erra exatamente o mesmo tanto no teste. O MASE tem de dar
        1: o modelo empata com repetir o ano anterior.
        """
        rng = np.random.default_rng(0)
        idx = pd.date_range("2000-01-01", periods=120, freq="MS")
        y = pd.Series(np.tile(np.arange(12, dtype=float), 10) + rng.normal(0, 1, 120), index=idx)
        train = y.iloc[:96].to_numpy()
        true = y.iloc[96:].to_numpy()

        naive_train_error = np.mean(np.abs(train[12:] - train[:-12]))
        pred = true + naive_train_error  # erra exatamente o MAE do naive
        assert mase(true, pred, train, season=12) == pytest.approx(1.0, rel=1e-9)

    def test_mase_below_one_means_better_than_naive(self):
        rng = np.random.default_rng(1)
        train = rng.normal(100, 10, 120)
        true = rng.normal(100, 10, 24)
        naive_err = np.mean(np.abs(train[12:] - train[:-12]))
        assert mase(true, true + naive_err / 2, train, season=12) < 1.0


# ------------------------------------------------------ significância
class TestDieboldMariano:
    """Impede que 'MAPE menor' volte a ser reportado como 'modelo melhor'."""

    def test_modelos_identicos_nao_diferem(self):
        rng = np.random.default_rng(0)
        y = rng.normal(100, 10, 60)
        p = y + rng.normal(0, 2, 60)
        stat, pval = diebold_mariano(y, p, p, horizon=1)
        assert stat == pytest.approx(0.0, abs=1e-12)
        assert pval == pytest.approx(1.0)

    def test_modelo_muito_melhor_e_detectado(self):
        rng = np.random.default_rng(1)
        y = rng.normal(100, 10, 60)
        bom = y + rng.normal(0, 0.5, 60)
        ruim = y + rng.normal(0, 8, 60)
        stat, pval = diebold_mariano(y, bom, ruim, horizon=1)
        assert stat < 0, "estatística negativa deve indicar o primeiro melhor"
        assert pval < 0.01

    def test_sinal_inverte_ao_trocar_a_ordem(self):
        rng = np.random.default_rng(2)
        y = rng.normal(100, 10, 60)
        a, b = y + rng.normal(0, 1, 60), y + rng.normal(0, 5, 60)
        s1, p1 = diebold_mariano(y, a, b, horizon=1)
        s2, p2 = diebold_mariano(y, b, a, horizon=1)
        assert s1 == pytest.approx(-s2)
        assert p1 == pytest.approx(p2)

    def test_ruido_puro_nao_e_significativo(self):
        """Dois modelos igualmente ruins não podem produzir vencedor."""
        rng = np.random.default_rng(3)
        y = rng.normal(100, 10, 80)
        a, b = y + rng.normal(0, 3, 80), y + rng.normal(0, 3, 80)
        _, pval = diebold_mariano(y, a, b, horizon=1)
        assert pval > 0.05

    def test_horizonte_maior_alarga_o_intervalo(self):
        """Com diferencial persistente, h=12 tem de dar p maior que h=1.

        É o efeito que impede declarar significância em previsão de 12 meses
        com o mesmo desembaraço de uma de 1 mês: origens consecutivas
        compartilham informação, o diferencial de perda fica autocorrelacionado,
        e o erro padrão correto é maior.

        A persistência é construída de propósito. Com ruído independente as
        autocovariâncias amostrais são aleatórias e podem até encolher a
        variância — a propriedade vale para o caso real, não para todo dado.
        """
        rng = np.random.default_rng(4)
        n = 60
        suave = np.convolve(rng.normal(0, 1, n + 11), np.ones(12) / 12, mode="valid")
        y = np.full(n, 100.0)
        a, b = y - suave, y - 2 * suave      # mesmo padrão, erro maior em b
        _, p_h1 = diebold_mariano(y, a, b, horizon=1)
        _, p_h12 = diebold_mariano(y, a, b, horizon=12)
        assert p_h12 > p_h1

    def test_variancia_hac_nunca_e_negativa(self):
        """Regressão: sem peso de Bartlett a variância saía negativa e o teste
        devolvia nan — justamente para pares de modelos bem diferentes."""
        rng = np.random.default_rng(5)
        for _ in range(200):
            y = rng.normal(100, 10, 36)
            a = y + rng.normal(0, 4, 36)
            b = y + rng.normal(0, 6, 36)
            stat, pval = diebold_mariano(y, a, b, horizon=12)
            assert np.isfinite(stat) and np.isfinite(pval), "HAC produziu nan"
            assert 0.0 <= pval <= 1.0

    def test_amostra_minuscula_e_inconclusiva_nao_empate(self):
        stat, pval = diebold_mariano([1.0, 2.0], [1.0, 3.0], [1.0, 4.0], horizon=1)
        assert np.isnan(stat) and np.isnan(pval)

    def test_tamanhos_diferentes_falham(self):
        with pytest.raises(ValueError, match="mesmo tamanho"):
            diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0], [1.0, 2.0, 3.0], horizon=1)

    def test_horizonte_invalido_falha(self):
        with pytest.raises(ValueError, match="horizon"):
            diebold_mariano([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], horizon=0)


# ------------------------------------------------------ múltiplos testes
class TestBenjaminiHochberg:
    """Impede que o menor p de um conjunto grande vire 'descoberta'."""

    def test_q_nunca_e_menor_que_p(self):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.7])
        q = benjamini_hochberg(p)
        assert np.all(q >= p - 1e-12), "a correção só pode ser conservadora"

    def test_q_e_monotono_na_ordem_dos_p(self):
        """Um p menor nunca pode receber um q maior."""
        p = np.array([0.001, 0.004, 0.006, 0.11, 0.53, 0.86])
        q = benjamini_hochberg(p)
        assert np.all(np.diff(q) >= -1e-12)

    def test_nan_do_baseline_e_preservado(self):
        """O baseline não é testado contra si mesmo e não entra na contagem."""
        p = np.array([0.01, np.nan, 0.02])
        q = benjamini_hochberg(p)
        assert np.isnan(q[1])
        assert np.isfinite(q[0]) and np.isfinite(q[2])
        # m = 2 (o NaN não conta): q do menor é 0.01 * 2 / 1 = 0.02
        assert q[0] == pytest.approx(0.02)

    def test_ruido_puro_nao_gera_descoberta(self):
        """21 p-valores uniformes: a 5% bruto ~1 passaria; com BH, nenhum."""
        rng = np.random.default_rng(4)
        p = rng.uniform(0, 1, 21)
        assert (benjamini_hochberg(p) < 0.05).sum() == 0

    def test_q_limitado_a_um(self):
        assert np.all(benjamini_hochberg(np.array([0.9, 0.95, 0.99])) <= 1.0)

    def test_lista_toda_nan_nao_quebra(self):
        q = benjamini_hochberg(np.array([np.nan, np.nan]))
        assert np.isnan(q).all()


# ------------------------------------------------------ incerteza
class TestIncerteza:
    """Garante que o intervalo e a curva de poder não mintam por construção."""

    def _par(self, n, ruido_a, ruido_b, seed=0):
        rng = np.random.default_rng(seed)
        y = rng.normal(100, 10, n)
        return y, y + rng.normal(0, ruido_a, n), y + rng.normal(0, ruido_b, n)

    def test_modelo_melhor_produz_intervalo_acima_de_zero(self):
        y, a, b = self._par(80, 0.5, 8.0)
        r = intervalo_bootstrap(y, a, b, horizon=1, n_amostras=800)
        assert r["ganho_pct"] > 0
        assert r["ganho_pct_lo"] > 0, "vantagem gritante não pode incluir zero"
        assert r["inclui_zero"] is False

    def test_modelos_equivalentes_produzem_intervalo_que_cruza_zero(self):
        y, a, b = self._par(80, 3.0, 3.0, seed=7)
        r = intervalo_bootstrap(y, a, b, horizon=1, n_amostras=800)
        assert r["inclui_zero"] is True

    def test_intervalo_contem_o_ponto_estimado(self):
        y, a, b = self._par(60, 2.0, 3.0, seed=3)
        r = intervalo_bootstrap(y, a, b, horizon=3, n_amostras=800)
        assert r["ganho_pct_lo"] <= r["ganho_pct"] <= r["ganho_pct_hi"]
        assert r["ganho_mae_lo"] <= r["ganho_mae"] <= r["ganho_mae_hi"]

    def test_horizonte_maior_alarga_o_intervalo(self):
        """Blocos maiores reconhecem a dependência e devolvem menos certeza."""
        y, a, b = self._par(96, 2.0, 3.0, seed=11)
        estreito = intervalo_bootstrap(y, a, b, horizon=1, n_amostras=1500)
        largo = intervalo_bootstrap(y, a, b, horizon=12, n_amostras=1500)
        larg_1 = estreito["ganho_pct_hi"] - estreito["ganho_pct_lo"]
        larg_12 = largo["ganho_pct_hi"] - largo["ganho_pct_lo"]
        assert larg_12 > larg_1

    def test_poder_cresce_com_o_tamanho_do_efeito(self):
        y, a, b = self._par(60, 3.0, 3.0, seed=5)
        r = efeito_minimo_detectavel(y, a, b, horizon=1, ganhos=[0.0, 0.25, 0.5],
                                     n_simulacoes=120)
        curva = r["curva_poder"]
        assert curva[0.0] <= curva[25.0] <= curva[50.0]

    def test_poder_reporta_falso_positivo_em_vantagem_zero(self):
        """Sem esse número, um teste mal calibrado passaria por poder alto."""
        y, a, b = self._par(60, 3.0, 3.0, seed=9)
        r = efeito_minimo_detectavel(y, a, b, horizon=1, ganhos=[0.0, 0.4],
                                     n_simulacoes=150)
        assert 0.0 <= r["taxa_falso_positivo"] <= 1.0
        assert r["taxa_falso_positivo"] == r["curva_poder"][0.0]
        assert isinstance(r["bem_calibrado"], bool)

    def test_amostra_minuscula_falha_alto(self):
        with pytest.raises(ValueError, match="insuficiente"):
            intervalo_bootstrap([1.0, 2.0], [1.0, 2.0], [1.0, 3.0], horizon=1)


# ------------------------------------------------------ estabilidade
class TestEstabilidade:
    """O ganho medido numa janela sobrevive à troca da janela?

    Este grupo trava o achado mais forte do projeto: em h=12 o ganho aparente
    de +11,6% vira −32,5% ao mudar o período avaliado. Sem esta verificação, um
    resultado que era característica de três anos específicos passaria por
    propriedade do modelo.
    """

    def test_troca_de_sinal_e_o_veredito_mais_grave(self):
        vd = veredito_estabilidade({36: +11.6, 72: -7.5, 144: -32.5, 273: -19.1})
        assert vd["troca_sinal"] is True
        assert vd["estavel"] is False
        assert "NÃO REPRODUZÍVEL" in vd["veredito"]

    def test_ganho_consistente_e_declarado_estavel(self):
        vd = veredito_estabilidade({36: 40.9, 72: 41.2, 144: 41.5, 284: 32.9})
        assert vd["troca_sinal"] is False
        assert vd["estavel"] is True
        assert "ESTÁVEL" in vd["veredito"]

    def test_mesmo_sinal_mas_amplitude_grande_e_instavel(self):
        """Não precisa trocar de sinal para não ser confiável."""
        vd = veredito_estabilidade({36: 5.0, 144: 45.0})
        assert vd["troca_sinal"] is False
        assert vd["estavel"] is False
        assert "INSTÁVEL" in vd["veredito"]

    def test_amplitude_e_extremos_sao_calculados(self):
        vd = veredito_estabilidade({36: -19.1, 72: 11.6})
        assert vd["ganho_min"] == pytest.approx(-19.1)
        assert vd["ganho_max"] == pytest.approx(11.6)
        assert vd["amplitude"] == pytest.approx(30.7)

    def test_uma_janela_so_nao_permite_julgar(self):
        with pytest.raises(ValueError, match="ao menos 2"):
            veredito_estabilidade({36: 11.6})

    def test_todos_negativos_nao_conta_como_troca_de_sinal(self):
        vd = veredito_estabilidade({36: -5.0, 144: -8.0})
        assert vd["troca_sinal"] is False
        assert vd["estavel"] is True


# ------------------------------------------------------ modelos
class TestModels:
    def test_seasonal_naive_returns_lag_12(self, toy_series):
        X, y = make_supervised_frame(toy_series, horizon=12)
        m = SeasonalNaive(season=12).fit(X, y)
        np.testing.assert_array_equal(m.predict(X), X["lag_12"].to_numpy())

    def test_registry_models_are_fittable(self, toy_series):
        X, y = make_supervised_frame(toy_series, horizon=12)
        for name, model in build_model_registry().items():
            model.fit(X, y)
            pred = model.predict(X)
            assert len(pred) == len(y), f"{name} devolveu tamanho errado"
            assert np.isfinite(pred).all(), f"{name} produziu valor não finito"


# ------------------------------------------------------ EMBARGO
class TestEmbargo:
    """Regressão do bug de validação encontrado na auditoria.

    O protocolo original treinava com todos os alvos anteriores ao mês
    previsto. Em previsão direta de horizonte h, os h-1 meses antes do alvo
    ainda não foram observados na origem da previsão. O efeito medido nesta
    série foi de 2,64 p.p. de MAPE — o bastante para inverter qual modelo
    vence. Estes testes impedem o retorno do bug.
    """

    def test_embargo_stops_at_forecast_origin(self):
        idx = pd.date_range("2010-01-01", "2015-01-01", freq="MS")
        cutoff = pd.Timestamp("2015-01-01")
        mask = train_mask_for(idx, cutoff, horizon=12)
        assert idx[mask].max() == pd.Timestamp("2014-01-01")

    def test_embargo_gap_equals_horizon_minus_one(self):
        """Exatamente h-1 linhas a menos que o protocolo defeituoso."""
        idx = pd.date_range("2000-01-01", "2015-01-01", freq="MS")
        cutoff = pd.Timestamp("2015-01-01")
        for horizon in (1, 3, 6, 12, 24):
            clean = train_mask_for(idx, cutoff, horizon, embargo=True).sum()
            leaky = train_mask_for(idx, cutoff, horizon, embargo=False).sum()
            assert leaky - clean == horizon - 1, f"h={horizon}"

    def test_embargo_is_noop_at_horizon_1(self):
        """Em h=1 a origem é o mês anterior: os dois protocolos coincidem."""
        idx = pd.date_range("2000-01-01", "2015-01-01", freq="MS")
        cutoff = pd.Timestamp("2015-01-01")
        np.testing.assert_array_equal(
            train_mask_for(idx, cutoff, 1, embargo=True),
            train_mask_for(idx, cutoff, 1, embargo=False),
        )

    def test_embargo_is_default(self):
        """O padrão da configuração tem de ser o protocolo correto."""
        assert BacktestConfig().embargo is True

    def test_no_training_target_after_origin(self):
        """Invariante geral: nenhum alvo de treino excede a origem."""
        idx = pd.date_range("1990-01-01", "2017-08-01", freq="MS")
        for horizon in (1, 3, 12):
            for cutoff in idx[-6:]:
                mask = train_mask_for(idx, cutoff, horizon)
                origin = cutoff - pd.DateOffset(months=horizon)
                assert idx[mask].max() <= origin


class TestFronteiraDasExogenas:
    """A flag `use_exogenous` precisa ligar e desligar exatamente um grupo.

    O risco aqui não é o código quebrar — é o relatório somar a importância de
    um grupo diferente do que a matriz contém, e ninguém perceber.
    """

    def _serie(self):
        idx = pd.date_range("1990-01-01", "2017-08-01", freq="MS")
        rng = np.random.default_rng(0)
        vals = 100 + 15 * np.sin(2 * np.pi * idx.month / 12) + rng.normal(0, 2, len(idx))
        return pd.Series(vals, index=idx, name="production_index")

    def test_desligar_remove_exatamente_as_oito_declaradas(self):
        y = self._serie()
        com, _ = make_supervised_frame(y, horizon=1, use_exogenous=True)
        sem, _ = make_supervised_frame(y, horizon=1, use_exogenous=False)
        removidas = set(com.columns) - set(sem.columns)
        assert removidas == set(EXOGENOUS_COLS)
        assert len(EXOGENOUS_COLS) == 8

    def test_sazonalidade_deterministica_permanece(self):
        """`is_holiday_peak_season` é mês ∈ {9..12}, não informação externa.

        Ela fica junto de `quarter` e dos termos de Fourier: são a sazonalidade
        escrita de outro jeito. Removê-las não seria desligar exógenas — seria
        tirar do modelo a noção de que dezembro difere de abril.
        """
        y = self._serie()
        sem, _ = make_supervised_frame(y, horizon=1, use_exogenous=False)
        for col in ("is_holiday_peak_season", "quarter", "time_trend",
                    "fourier_sin_1", "fourier_cos_1"):
            assert col in sem.columns, f"{col} sumiu ao desligar as exógenas"

    def test_nenhuma_exogena_e_funcao_pura_do_mes(self):
        """Critério objetivo da fronteira, não julgamento de nome.

        Uma coluna que só depende do número do mês é sazonalidade. Se alguma
        entrar em EXOGENOUS_COLS, a lista está classificando errado — e é assim
        que `is_holiday_peak_season` seria removida por engano.
        """
        cal = make_calendar_features(pd.date_range("1990-01-01", "2017-12-01", freq="MS"))
        for col in EXOGENOUS_COLS:
            por_mes = cal.groupby(cal.index.month)[col].nunique()
            assert (por_mes > 1).any(), (
                f"{col} é constante dentro de cada mês: é sazonalidade "
                f"determinística, não informação externa"
            )


class TestArtefatosJson:
    """Todo artefato JSON do pipeline tem de ser legível por parser estrito.

    A arquitetura de produção proposta consome estes arquivos no Spark. Um
    JSON que abre no editor e quebra no Spark é o pior tipo de defeito: só
    aparece do outro lado da entrega.
    """

    def test_nan_e_infinito_viram_nulo(self):
        """``NaN`` é a resposta certa em alguns p-valores; o token não é JSON.

        A tradução é ``null`` — o jeito que JSON tem de dizer "sem valor" —
        e é o que Parquet e Spark leem como nulo do outro lado.
        """
        assert sanitizar_json(float("nan")) is None
        assert sanitizar_json(float("inf")) is None
        assert sanitizar_json(float("-inf")) is None
        assert sanitizar_json(np.float64("nan")) is None
        assert sanitizar_json(3.5) == 3.5

    def test_tipos_do_numpy_viram_tipos_do_python(self):
        """``np.float64`` e companhia não são serializáveis por ``json``.

        Como quase todo número deste pipeline vem do numpy, sem esta conversão
        cada campo novo é um ``TypeError`` descoberto no fim de uma rodada de
        sete minutos.
        """
        limpo = sanitizar_json({
            "f": np.float64(1.5), "i": np.int64(7), "b": np.bool_(True),
            "arr": np.array([1.0, np.nan]),
        })
        assert limpo == {"f": 1.5, "i": 7, "b": True, "arr": [1.0, None]}
        assert isinstance(limpo["i"], int)
        assert isinstance(limpo["b"], bool)

    def test_estruturas_aninhadas_sao_percorridas(self):
        """O NaN costuma estar fundo na estrutura, não no topo."""
        limpo = sanitizar_json(
            {"por_horizonte": {12: {"janelas": [{"p": float("nan")}]}}}
        )
        # Chave numérica vira string: JSON não admite outra coisa.
        assert limpo == {"por_horizonte": {"12": {"janelas": [{"p": None}]}}}

    def test_arquivo_gravado_e_json_estrito(self, tmp_path):
        """A garantia de ponta: o que vai para o disco passa em ``strict``."""
        destino = salvar_json({"p": float("nan"), "ok": 1.0},
                              tmp_path / "sub" / "a.json")
        texto = destino.read_text(encoding="utf-8")
        assert "NaN" not in texto
        # parse_constant dispara exatamente nos tokens que não são JSON.
        def _proibido(token):  # pragma: no cover - só roda se houver regressão
            raise AssertionError(f"token não-JSON no arquivo: {token}")

        assert json.loads(texto, parse_constant=_proibido) == {"p": None, "ok": 1.0}

    def test_texto_impresso_e_o_mesmo_do_arquivo(self, tmp_path):
        """Terminal e disco não podem divergir — mesma sanitização nos dois."""
        dados = {"p": float("nan"), "n": np.int64(3)}
        destino = salvar_json(dados, tmp_path / "a.json")
        assert formatar_json(dados) == destino.read_text(encoding="utf-8")
