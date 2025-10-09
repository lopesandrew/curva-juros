"""
Módulo para coleta de dados de Treasuries US via FRED API
"""

from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TreasuryData:
    """Classe para buscar dados de Treasuries do FRED"""

    NOMINAL_SERIES = {
        "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO",
        "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3",
        "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10",
        "20Y": "DGS20", "30Y": "DGS30"
    }

    TIPS_SERIES = {
        "5Y": "DFII5", "7Y": "DFII7", "10Y": "DFII10",
        "20Y": "DFII20", "30Y": "DFII30"
    }

    def __init__(self, api_key: str):
        """Inicializa cliente FRED

        Args:
            api_key: FRED API key
        """
        self.fred = Fred(api_key=api_key)

    def fetch_treasuries(self, data_ref=None, lookback_days=365, salvar_csv=True):
        """Baixa dados de Treasuries do FRED

        Args:
            data_ref: Data de referência (default: hoje)
            lookback_days: Dias históricos para baixar (default: 365)
            salvar_csv: Se True, salva dados em CSV (default: True)

        Returns:
            DataFrame com taxas nominais, TIPS, slopes e breakevens
        """
        data_fim = data_ref or datetime.now()
        data_inicio = data_fim - timedelta(days=lookback_days)

        logger.info(f"Buscando Treasuries de {data_inicio.date()} até {data_fim.date()}")

        # Baixa nominais
        df_nom = self._fetch_series_dict(self.NOMINAL_SERIES, data_inicio, data_fim, "nominal")

        # Baixa TIPS
        df_tips = self._fetch_series_dict(self.TIPS_SERIES, data_inicio, data_fim, "TIPS")
        df_tips.columns = [f"{col}_TIPS" for col in df_tips.columns]

        # Combina
        df = df_nom.join(df_tips, how='outer')

        # Calcula slopes
        if "2Y" in df and "10Y" in df:
            df["2s10s_bp"] = (df["10Y"] - df["2Y"]) * 100
        if "5Y" in df and "30Y" in df:
            df["5s30s_bp"] = (df["30Y"] - df["5Y"]) * 100

        # Calcula breakevens
        for tenor in ["5Y", "7Y", "10Y", "20Y", "30Y"]:
            if tenor in df and f"{tenor}_TIPS" in df:
                df[f"BE_{tenor}"] = df[tenor] - df[f"{tenor}_TIPS"]

        df.index.name = "date"
        df = df.dropna(how='all')

        # Salva CSV
        if salvar_csv:
            arquivo_csv = f"dados/ust_{data_fim.strftime('%Y%m%d')}.csv"
            df.to_csv(arquivo_csv)
            logger.info(f"Dados salvos em {arquivo_csv}")

        return df

    def _fetch_series_dict(self, series_dict, data_inicio, data_fim, tipo):
        """Helper para buscar múltiplas séries

        Args:
            series_dict: Dict com label: series_id
            data_inicio: Data inicial
            data_fim: Data final
            tipo: Tipo de série (para log)

        Returns:
            DataFrame com as séries
        """
        df = pd.DataFrame()

        for label, series_id in series_dict.items():
            try:
                df[label] = self.fred.get_series(series_id, data_inicio, data_fim)
                logger.debug(f"Série {tipo} {label} baixada com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao baixar {tipo} {label} ({series_id}): {e}")

        return df

    def get_latest_data(self):
        """Retorna último dado disponível de cada série

        Returns:
            Series com últimas taxas disponíveis
        """
        df = self.fetch_treasuries(lookback_days=30, salvar_csv=False)
        return df.iloc[-1]

    def calcular_variacoes(self, df, periodos=[1, 5, 21]):
        """Calcula variações das taxas

        Args:
            df: DataFrame com histórico de taxas
            periodos: Lista de períodos em dias úteis (default: [1, 5, 21])

        Returns:
            DataFrame com variações
        """
        variacoes = {}

        colunas_taxas = [col for col in df.columns if not col.endswith('_bp') and not col.startswith('BE_')]

        for periodo in periodos:
            label = f"{periodo}D" if periodo == 1 else f"{periodo}D"
            if periodo == 5:
                label = "1W"
            elif periodo == 21:
                label = "1M"

            for col in colunas_taxas:
                variacoes[f"{col}_{label}"] = df[col] - df[col].shift(periodo)

        df_var = pd.DataFrame(variacoes, index=df.index)
        return df_var


def buscar_treasuries(config):
    """Função principal para buscar dados de Treasuries

    Args:
        config: Dict com configurações (deve conter 'treasuries.fred_api_key')

    Returns:
        DataFrame com dados de Treasuries
    """
    ust_cfg = config.get('treasuries', {})
    api_key = ust_cfg.get('fred_api_key')

    if not api_key:
        logger.error("FRED API key não encontrada no config")
        return None

    lookback_days = ust_cfg.get('lookback_days', 365)

    treasury = TreasuryData(api_key)
    df = treasury.fetch_treasuries(lookback_days=lookback_days)

    return df
