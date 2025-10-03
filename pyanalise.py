"""
Módulo de análises avançadas para NTN-B e DI Futuro.

Funções para cálculo de métricas avançadas:
- Breakeven Inflation
- Steepness (inclinação da curva)
- Estatísticas históricas (High/Low)
- Médias móveis

Autor: Claude Code
Data: 2025-10-02
"""

import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


logger = logging.getLogger(__name__)


def calcular_breakeven_inflation(
    df_ntnb: pd.DataFrame,
    df_di: pd.DataFrame,
    pares_vencimento: List[Dict[str, str]]
) -> pd.DataFrame:
    """
    Calcula a inflação implícita (Breakeven Inflation) comparando NTN-B e DI.

    BEI = [(1 + Taxa_DI_nominal) / (1 + Yield_NTNB_real)] - 1

    Args:
        df_ntnb: DataFrame com yields NTN-B (colunas: Data, Vencimento, BidRate)
        df_di: DataFrame com taxas DI (colunas: Data, Ticker, Taxa, Vencimento)
        pares_vencimento: Lista de dicionários {ntnb: "2030-08-15", di: "DI1F30"}

    Returns:
        DataFrame com colunas: Data, Vencimento, NTNB_Real, DI_Nominal, BEI
    """
    logger.info("Calculando Breakeven Inflation...")

    if len(df_ntnb) == 0 or len(df_di) == 0:
        logger.warning("DataFrames vazios, retornando tabela vazia")
        return pd.DataFrame()

    resultados = []

    for par in pares_vencimento:
        venc_ntnb = par.get('ntnb')
        ticker_di = par.get('di')

        if not venc_ntnb or not ticker_di:
            logger.warning(f"Par inválido: {par}")
            continue

        # Filtra dados NTN-B
        df_ntnb_venc = df_ntnb[df_ntnb['Vencimento'] == venc_ntnb].copy()

        # Filtra dados DI
        df_di_ticker = df_di[df_di['Ticker'] == ticker_di].copy()

        if len(df_ntnb_venc) == 0:
            logger.warning(f"Sem dados NTN-B para vencimento {venc_ntnb}")
            continue

        if len(df_di_ticker) == 0:
            logger.warning(f"Sem dados DI para ticker {ticker_di}")
            continue

        # Merge por data
        df_merge = pd.merge(
            df_ntnb_venc[['Data', 'BidRate']],
            df_di_ticker[['Data', 'Taxa']],
            on='Data',
            how='inner'
        )

        if len(df_merge) == 0:
            logger.warning(f"Sem datas em comum entre {venc_ntnb} e {ticker_di}")
            continue

        # Calcula BEI
        # BidRate da NTN-B vem em decimal (ex: 0.065 = 6.5%)
        # Taxa DI vem em % (ex: 13.47)
        df_merge['NTNB_Real'] = df_merge['BidRate'] * 100  # Converte para %
        df_merge['DI_Nominal'] = df_merge['Taxa']

        # Fórmula BEI: [(1 + DI/100) / (1 + NTNB/100)] - 1
        df_merge['BEI'] = (
            ((1 + df_merge['DI_Nominal'] / 100) / (1 + df_merge['NTNB_Real'] / 100)) - 1
        ) * 100  # Resultado em %

        df_merge['Vencimento'] = venc_ntnb
        df_merge['Ticker_DI'] = ticker_di

        resultados.append(df_merge[['Data', 'Vencimento', 'Ticker_DI', 'NTNB_Real', 'DI_Nominal', 'BEI']])

    if len(resultados) == 0:
        logger.warning("Nenhum resultado BEI calculado")
        return pd.DataFrame()

    df_result = pd.concat(resultados, ignore_index=True)
    df_result = df_result.sort_values(['Vencimento', 'Data'])

    logger.info(f"BEI calculado para {len(pares_vencimento)} pares, {len(df_result)} registros")

    return df_result


def calcular_steepness(
    df_historico: pd.DataFrame,
    ticker_curto: str,
    ticker_longo: str,
    coluna_taxa: str = 'Taxa'
) -> pd.DataFrame:
    """
    Calcula steepness (inclinação da curva) entre dois vencimentos.

    Steepness = Taxa_Longo - Taxa_Curto

    Args:
        df_historico: DataFrame com histórico (colunas: Data, Ticker, Taxa/BidRate)
        ticker_curto: Ticker do vencimento curto (ex: "DI1F27")
        ticker_longo: Ticker do vencimento longo (ex: "DI1F35")
        coluna_taxa: Nome da coluna de taxa ('Taxa' para DI, 'BidRate' para NTN-B)

    Returns:
        DataFrame com colunas: Data, Ticker_Curto, Ticker_Longo, Taxa_Curto, Taxa_Longo, Steepness
    """
    logger.info(f"Calculando steepness: {ticker_longo} - {ticker_curto}")

    if len(df_historico) == 0:
        logger.warning("DataFrame vazio, retornando tabela vazia")
        return pd.DataFrame()

    # Identifica coluna de ticker
    coluna_ticker = 'Ticker' if 'Ticker' in df_historico.columns else 'Vencimento'

    # Filtra dados
    df_curto = df_historico[df_historico[coluna_ticker] == ticker_curto][['Data', coluna_taxa]].copy()
    df_longo = df_historico[df_historico[coluna_ticker] == ticker_longo][['Data', coluna_taxa]].copy()

    if len(df_curto) == 0:
        logger.warning(f"Sem dados para {ticker_curto}")
        return pd.DataFrame()

    if len(df_longo) == 0:
        logger.warning(f"Sem dados para {ticker_longo}")
        return pd.DataFrame()

    # Merge por data
    df_merge = pd.merge(
        df_curto,
        df_longo,
        on='Data',
        how='inner',
        suffixes=('_Curto', '_Longo')
    )

    if len(df_merge) == 0:
        logger.warning(f"Sem datas em comum entre {ticker_curto} e {ticker_longo}")
        return pd.DataFrame()

    # Calcula steepness
    taxa_curto_col = f'{coluna_taxa}_Curto'
    taxa_longo_col = f'{coluna_taxa}_Longo'

    # Se BidRate, converte para %
    if coluna_taxa == 'BidRate':
        df_merge[taxa_curto_col] = df_merge[taxa_curto_col] * 100
        df_merge[taxa_longo_col] = df_merge[taxa_longo_col] * 100

    df_merge['Steepness'] = df_merge[taxa_longo_col] - df_merge[taxa_curto_col]

    df_merge['Ticker_Curto'] = ticker_curto
    df_merge['Ticker_Longo'] = ticker_longo

    df_merge = df_merge.rename(columns={
        taxa_curto_col: 'Taxa_Curto',
        taxa_longo_col: 'Taxa_Longo'
    })

    df_result = df_merge[['Data', 'Ticker_Curto', 'Ticker_Longo', 'Taxa_Curto', 'Taxa_Longo', 'Steepness']]
    df_result = df_result.sort_values('Data')

    logger.info(f"Steepness calculado: {len(df_result)} registros")

    return df_result


def calcular_estatisticas_historicas(
    df_historico: pd.DataFrame,
    periodos: List[int] = [30, 90, 365],
    coluna_taxa: str = 'Taxa'
) -> pd.DataFrame:
    """
    Calcula estatísticas históricas (High, Low, Média) para cada ticker/vencimento.

    Args:
        df_historico: DataFrame com histórico (colunas: Data, Ticker/Vencimento, Taxa/BidRate)
        periodos: Lista de períodos em dias (ex: [30, 90, 365])
        coluna_taxa: Nome da coluna de taxa

    Returns:
        DataFrame com colunas: Ticker, Taxa_Atual, High_Xd, Low_Xd, Media_Xd, Posicao_Percentil_Xd
    """
    logger.info(f"Calculando estatísticas históricas para períodos: {periodos}")

    if len(df_historico) == 0:
        logger.warning("DataFrame vazio, retornando tabela vazia")
        return pd.DataFrame()

    # Identifica coluna de ticker
    coluna_ticker = 'Ticker' if 'Ticker' in df_historico.columns else 'Vencimento'

    # Data mais recente
    data_max = df_historico['Data'].max()

    resultados = []

    tickers = df_historico[coluna_ticker].unique()

    for ticker in tickers:
        df_ticker = df_historico[df_historico[coluna_ticker] == ticker].copy()
        df_ticker = df_ticker.sort_values('Data')

        # Taxa atual
        taxa_atual = df_ticker[df_ticker['Data'] == data_max][coluna_taxa].values
        if len(taxa_atual) == 0:
            logger.warning(f"Sem dados atuais para {ticker}")
            continue

        taxa_atual = taxa_atual[0]

        # Se BidRate, converte para %
        if coluna_taxa == 'BidRate':
            taxa_atual = taxa_atual * 100

        stats = {coluna_ticker: ticker, 'Taxa_Atual': taxa_atual}

        # Calcula stats para cada período
        for dias in periodos:
            data_min = data_max - timedelta(days=dias)
            df_periodo = df_ticker[df_ticker['Data'] >= data_min].copy()

            if len(df_periodo) == 0:
                stats[f'High_{dias}d'] = np.nan
                stats[f'Low_{dias}d'] = np.nan
                stats[f'Media_{dias}d'] = np.nan
                stats[f'Percentil_{dias}d'] = np.nan
                continue

            taxas = df_periodo[coluna_taxa].values

            # Se BidRate, converte para %
            if coluna_taxa == 'BidRate':
                taxas = taxas * 100

            high = np.max(taxas)
            low = np.min(taxas)
            media = np.mean(taxas)

            # Percentil: onde a taxa atual está no range
            # 0% = na mínima, 100% = na máxima
            if high > low:
                percentil = ((taxa_atual - low) / (high - low)) * 100
            else:
                percentil = 50.0

            stats[f'High_{dias}d'] = high
            stats[f'Low_{dias}d'] = low
            stats[f'Media_{dias}d'] = media
            stats[f'Percentil_{dias}d'] = percentil

        resultados.append(stats)

    if len(resultados) == 0:
        logger.warning("Nenhuma estatística calculada")
        return pd.DataFrame()

    df_result = pd.DataFrame(resultados)

    logger.info(f"Estatísticas calculadas para {len(tickers)} tickers")

    return df_result


def calcular_medias_moveis(
    df_historico: pd.DataFrame,
    periodos: List[int] = [20, 50, 200],
    coluna_taxa: str = 'Taxa'
) -> pd.DataFrame:
    """
    Calcula médias móveis para cada ticker/vencimento.

    Args:
        df_historico: DataFrame com histórico (colunas: Data, Ticker/Vencimento, Taxa/BidRate)
        periodos: Lista de períodos para MA (ex: [20, 50, 200])
        coluna_taxa: Nome da coluna de taxa

    Returns:
        DataFrame com colunas: Data, Ticker, Taxa, MA20, MA50, MA200
    """
    logger.info(f"Calculando médias móveis: {periodos}")

    if len(df_historico) == 0:
        logger.warning("DataFrame vazio, retornando tabela vazia")
        return pd.DataFrame()

    # Identifica coluna de ticker
    coluna_ticker = 'Ticker' if 'Ticker' in df_historico.columns else 'Vencimento'

    resultados = []

    tickers = df_historico[coluna_ticker].unique()

    for ticker in tickers:
        df_ticker = df_historico[df_historico[coluna_ticker] == ticker].copy()
        df_ticker = df_ticker.sort_values('Data')

        # Cria cópia da taxa
        taxa_col = df_ticker[coluna_taxa].copy()

        # Se BidRate, converte para %
        if coluna_taxa == 'BidRate':
            taxa_col = taxa_col * 100

        df_ticker['Taxa_Calc'] = taxa_col

        # Calcula MAs
        for periodo in periodos:
            df_ticker[f'MA{periodo}'] = df_ticker['Taxa_Calc'].rolling(window=periodo, min_periods=1).mean()

        df_ticker[coluna_ticker] = ticker

        colunas_resultado = ['Data', coluna_ticker, 'Taxa_Calc'] + [f'MA{p}' for p in periodos]
        resultados.append(df_ticker[colunas_resultado])

    if len(resultados) == 0:
        logger.warning("Nenhuma média móvel calculada")
        return pd.DataFrame()

    df_result = pd.concat(resultados, ignore_index=True)
    df_result = df_result.sort_values(['Data', coluna_ticker])

    logger.info(f"Médias móveis calculadas para {len(tickers)} tickers")

    return df_result


if __name__ == "__main__":
    # Testes do módulo
    print("=" * 80)
    print("TESTE: Modulo pyanalise.py")
    print("=" * 80)

    # Teste básico com dados fictícios
    import pydifuturo

    print("\n1. Testando cálculo de Steepness...")
    df_di = pydifuturo.carregar_historico()
    if len(df_di) > 0:
        df_steep = calcular_steepness(df_di, 'DI1F27', 'DI1F35', coluna_taxa='Taxa')
        print(f"   Steepness calculado: {len(df_steep)} registros")
        if len(df_steep) > 0:
            print(f"   Steepness atual: {df_steep['Steepness'].iloc[-1]:.2f} bps")

    print("\n2. Testando estatísticas históricas...")
    if len(df_di) > 0:
        df_stats = calcular_estatisticas_historicas(df_di, periodos=[30, 90])
        print(f"   Estatísticas calculadas para {len(df_stats)} tickers")
        if len(df_stats) > 0:
            print(df_stats[['Ticker', 'Taxa_Atual', 'High_30d', 'Low_30d']].head())

    print("\n" + "=" * 80)
    print("Testes concluídos!")
