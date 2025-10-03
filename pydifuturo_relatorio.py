"""
Módulo para gerar relatórios e gráficos de DI Futuro.

Este módulo complementa pydifuturo.py com funções de visualização
e integração ao relatório HTML de NTN-B.

Autor: Claude Code
Data: 2025-10-02
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import pydifuturo


logger = logging.getLogger(__name__)


def calcular_variacoes_di(
    df_historico: pd.DataFrame,
    deslocamentos: List[int] = [1, 7, 30]
) -> pd.DataFrame:
    """
    Calcula tabela de variações de DI Futuro similar à de NTN-B.

    Args:
        df_historico: DataFrame com histórico completo (colunas: Data, Ticker, Taxa)
        deslocamentos: Lista de períodos para calcular variações (em dias)

    Returns:
        DataFrame com colunas: Vencimento, Taxa Atual, Taxa D-X, Var D-X (bps), ...
    """
    if len(df_historico) == 0:
        logger.warning("Histórico DI vazio, retornando tabela vazia")
        return pd.DataFrame()

    # Data mais recente
    data_atual = df_historico['Data'].max()
    df_atual = df_historico[df_historico['Data'] == data_atual].copy()

    if len(df_atual) == 0:
        logger.warning(f"Nenhum dado para data {data_atual}")
        return pd.DataFrame()

    logger.info(f"Calculando variações DI para {len(df_atual)} contratos na data {data_atual}")

    # Ordena por vencimento
    df_atual = df_atual.sort_values('Vencimento').reset_index(drop=True)

    # Inicia tabela de variações
    df_var = df_atual[['Ticker', 'Vencimento', 'Taxa']].copy()
    df_var = df_var.rename(columns={'Taxa': 'Taxa Atual'})

    # Para cada deslocamento
    for dias in deslocamentos:
        data_alvo = data_atual - timedelta(days=dias)

        # Busca dados do período
        df_periodo = df_historico[df_historico['Data'] <= data_alvo]

        if len(df_periodo) == 0:
            logger.warning(f"Sem dados para D-{dias} (data alvo: {data_alvo})")
            continue

        # Pega data mais recente disponível no período
        data_encontrada = df_periodo['Data'].max()
        df_passado = df_periodo[df_periodo['Data'] == data_encontrada].copy()

        # Merge com dados atuais
        df_passado = df_passado[['Ticker', 'Taxa']].rename(columns={'Taxa': f'Taxa D-{dias}'})
        df_var = df_var.merge(df_passado, on='Ticker', how='left')

        # Calcula variação em bps
        if f'Taxa D-{dias}' in df_var.columns:
            df_var[f'Var D-{dias} (bps)'] = (df_var['Taxa Atual'] - df_var[f'Taxa D-{dias}']) * 100

    return df_var


def preparar_tabela_html_di(df_var: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara tabela de variações DI para exibição no HTML.

    Formata datas, arredonda valores e organiza colunas.

    Args:
        df_var: DataFrame com variações calculadas

    Returns:
        DataFrame formatado para HTML
    """
    if len(df_var) == 0:
        return pd.DataFrame()

    df_html = df_var.copy()

    # Formata vencimento
    df_html['Vencimento'] = pd.to_datetime(df_html['Vencimento']).dt.strftime('%d/%m/%Y')

    # Formata taxas e variações
    for col in df_html.columns:
        if 'Taxa' in col and col != 'Vencimento':
            # Formata taxa como "XX,XX%" com 2 casas decimais
            df_html[col] = df_html[col].apply(lambda x: f"{x:.2f}%".replace('.', ',') if pd.notna(x) else '-')
        elif 'Var' in col and 'bps' in col:
            # Formata variação em bps com 2 casas decimais
            df_html[col] = df_html[col].apply(lambda x: f"{x:.2f}".replace('.', ',') if pd.notna(x) else '-')

    # Reorganiza colunas
    colunas_ordenadas = ['Vencimento', 'Taxa Atual']

    # Adiciona pares Taxa D-X / Var D-X na ordem
    for col in df_html.columns:
        if col.startswith('Taxa D-'):
            dias = col.replace('Taxa D-', '')
            colunas_ordenadas.append(col)
            col_var = f'Var D-{dias} (bps)'
            if col_var in df_html.columns:
                colunas_ordenadas.append(col_var)

    # Remove Ticker (não precisa exibir no HTML)
    if 'Ticker' in df_html.columns:
        df_html = df_html.drop(columns=['Ticker'])

    # Filtra colunas existentes
    colunas_final = [c for c in colunas_ordenadas if c in df_html.columns]
    df_html = df_html[colunas_final]

    return df_html


def gerar_grafico_curva_di(
    df_historico: pd.DataFrame,
    data_referencia: datetime.date,
    deslocamentos: List[int] = [1, 7, 30],
    figsize: Tuple[int, int] = (11, 7),
    dpi: int = 150
) -> plt.Figure:
    """
    Gera gráfico da curva de DI Futuro com comparação histórica.

    Args:
        df_historico: DataFrame com histórico completo
        data_referencia: Data de referência (tipicamente hoje)
        deslocamentos: Períodos para comparação
        figsize: Tamanho da figura (largura, altura)
        dpi: Resolução do gráfico

    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Dados atuais
    df_atual = df_historico[df_historico['Data'] == data_referencia].copy()

    # Filtra apenas vencimentos desejados (27, 28, 29, 30, 32, 35)
    vencimentos_filtro = ['DI1F27', 'DI1F28', 'DI1F29', 'DI1F30', 'DI1F32', 'DI1F35']
    df_atual = df_atual[df_atual['Ticker'].isin(vencimentos_filtro)].copy()

    df_atual = df_atual.sort_values('Vencimento')

    if len(df_atual) == 0:
        logger.warning(f"Sem dados para data {data_referencia}")
        ax.text(0.5, 0.5, 'Sem dados disponíveis', ha='center', va='center')
        return fig

    # Plot curva atual (mais destacada)
    ax.plot(
        df_atual['Vencimento'],
        df_atual['Taxa'],
        marker='o',
        linewidth=2.5,
        markersize=7,
        label=f'Atual ({data_referencia.strftime("%d/%m/%Y")})',
        color='#1f77b4',
        zorder=10
    )

    # Adiciona anotação apenas no último ponto
    ultimo_ponto = df_atual.iloc[-1]
    ax.annotate(
        f"{ultimo_ponto['Taxa']:.2f}%",
        xy=(ultimo_ponto['Vencimento'], ultimo_ponto['Taxa']),
        xytext=(0, 8),
        textcoords="offset points",
        fontsize=8,
        ha="center",
        va="bottom",
        color='#1f77b4',
        fontweight='bold'
    )

    # Plot curvas históricas
    cores = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, dias in enumerate(deslocamentos):
        data_alvo = data_referencia - timedelta(days=dias)
        df_periodo = df_historico[df_historico['Data'] <= data_alvo]

        if len(df_periodo) > 0:
            data_encontrada = df_periodo['Data'].max()
            df_passado = df_periodo[df_periodo['Data'] == data_encontrada].copy()

            # Aplica o mesmo filtro de vencimentos
            df_passado = df_passado[df_passado['Ticker'].isin(vencimentos_filtro)].copy()

            df_passado = df_passado.sort_values('Vencimento')

            if len(df_passado) > 0:
                cor = cores[idx % len(cores)]
                ax.plot(
                    df_passado['Vencimento'],
                    df_passado['Taxa'],
                    marker='s',
                    linewidth=1.5,
                    markersize=5,
                    label=f'D-{dias} ({data_encontrada.strftime("%d/%m/%Y")})',
                    color=cor,
                    alpha=0.7,
                    zorder=5
                )

                # Adiciona anotação no último ponto desta curva
                ultimo_ponto_hist = df_passado.iloc[-1]
                ax.annotate(
                    f"{ultimo_ponto_hist['Taxa']:.2f}%",
                    xy=(ultimo_ponto_hist['Vencimento'], ultimo_ponto_hist['Taxa']),
                    xytext=(0, 8),
                    textcoords="offset points",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                    color=cor,
                    alpha=0.7
                )

    # Formatação
    ax.set_xlabel('Vencimento', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa (% a.a.)', fontsize=11, fontweight='bold')
    ax.set_title('Curva de Juros', fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Formata eixo X com datas
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Ajusta layout
    fig.tight_layout()

    return fig


def gerar_grafico_historico_di(
    df_historico: pd.DataFrame,
    tickers: List[str],
    dias_lookback: int,
    titulo: str = "Histórico de Taxas - DI Futuro",
    figsize: Tuple[int, int] = (14, 6),
    dpi: int = 150
) -> plt.Figure:
    """
    Gera gráfico histórico de taxas para contratos DI específicos.

    Args:
        df_historico: DataFrame com histórico completo
        tickers: Lista de tickers para plotar (ex: ['DI1F26', 'DI1F27'])
        dias_lookback: Número de dias para retroceder no gráfico
        titulo: Título do gráfico
        figsize: Tamanho da figura
        dpi: Resolução

    Returns:
        Figura matplotlib
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Filtra período
    data_max = df_historico['Data'].max()
    data_min = data_max - timedelta(days=dias_lookback)
    df_periodo = df_historico[df_historico['Data'] >= data_min].copy()

    if len(df_periodo) == 0:
        logger.warning(f"Sem dados para período de {dias_lookback} dias")
        ax.text(0.5, 0.5, 'Sem dados disponíveis', ha='center', va='center')
        return fig

    # Plot cada ticker
    cores = plt.cm.tab10(range(len(tickers)))

    for idx, ticker in enumerate(tickers):
        df_ticker = df_periodo[df_periodo['Ticker'] == ticker].copy()

        if len(df_ticker) == 0:
            logger.warning(f"Sem dados para ticker {ticker}")
            continue

        df_ticker = df_ticker.sort_values('Data')

        # Extrai ano de vencimento do ticker (ex: DI1F26 -> 2026)
        ano_venc = '20' + ticker[-2:] if len(ticker) >= 2 else ''
        label = f"{ticker} (Venc: Jan/{ano_venc})"

        ax.plot(
            df_ticker['Data'],
            df_ticker['Taxa'],
            marker='',
            linewidth=2,
            label=label,
            color=cores[idx],
            alpha=0.85
        )

        # Adiciona anotação com último valor
        if len(df_ticker) > 0:
            x_last = df_ticker['Data'].iloc[-1]
            y_last = df_ticker['Taxa'].iloc[-1]
            ax.annotate(
                f"{y_last:.2f}%",
                xy=(x_last, y_last),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=8,
                ha="left",
                va="center",
                color=cores[idx],
                fontweight='bold'
            )

    # Formatação
    ax.set_xlabel('Data', fontsize=11, fontweight='bold')
    ax.set_ylabel('Taxa (% a.a.)', fontsize=11, fontweight='bold')
    ax.set_title(titulo, fontsize=13, fontweight='bold', pad=15)
    ax.legend(loc='best', framealpha=0.95, fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Formata eixo X
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Ajusta layout
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Teste do módulo
    print("=" * 80)
    print("TESTE: Modulo pydifuturo_relatorio.py")
    print("=" * 80)

    # Carrega histórico
    df_hist = pydifuturo.carregar_historico()

    if len(df_hist) > 0:
        print(f"\nHistorico carregado: {len(df_hist)} registros")

        # Testa cálculo de variações
        df_var = calcular_variacoes_di(df_hist, deslocamentos=[1, 7, 30])
        print(f"\nTabela de variacoes:")
        print(df_var.to_string())

        # Testa preparação HTML
        df_html = preparar_tabela_html_di(df_var)
        print(f"\nTabela HTML formatada:")
        print(df_html.head().to_string())

        # Testa gráficos
        print(f"\nGerando graficos...")
        data_ref = df_hist['Data'].max()

        fig_curva = gerar_grafico_curva_di(df_hist, data_ref)
        print("Grafico de curva gerado")

        tickers = df_hist['Ticker'].unique()[:5]
        fig_hist = gerar_grafico_historico_di(df_hist, tickers, 90, "Teste Historico 90 dias")
        print("Grafico historico gerado")

        plt.show()
    else:
        print("\nNenhum historico disponivel. Execute pydifuturo.py primeiro.")
