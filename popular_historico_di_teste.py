"""
Script para popular histórico de DI Futuro com dados simulados.

Este script é útil para:
1. Testar o sistema de relatórios com dados históricos
2. Visualizar gráficos antes de acumular dados reais
3. Validar cálculos de variações

IMPORTANTE: Este script gera dados SIMULADOS. Use apenas para teste!
Para produção, deixe o sistema coletar dados reais diariamente.

Autor: Claude Code
Data: 2025-10-02
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

import pydifuturo


def gerar_dados_historicos_simulados(
    dias: int = 90,
    tickers: list = None,
    volatilidade: float = 0.02
) -> pd.DataFrame:
    """
    Gera dados históricos simulados de DI Futuro.

    Args:
        dias: Número de dias úteis de histórico
        tickers: Lista de tickers (se None, usa todos DI1F disponíveis)
        volatilidade: Volatilidade diária das taxas (padrão: 2%)

    Returns:
        DataFrame com histórico simulado
    """
    # Se não especificou tickers, usa os disponíveis hoje
    if tickers is None:
        df_hoje = pydifuturo.carregar_historico()
        if len(df_hoje) > 0:
            tickers = sorted(df_hoje['Ticker'].unique())
            # Pega taxas atuais como base
            taxas_base = df_hoje.set_index('Ticker')['Taxa'].to_dict()
            vencimentos = df_hoje.set_index('Ticker')['Vencimento'].to_dict()
        else:
            print("Erro: Nenhum dado atual encontrado. Execute pyieldntnb.py primeiro.")
            return pd.DataFrame()
    else:
        # Taxas base simuladas (13% base)
        taxas_base = {ticker: 13.0 + np.random.uniform(-1, 2) for ticker in tickers}
        vencimentos = {}

    print(f"Gerando {dias} dias de histórico para {len(tickers)} tickers...")

    # Data inicial (hoje - dias)
    data_fim = datetime.now().date()
    data_inicio = data_fim - timedelta(days=dias * 7 // 5)  # Aproximação dias úteis

    # Gera datas (apenas dias úteis - aproximação: seg-sex)
    datas = []
    data_atual = data_inicio
    while data_atual <= data_fim and len(datas) < dias:
        # Pula fins de semana
        if data_atual.weekday() < 5:  # 0=seg, 4=sex
            datas.append(data_atual)
        data_atual += timedelta(days=1)

    # Gera dados
    registros = []
    for ticker in tickers:
        taxa_base = taxas_base.get(ticker, 13.0)
        vencimento = vencimentos.get(ticker)

        # Random walk para simular movimento de taxas
        taxas = [taxa_base]
        for i in range(1, len(datas)):
            # Movimento browniano geométrico simplificado
            mudanca = np.random.normal(0, volatilidade)
            mean_reversion = (taxa_base - taxas[-1]) * 0.05  # Tendência de reverter à média
            nova_taxa = taxas[-1] + mudanca + mean_reversion
            # Limita variação (taxas não ficam negativas nem muito altas)
            nova_taxa = max(5.0, min(25.0, nova_taxa))
            taxas.append(nova_taxa)

        # Cria registros
        for data, taxa in zip(datas, taxas):
            registros.append({
                'Data': data,
                'Ticker': ticker,
                'Vencimento': vencimento,
                'Taxa': round(taxa, 3),
                'Volume': np.random.uniform(1e8, 1e10),  # Volume simulado
                'Contratos_Abertos': int(np.random.uniform(1e5, 1e6)),
                'Num_Negocios': int(np.random.uniform(100, 5000)),
                'Dias_Corridos': 0,  # Será calculado depois se necessário
                'Dias_Uteis': 0
            })

    df = pd.DataFrame(registros)
    print(f"Gerados {len(df)} registros ({len(datas)} datas x {len(tickers)} tickers)")

    return df


if __name__ == "__main__":
    print("=" * 80)
    print("POPULAR HISTÓRICO DI FUTURO - DADOS SIMULADOS PARA TESTE")
    print("=" * 80)
    print()
    print("ATENÇÃO: Este script gera dados SIMULADOS apenas para teste!")
    print("Para produção, deixe o pyieldntnb.py coletar dados reais diariamente.")
    print()

    # Pergunta confirmação
    resposta = input("Deseja continuar e SUBSTITUIR o histórico atual? (s/N): ")
    if resposta.lower() != 's':
        print("Operação cancelada.")
        exit()

    # Gera dados históricos (últimos 90 dias úteis)
    df_simulado = gerar_dados_historicos_simulados(dias=90)

    if len(df_simulado) > 0:
        # Salva
        arquivo = pydifuturo.salvar_historico(df_simulado, dir_dados="dados")
        print()
        print(f"✓ Histórico simulado salvo em: {arquivo}")
        print(f"✓ Total de registros: {len(df_simulado):,}")
        print(f"✓ Período: {df_simulado['Data'].min()} a {df_simulado['Data'].max()}")
        print(f"✓ Tickers: {df_simulado['Ticker'].nunique()}")
        print()
        print("Agora você pode executar pyieldntnb.py para gerar o relatório com gráficos!")
        print()
        print("LEMBRETE: Quando for para produção, delete este arquivo CSV e deixe")
        print("          o sistema coletar dados reais da API da B3 diariamente.")
    else:
        print("✗ Erro ao gerar dados simulados")
