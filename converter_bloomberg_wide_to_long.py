"""
Script para converter Excel da Bloomberg em formato WIDE para formato LONG (CSV).

Formato WIDE (Bloomberg):
    Data       | ODF27 | ODF28 | ODF30 | ...
    2020-01-02 | 6.71  | 6.79  | 6.99  | ...
    2020-01-03 | 6.75  | 6.83  | 7.03  | ...

Formato LONG (Sistema):
    Data       | Ticker | Taxa
    2020-01-02 | DI1F27 | 6.71
    2020-01-02 | DI1F28 | 6.79
    2020-01-02 | DI1F30 | 6.99
    2020-01-03 | DI1F27 | 6.75

Autor: Claude Code
Data: 2025-10-02
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


# Mapeamento Bloomberg ticker -> Sistema ticker
MAPEAMENTO_TICKERS = {
    'ODF26 Comdty': 'DI1F26',
    'ODF27 Comdty': 'DI1F27',
    'ODF28 Comdty': 'DI1F28',
    'ODF29 Comdty': 'DI1F29',
    'ODF30 Comdty': 'DI1F30',
    'ODF31 Comdty': 'DI1F31',
    'ODF32 Comdty': 'DI1F32',
    'ODF33 Comdty': 'DI1F33',
    'ODF34 Comdty': 'DI1F34',
    'ODF35 Comdty': 'DI1F35',
    'ODF36 Comdty': 'DI1F36',
    'ODF37 Comdty': 'DI1F37',
    'ODF38 Comdty': 'DI1F38',
    'ODF39 Comdty': 'DI1F39',
    'ODF40 Comdty': 'DI1F40',
}

# Mapeamento ticker -> vencimento
VENCIMENTOS = {
    'DI1F26': '2026-01-02',
    'DI1F27': '2027-01-04',
    'DI1F28': '2028-01-03',
    'DI1F29': '2029-01-02',
    'DI1F30': '2030-01-02',
    'DI1F31': '2031-01-02',
    'DI1F32': '2032-01-02',
    'DI1F33': '2033-01-03',
    'DI1F34': '2034-01-02',
    'DI1F35': '2035-01-02',
    'DI1F36': '2036-01-02',
    'DI1F37': '2037-01-02',
    'DI1F38': '2038-01-04',
    'DI1F39': '2039-01-03',
    'DI1F40': '2040-01-02',
}


def converter_bloomberg_wide_to_long(
    arquivo_excel="dados/di_historico.xlsx",
    arquivo_csv="dados/di_historico.csv"
):
    """
    Converte Excel Bloomberg (wide format) para CSV sistema (long format).
    """
    print("=" * 80)
    print("CONVERSÃO BLOOMBERG WIDE -> LONG FORMAT")
    print("=" * 80)
    print()

    # 1. Lê Excel
    print(f"1. Lendo arquivo: {arquivo_excel}")
    if not os.path.exists(arquivo_excel):
        print(f"   ERRO: Arquivo não encontrado!")
        return False

    try:
        df = pd.read_excel(arquivo_excel, sheet_name=0)
        print(f"   OK - {len(df):,} linhas, {len(df.columns)} colunas")
    except Exception as e:
        print(f"   ERRO: {e}")
        return False
    print()

    # 2. Identifica coluna de data
    print("2. Identificando coluna de data...")
    coluna_data = None
    for col in df.columns:
        col_lower = str(col).lower()
        if 'px_last' in col_lower or 'date' in col_lower or 'data' in col_lower:
            coluna_data = col
            break

    if coluna_data is None:
        # Assume primeira coluna
        coluna_data = df.columns[0]
        print(f"   ! Assumindo primeira coluna como data: {coluna_data}")
    else:
        print(f"   OK - Coluna de data: {coluna_data}")
    print()

    # 3. Converte coluna de data
    print("3. Convertendo datas...")
    try:
        df[coluna_data] = pd.to_datetime(df[coluna_data])
        # Remove timezone se houver
        if hasattr(df[coluna_data].dtype, 'tz') and df[coluna_data].dtype.tz is not None:
            df[coluna_data] = df[coluna_data].dt.tz_localize(None)
        print(f"   OK - Período: {df[coluna_data].min().date()} a {df[coluna_data].max().date()}")
    except Exception as e:
        print(f"   ERRO: {e}")
        return False
    print()

    # 4. Identifica colunas de tickers
    print("4. Identificando colunas de tickers...")
    colunas_tickers = [col for col in df.columns if col != coluna_data]
    print(f"   Encontradas {len(colunas_tickers)} colunas de tickers:")
    for col in colunas_tickers[:10]:  # Mostra primeiras 10
        print(f"     - {col}")
    if len(colunas_tickers) > 10:
        print(f"     ... e mais {len(colunas_tickers) - 10}")
    print()

    # 5. Transforma de WIDE para LONG
    print("5. Transformando de WIDE para LONG format...")

    registros = []
    for idx, row in df.iterrows():
        data = row[coluna_data].date()

        for col in colunas_tickers:
            taxa = row[col]

            # Pula se taxa é NaN
            if pd.isna(taxa):
                continue

            # Mapeia ticker Bloomberg -> Sistema
            ticker_bloomberg = col
            ticker_sistema = MAPEAMENTO_TICKERS.get(ticker_bloomberg, None)

            if ticker_sistema is None:
                # Tenta extrair número do ticker Bloomberg (ex: ODF27 -> 27 -> DI1F27)
                try:
                    # Remove "Comdty" e espaços
                    ticker_limpo = ticker_bloomberg.replace(' Comdty', '').replace('Comdty', '').strip()
                    # Extrai número (ex: ODF27 -> 27)
                    if 'ODF' in ticker_limpo:
                        num = ticker_limpo.replace('ODF', '').strip()
                        ticker_sistema = f'DI1F{num}'
                    else:
                        continue  # Pula se não conseguir mapear
                except:
                    continue

            # Pega vencimento
            vencimento = VENCIMENTOS.get(ticker_sistema, None)

            registros.append({
                'Data': data,
                'Ticker': ticker_sistema,
                'Vencimento': vencimento,
                'Taxa': float(taxa),
                'Volume': 0.0,
                'Contratos_Abertos': 0,
                'Num_Negocios': 0,
                'Dias_Corridos': 0,
                'Dias_Uteis': 0
            })

        if (idx + 1) % 100 == 0:
            print(f"   Processando... {idx + 1:,} / {len(df):,} linhas")

    df_long = pd.DataFrame(registros)
    print(f"   OK - {len(df_long):,} registros gerados")
    print()

    # 6. Valida dados
    print("6. Validando dados...")
    print(f"   Período: {df_long['Data'].min()} a {df_long['Data'].max()}")
    print(f"   Tickers únicos: {df_long['Ticker'].nunique()}")
    print(f"   Tickers: {sorted(df_long['Ticker'].unique())}")
    print(f"   Taxa média: {df_long['Taxa'].mean():.2f}%")
    print(f"   Taxa mínima: {df_long['Taxa'].min():.2f}%")
    print(f"   Taxa máxima: {df_long['Taxa'].max():.2f}%")
    print()

    # 7. Remove duplicatas
    print("7. Removendo duplicatas...")
    antes = len(df_long)
    df_long = df_long.sort_values(['Ticker', 'Data'])
    df_long = df_long.drop_duplicates(subset=['Data', 'Ticker'], keep='last')
    depois = len(df_long)
    if antes > depois:
        print(f"   OK - {antes - depois} duplicatas removidas")
    else:
        print(f"   OK - Nenhuma duplicata")
    print()

    # 8. Formata datas para string
    print("8. Formatando datas...")
    df_long['Data'] = df_long['Data'].astype(str)
    if 'Vencimento' in df_long.columns:
        df_long['Vencimento'] = df_long['Vencimento'].fillna('')
    print(f"   OK - Datas formatadas")
    print()

    # 9. Reorganiza colunas
    print("9. Reorganizando colunas...")
    colunas_finais = [
        'Data', 'Ticker', 'Vencimento', 'Taxa', 'Volume',
        'Contratos_Abertos', 'Num_Negocios', 'Dias_Corridos', 'Dias_Uteis'
    ]
    df_long = df_long[colunas_finais]
    print(f"   OK - Colunas ordenadas")
    print()

    # 10. Salva CSV
    print(f"10. Salvando arquivo: {arquivo_csv}")
    try:
        os.makedirs(os.path.dirname(arquivo_csv) if os.path.dirname(arquivo_csv) else '.', exist_ok=True)
        df_long.to_csv(arquivo_csv, index=False, encoding='utf-8-sig')
        print(f"    OK - Arquivo salvo!")
    except Exception as e:
        print(f"    ERRO: {e}")
        return False
    print()

    # Resumo
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Total de registros: {len(df_long):,}")
    print(f"Período: {df_long['Data'].min()} a {df_long['Data'].max()}")
    print(f"Tickers: {df_long['Ticker'].nunique()} únicos")
    print(f"Taxa média: {df_long['Taxa'].mean():.2f}%")
    print()
    print(f"Arquivo CSV salvo em: {os.path.abspath(arquivo_csv)}")
    print()
    print("Pronto! Execute: python pyieldntnb.py")
    print("=" * 80)

    return True


if __name__ == "__main__":
    arquivo_excel = r"C:\Users\Andrew Lopes\OneDrive - BCP Securities\Área de Trabalho\Python NTN-B\dados\di_historico.xlsx"
    arquivo_csv = r"C:\Users\Andrew Lopes\OneDrive - BCP Securities\Área de Trabalho\Python NTN-B\dados\di_historico.csv"

    sucesso = converter_bloomberg_wide_to_long(arquivo_excel, arquivo_csv)

    if sucesso:
        print("\nConversao concluida com sucesso!")
    else:
        print("\nErro na conversao.")
