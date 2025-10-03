"""
Script para converter arquivo Excel da Bloomberg (.xlsx) para CSV do sistema DI Futuro.

Este script:
1. Lê o arquivo di_historico.xlsx
2. Detecta e mapeia colunas automaticamente
3. Converte para o formato esperado pelo sistema
4. Salva como di_historico.csv

Autor: Claude Code
Data: 2025-10-02
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os


# Mapeamento de colunas possíveis (case-insensitive)
MAPEAMENTO_COLUNAS = {
    # Variações de "Data"
    'date': 'Data',
    'data': 'Data',
    'dt': 'Data',
    'datetime': 'Data',

    # Variações de "Ticker"
    'ticker': 'Ticker',
    'symbol': 'Ticker',
    'contract': 'Ticker',
    'codigo': 'Ticker',
    'contrato': 'Ticker',

    # Variações de "Vencimento"
    'maturity': 'Vencimento',
    'maturity date': 'Vencimento',
    'expiry': 'Vencimento',
    'vencimento': 'Vencimento',
    'expiration': 'Vencimento',

    # Variações de "Taxa"
    'rate': 'Taxa',
    'yield': 'Taxa',
    'taxa': 'Taxa',
    'last price': 'Taxa',
    'px_last': 'Taxa',
    'price': 'Taxa',

    # Variações de "Volume"
    'volume': 'Volume',
    'trading volume': 'Volume',
    'vol': 'Volume',

    # Variações de "Contratos Abertos"
    'open interest': 'Contratos_Abertos',
    'contratos abertos': 'Contratos_Abertos',
    'oi': 'Contratos_Abertos',
    'open_interest': 'Contratos_Abertos',

    # Variações de "Num Negocios"
    'num trades': 'Num_Negocios',
    'number of trades': 'Num_Negocios',
    'trades': 'Num_Negocios',
    'num_trades': 'Num_Negocios',
}


def converter_taxa_se_necessario(taxa):
    """Converte taxa de decimal para percentual se necessário."""
    if pd.isna(taxa):
        return np.nan

    taxa_float = float(taxa)

    # Se taxa < 1.0, provavelmente está em decimal (0.1345 = 13.45%)
    if taxa_float < 1.0:
        return taxa_float * 100
    else:
        return taxa_float


def validar_ticker(ticker):
    """Valida formato do ticker DI1F."""
    if pd.isna(ticker):
        return False
    ticker_str = str(ticker).strip().upper()
    if not ticker_str.startswith('DI1'):
        return False
    if len(ticker_str) < 5:
        return False
    mes = ticker_str[3]
    return mes in 'FGHJKMNQUVXZ'


def converter_excel_para_csv(
    arquivo_excel="dados/di_historico.xlsx",
    arquivo_csv="dados/di_historico.csv"
):
    """
    Converte arquivo Excel da Bloomberg para CSV do sistema.

    Args:
        arquivo_excel: Caminho do arquivo .xlsx
        arquivo_csv: Caminho do arquivo .csv de saída
    """
    print("=" * 80)
    print("CONVERSÃO EXCEL -> CSV - DI FUTURO")
    print("=" * 80)
    print()

    # 1. Verifica se arquivo existe
    print(f"1. Verificando arquivo: {arquivo_excel}")
    if not os.path.exists(arquivo_excel):
        print(f"   ERRO: Arquivo não encontrado!")
        print(f"   Caminho esperado: {os.path.abspath(arquivo_excel)}")
        return False
    print(f"   OK - Arquivo encontrado")
    print()

    # 2. Lê arquivo Excel
    print("2. Lendo arquivo Excel...")
    try:
        # Tenta ler a primeira planilha
        df = pd.read_excel(arquivo_excel, sheet_name=0)
        print(f"   OK - Arquivo lido com sucesso")
        print(f"   Total de linhas: {len(df):,}")
        print(f"   Colunas encontradas: {list(df.columns)}")
    except Exception as e:
        print(f"   ERRO ao ler Excel: {e}")
        return False
    print()

    # 3. Remove linhas vazias
    print("3. Removendo linhas vazias...")
    antes = len(df)
    df = df.dropna(how='all')
    depois = len(df)
    if antes > depois:
        print(f"   OK - {antes - depois} linhas vazias removidas")
    else:
        print(f"   OK - Nenhuma linha vazia encontrada")
    print()

    # 4. Mapeia colunas
    print("4. Mapeando nomes de colunas...")

    # Normaliza nomes (lowercase, remove espaços extras)
    colunas_normalizadas = {col: col.lower().strip() for col in df.columns}
    df = df.rename(columns=colunas_normalizadas)

    # Mapeia para nomes do sistema
    colunas_mapeadas = {}
    for col_atual in df.columns:
        if col_atual in MAPEAMENTO_COLUNAS:
            novo_nome = MAPEAMENTO_COLUNAS[col_atual]
            colunas_mapeadas[col_atual] = novo_nome
            print(f"   '{col_atual}' -> '{novo_nome}'")

    if colunas_mapeadas:
        df = df.rename(columns=colunas_mapeadas)
        print(f"   OK - {len(colunas_mapeadas)} colunas mapeadas")
    else:
        print("   ! Nenhuma coluna mapeada automaticamente")
        print(f"   Colunas atuais: {list(df.columns)}")
    print()

    # 5. Valida colunas obrigatórias
    print("5. Validando colunas obrigatórias...")
    obrigatorias = ['Data', 'Ticker', 'Vencimento', 'Taxa']
    faltantes = [col for col in obrigatorias if col not in df.columns]

    if faltantes:
        print(f"   ERRO: Colunas obrigatórias faltantes: {faltantes}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        print()
        print("   Sugestão: Verifique os nomes das colunas no Excel e ajuste o mapeamento")
        return False
    else:
        print(f"   OK - Todas as colunas obrigatórias presentes")
    print()

    # 6. Converte tipos de dados
    print("6. Convertendo tipos de dados...")

    # Datas
    for col in ['Data', 'Vencimento']:
        if col in df.columns:
            try:
                # Tenta converter para datetime
                df[col] = pd.to_datetime(df[col], errors='coerce')
                # Remove timezone se houver
                if df[col].dtype.name == 'datetime64[ns, UTC]':
                    df[col] = df[col].dt.tz_localize(None)
                # Converte para string no formato ISO
                df[col] = df[col].dt.strftime('%Y-%m-%d')
                print(f"   OK - {col} convertida para formato ISO")
            except Exception as e:
                print(f"   ERRO ao converter {col}: {e}")
                return False

    # Taxas
    if 'Taxa' in df.columns:
        df['Taxa'] = df['Taxa'].apply(converter_taxa_se_necessario)
        media = df['Taxa'].mean()
        print(f"   OK - Taxa convertida (média: {media:.2f}%)")

        if media < 5 or media > 30:
            print(f"   ATENCAO: Taxa média fora do esperado (5-30%): {media:.2f}%")

    # Ticker (uppercase e strip)
    if 'Ticker' in df.columns:
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        print(f"   OK - Ticker normalizado")
    print()

    # 7. Valida tickers
    print("7. Validando tickers...")
    df['ticker_valido'] = df['Ticker'].apply(validar_ticker)
    invalidos = df[~df['ticker_valido']]

    if len(invalidos) > 0:
        print(f"   ATENCAO: {len(invalidos)} tickers inválidos encontrados:")
        print(invalidos[['Data', 'Ticker']].head(10).to_string(index=False))
        print()
        resposta = input("   Remover linhas com tickers inválidos? (s/N): ")
        if resposta.lower() == 's':
            df = df[df['ticker_valido']]
            print(f"   OK - Linhas removidas. Total restante: {len(df):,}")
        else:
            print(f"   ! Mantendo todas as linhas")
    else:
        print(f"   OK - Todos os tickers são válidos")

    df = df.drop(columns=['ticker_valido'])

    tickers_unicos = sorted(df['Ticker'].unique())
    print(f"   Tickers únicos: {tickers_unicos}")
    print()

    # 8. Adiciona colunas faltantes
    print("8. Adicionando colunas faltantes com valores padrão...")
    colunas_opcionais = {
        'Volume': 0.0,
        'Contratos_Abertos': 0,
        'Num_Negocios': 0,
        'Dias_Corridos': 0,
        'Dias_Uteis': 0
    }

    for col, valor_padrao in colunas_opcionais.items():
        if col not in df.columns:
            df[col] = valor_padrao
            print(f"   Adicionando {col} (valor padrão: {valor_padrao})")
        else:
            print(f"   OK - {col} já existe")
    print()

    # 9. Remove duplicatas
    print("9. Removendo duplicatas...")
    antes = len(df)
    df = df.sort_values(['Ticker', 'Data'])
    df = df.drop_duplicates(subset=['Data', 'Ticker'], keep='last')
    depois = len(df)

    if antes > depois:
        print(f"   OK - {antes - depois} duplicatas removidas")
    else:
        print(f"   OK - Nenhuma duplicata encontrada")
    print()

    # 10. Reorganiza colunas
    print("10. Reorganizando colunas...")
    colunas_finais = [
        'Data', 'Ticker', 'Vencimento', 'Taxa', 'Volume',
        'Contratos_Abertos', 'Num_Negocios', 'Dias_Corridos', 'Dias_Uteis'
    ]
    df = df[colunas_finais]
    print(f"    OK - Colunas ordenadas")
    print()

    # 11. Salva CSV
    print(f"11. Salvando arquivo: {arquivo_csv}")
    try:
        os.makedirs(os.path.dirname(arquivo_csv) if os.path.dirname(arquivo_csv) else '.', exist_ok=True)
        df.to_csv(arquivo_csv, index=False, encoding='utf-8-sig')
        print(f"    OK - Arquivo salvo com sucesso!")
    except Exception as e:
        print(f"    ERRO ao salvar: {e}")
        return False
    print()

    # Resumo
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Total de registros: {len(df):,}")
    print(f"Período: {df['Data'].min()} a {df['Data'].max()}")
    print(f"Tickers: {df['Ticker'].nunique()} únicos - {tickers_unicos}")
    print(f"Taxa média: {df['Taxa'].mean():.2f}%")
    print(f"Taxa mínima: {df['Taxa'].min():.2f}%")
    print(f"Taxa máxima: {df['Taxa'].max():.2f}%")
    print()
    print(f"Arquivo CSV salvo em: {os.path.abspath(arquivo_csv)}")
    print()
    print("Pronto! Agora você pode executar: python pyieldntnb.py")
    print("=" * 80)

    return True


if __name__ == "__main__":
    # Executa conversão
    arquivo_excel = r"C:\Users\Andrew Lopes\OneDrive - BCP Securities\Área de Trabalho\Python NTN-B\dados\di_historico.xlsx"
    arquivo_csv = r"C:\Users\Andrew Lopes\OneDrive - BCP Securities\Área de Trabalho\Python NTN-B\dados\di_historico.csv"

    sucesso = converter_excel_para_csv(arquivo_excel, arquivo_csv)

    if sucesso:
        print("\nConversao concluida com sucesso!")
    else:
        print("\nErro na conversao. Verifique as mensagens acima.")
