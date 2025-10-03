"""
Script auxiliar para preparar CSV da Bloomberg para integração com o sistema DI Futuro.

Este script ajuda a:
1. Validar formato do CSV da Bloomberg
2. Ajustar nomes de colunas
3. Converter formatos de data
4. Validar dados (taxas, tickers, etc.)
5. Adicionar colunas faltantes com valores padrão

Uso:
    python preparar_csv_bloomberg.py <arquivo_bloomberg.csv>

Autor: Claude Code
Data: 2025-10-02
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime


# Mapeamento de colunas (Bloomberg -> Sistema)
MAPEAMENTO_COLUNAS = {
    # Possíveis nomes da Bloomberg -> Nome esperado
    'Date': 'Data',
    'date': 'Data',
    'Data': 'Data',

    'Ticker': 'Ticker',
    'ticker': 'Ticker',
    'Symbol': 'Ticker',
    'Contract': 'Ticker',

    'Maturity': 'Vencimento',
    'Maturity Date': 'Vencimento',
    'Expiry': 'Vencimento',
    'Vencimento': 'Vencimento',

    'Rate': 'Taxa',
    'Yield': 'Taxa',
    'Taxa': 'Taxa',
    'Last Price': 'Taxa',

    'Volume': 'Volume',
    'Trading Volume': 'Volume',
    'Vol': 'Volume',

    'Open Interest': 'Contratos_Abertos',
    'Contratos Abertos': 'Contratos_Abertos',
    'OI': 'Contratos_Abertos',

    'Num Trades': 'Num_Negocios',
    'Number of Trades': 'Num_Negocios',
    'Trades': 'Num_Negocios',
}


def validar_ticker(ticker: str) -> bool:
    """Valida se ticker está no formato DI1FXX."""
    if not isinstance(ticker, str):
        return False
    if not ticker.startswith('DI1'):
        return False
    if len(ticker) < 5:
        return False
    # Verifica se termina com letra do mês Adicionando 2 dígitos do ano
    mes = ticker[3]
    ano = ticker[4:6]
    return mes in 'FGHJKMNQUVXZ' and ano.isdigit()


def converter_taxa_se_necessario(taxa: float) -> float:
    """
    Converte taxa de decimal para percentual se necessário.

    Se taxa < 1.0, assume que está em decimal (0.1345) e multiplica por 100.
    Se taxa >= 1.0, assume que já está em percentual (13.45).
    """
    if pd.isna(taxa):
        return np.nan

    if taxa < 1.0:
        # Provavelmente está em decimal
        return taxa * 100
    else:
        # Já está em percentual
        return taxa


def preparar_csv_bloomberg(arquivo_entrada: str, arquivo_saida: str = None):
    """
    Prepara CSV da Bloomberg para integração.

    Args:
        arquivo_entrada: Caminho do CSV da Bloomberg
        arquivo_saida: Caminho do CSV de saída (opcional, padrão: di_historico.csv)
    """
    if arquivo_saida is None:
        arquivo_saida = "dados/di_historico.csv"

    print("=" * 80)
    print("PREPARAÇÃO DE CSV DA BLOOMBERG - DI FUTURO")
    print("=" * 80)
    print()

    # 1. Carrega arquivo
    print(f"1. Carregando arquivo: {arquivo_entrada}")
    try:
        # Tenta diferentes encodings
        df = None
        for encoding in ['utf-8-sig', 'utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(arquivo_entrada, encoding=encoding)
                print(f"   OK - Arquivo carregado com encoding: {encoding}")
                break
            except:
                continue

        if df is None:
            raise ValueError("Nao foi possivel ler o arquivo com nenhum encoding comum")
    except Exception as e:
        print(f"   ERRO ao carregar arquivo: {e}")
        return False

    print(f"   Total de linhas: {len(df):,}")
    print(f"   Colunas: {list(df.columns)}")
    print()

    # 2. Mapeia colunas
    print("2. Mapeando nomes de colunas...")
    colunas_mapeadas = {}
    for col in df.columns:
        if col in MAPEAMENTO_COLUNAS:
            novo_nome = MAPEAMENTO_COLUNAS[col]
            colunas_mapeadas[col] = novo_nome
            print(f"   '{col}' -> '{novo_nome}'")

    if colunas_mapeadas:
        df = df.rename(columns=colunas_mapeadas)
        print(f"   OK {len(colunas_mapeadas)} colunas renomeadas")
    else:
        print("   ! Nenhuma coluna mapeada (nomes já podem estar corretos)")
    print()

    # 3. Valida colunas obrigatórias
    print("3. Validando colunas obrigatórias...")
    obrigatorias = ['Data', 'Ticker', 'Vencimento', 'Taxa']
    faltantes = [col for col in obrigatorias if col not in df.columns]

    if faltantes:
        print(f"   ERRO ERRO: Colunas obrigatórias faltantes: {faltantes}")
        print(f"   Colunas disponíveis: {list(df.columns)}")
        print()
        print("   Ajuste o mapeamento MAPEAMENTO_COLUNAS no script ou")
        print("   renomeie as colunas manualmente no CSV.")
        return False
    else:
        print(f"   OK Todas as colunas obrigatórias presentes")
    print()

    # 4. Converte datas
    print("4. Convertendo datas para formato ISO...")
    for col in ['Data', 'Vencimento']:
        if col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                print(f"   OK {col} convertida")
            except Exception as e:
                print(f"   ERRO Erro ao converter {col}: {e}")
                return False
    print()

    # 5. Valida e converte taxas
    print("5. Validando e convertendo taxas...")
    df['Taxa'] = df['Taxa'].apply(converter_taxa_se_necessario)
    media_taxa = df['Taxa'].mean()

    if media_taxa < 1.0:
        print(f"   ATENCAO  Taxa média muito baixa ({media_taxa:.4f}%)")
        print(f"   Isso pode indicar que taxas estão em decimal.")
        print(f"   Multiplicando por 100...")
        df['Taxa'] = df['Taxa'] * 100
        media_taxa = df['Taxa'].mean()

    print(f"   Taxa média: {media_taxa:.2f}%")
    print(f"   Taxa mínima: {df['Taxa'].min():.2f}%")
    print(f"   Taxa máxima: {df['Taxa'].max():.2f}%")

    if media_taxa < 5 or media_taxa > 30:
        print(f"   ATENCAO  ATENÇÃO: Taxa média fora do esperado (5-30%)")
        print(f"   Verifique se os dados estão corretos")
    print()

    # 6. Valida tickers
    print("6. Validando tickers...")
    tickers_invalidos = df[~df['Ticker'].apply(validar_ticker)]['Ticker'].unique()

    if len(tickers_invalidos) > 0:
        print(f"   ATENCAO  Tickers inválidos encontrados: {list(tickers_invalidos)[:10]}")
        print(f"   Total: {len(tickers_invalidos)}")

        resposta = input("   Remover linhas com tickers inválidos? (s/N): ")
        if resposta.lower() == 's':
            df = df[df['Ticker'].apply(validar_ticker)]
            print(f"   OK Linhas removidas. Total restante: {len(df):,}")
    else:
        print(f"   OK Todos os tickers são válidos")

    tickers_unicos = sorted(df['Ticker'].unique())
    print(f"   Tickers únicos: {tickers_unicos}")
    print()

    # 7. Adiciona colunas opcionais faltantes
    print("7. Adicionando colunas opcionais faltantes...")
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
    print()

    # 8. Ordena e remove duplicatas
    print("8. Ordenando e removendo duplicatas...")
    antes = len(df)
    df = df.sort_values(['Ticker', 'Data'])
    df = df.drop_duplicates(subset=['Data', 'Ticker'], keep='last')
    depois = len(df)

    if antes > depois:
        print(f"   OK {antes - depois} duplicatas removidas")
    else:
        print(f"   OK Nenhuma duplicata encontrada")
    print()

    # 9. Reorganiza colunas na ordem esperada
    print("9. Reorganizando colunas...")
    colunas_finais = [
        'Data', 'Ticker', 'Vencimento', 'Taxa', 'Volume',
        'Contratos_Abertos', 'Num_Negocios', 'Dias_Corridos', 'Dias_Uteis'
    ]
    df = df[colunas_finais]
    print(f"   OK Colunas ordenadas: {colunas_finais}")
    print()

    # 10. Salva arquivo
    print(f"10. Salvando arquivo: {arquivo_saida}")
    try:
        import os
        os.makedirs(os.path.dirname(arquivo_saida) if os.path.dirname(arquivo_saida) else '.', exist_ok=True)
        df.to_csv(arquivo_saida, index=False, encoding='utf-8-sig')
        print(f"    OK Arquivo salvo com sucesso!")
    except Exception as e:
        print(f"    ERRO Erro ao salvar: {e}")
        return False
    print()

    # Resumo final
    print("=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Total de registros: {len(df):,}")
    print(f"Período: {df['Data'].min()} a {df['Data'].max()}")
    print(f"Tickers: {df['Ticker'].nunique()} únicos")
    print(f"Taxa média: {df['Taxa'].mean():.2f}%")
    print()
    print("Arquivo pronto para uso! Execute pyieldntnb.py para gerar o relatório.")
    print("=" * 80)

    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python preparar_csv_bloomberg.py <arquivo_bloomberg.csv> [arquivo_saida.csv]")
        print()
        print("Exemplo:")
        print("  python preparar_csv_bloomberg.py bloomberg_di_export.csv")
        print("  python preparar_csv_bloomberg.py bloomberg_di_export.csv dados/di_historico.csv")
        sys.exit(1)

    arquivo_entrada = sys.argv[1]
    arquivo_saida = sys.argv[2] if len(sys.argv) > 2 else None

    sucesso = preparar_csv_bloomberg(arquivo_entrada, arquivo_saida)

    sys.exit(0 if sucesso else 1)
