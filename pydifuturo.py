"""
Módulo para coleta e processamento de dados de DI Futuro da B3.

Este módulo coleta dados de contratos DI1 (DI Futuro) através da API da B3
e armazena historicamente em arquivo CSV para análise de variações.

Autor: Claude Code
Data: 2025-10-02
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys

import requests
import pandas as pd
import numpy as np


# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Códigos de mês para futuros (padrão BMF)
CODIGOS_MES = {
    1: 'F',   # Janeiro
    2: 'G',   # Fevereiro
    3: 'H',   # Março
    4: 'J',   # Abril
    5: 'K',   # Maio
    6: 'M',   # Junho
    7: 'N',   # Julho
    8: 'Q',   # Agosto
    9: 'U',   # Setembro
    10: 'V',  # Outubro
    11: 'X',  # Novembro
    12: 'Z'   # Dezembro
}

# Mapa inverso: letra -> número do mês
MES_DE_CODIGO = {v: k for k, v in CODIGOS_MES.items()}


def coletar_di_api(meses_vencimento: List[int] = [1]) -> pd.DataFrame:
    """
    Coleta dados de DI Futuro através da API pública da B3.

    API Endpoint: https://cotacao.b3.com.br/mds/api/v1/DerivativeQuotation/DI1

    IMPORTANTE: A API da B3 já retorna a taxa anual (% a.a.) diretamente
    no campo 'curPrc', não o PU (Preço Unitário).

    Args:
        meses_vencimento: Lista de meses de vencimento (1=Jan, 2=Fev, etc)

    Returns:
        DataFrame com dados dos contratos DI1F (já com Taxa em %)
    """
    url = "https://cotacao.b3.com.br/mds/api/v1/DerivativeQuotation/DI1"

    try:
        logger.info(f"Coletando dados DI Futuro da API B3: {url}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()
        logger.info(f"Resposta da API recebida ({len(response.content):,} bytes)")

        # Parseia dados
        contratos = data.get('Scty', [])
        logger.info(f"Total de contratos retornados: {len(contratos)}")

        registros = []
        codigos_desejados = [CODIGOS_MES[mes] for mes in meses_vencimento]

        for contrato in contratos:
            simbolo = contrato.get('symb', '')

            # Filtra apenas DI1
            if not simbolo.startswith('DI1'):
                continue

            # Filtra por mês de vencimento
            if len(simbolo) >= 4:
                codigo_mes = simbolo[3]
                if codigo_mes not in codigos_desejados:
                    continue

            # Extrai dados
            scty_qtn = contrato.get('SctyQtn', {})
            asset_summry = contrato.get('asset', {}).get('AsstSummry', {})

            # Taxa atual (% a.a.) - API já retorna a taxa!
            taxa = scty_qtn.get('curPrc')
            if taxa is None:
                taxa = scty_qtn.get('prvsDayAdjstmntPric')

            # Pula se não tiver taxa
            if taxa is None or taxa <= 0:
                continue

            # Data de vencimento
            vencimento_str = asset_summry.get('mtrtyCode', '')
            if vencimento_str:
                try:
                    vencimento = datetime.strptime(vencimento_str, '%Y-%m-%d').date()
                except:
                    vencimento = None
            else:
                vencimento = None

            # Volume e contratos
            volume = asset_summry.get('grssAmt', 0)
            contratos_abertos = asset_summry.get('opnCtrcts', 0)
            num_negocios = asset_summry.get('tradQty', 0)

            registros.append({
                'Data': datetime.now().date(),
                'Ticker': simbolo,
                'Vencimento': vencimento,
                'Taxa': taxa,  # Já em % a.a.
                'Volume': volume,
                'Contratos_Abertos': contratos_abertos,
                'Num_Negocios': num_negocios
            })

        df = pd.DataFrame(registros)
        logger.info(f"Contratos DI1F filtrados: {len(df)}")

        if len(df) > 0:
            logger.info(f"Taxas coletadas. Média: {df['Taxa'].mean():.2f}%, Min: {df['Taxa'].min():.2f}%, Max: {df['Taxa'].max():.2f}%")

        return df

    except Exception as e:
        logger.error(f"Erro ao coletar dados da API B3: {e}")
        return pd.DataFrame()


def adicionar_dias_vencimento(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona colunas com dias corridos e dias úteis até vencimento.

    Args:
        df: DataFrame com colunas Data e Vencimento

    Returns:
        DataFrame com colunas Dias_Corridos e Dias_Uteis adicionadas
    """
    if len(df) == 0:
        return df

    df = df.copy()

    # Calcula dias corridos até vencimento
    df['Dias_Corridos'] = (pd.to_datetime(df['Vencimento']) - pd.to_datetime(df['Data'])).dt.days

    # Aproximação: dias úteis = dias corridos * (252/360)
    df['Dias_Uteis'] = (df['Dias_Corridos'] * 252 / 360).round().astype(int)

    return df


def carregar_historico(dir_dados: str = "dados") -> pd.DataFrame:
    """
    Carrega histórico consolidado de DI Futuro de arquivo CSV.

    Args:
        dir_dados: Diretório onde está o arquivo

    Returns:
        DataFrame com histórico completo
    """
    arquivo = os.path.join(dir_dados, "di_historico.csv")

    if os.path.exists(arquivo):
        try:
            df = pd.read_csv(arquivo, parse_dates=['Data', 'Vencimento'])
            df['Data'] = pd.to_datetime(df['Data']).dt.date
            df['Vencimento'] = pd.to_datetime(df['Vencimento']).dt.date
            logger.info(f"Histórico carregado: {len(df):,} registros de {arquivo}")
            return df
        except Exception as e:
            logger.error(f"Erro ao carregar histórico: {e}")
            return pd.DataFrame()
    else:
        logger.info(f"Arquivo {arquivo} não existe, iniciando histórico vazio")
        return pd.DataFrame()


def salvar_historico(df: pd.DataFrame, dir_dados: str = "dados") -> str:
    """
    Salva DataFrame de histórico DI Futuro em CSV.

    Args:
        df: DataFrame com dados processados
        dir_dados: Diretório de destino

    Returns:
        Caminho do arquivo salvo
    """
    os.makedirs(dir_dados, exist_ok=True)
    arquivo = os.path.join(dir_dados, "di_historico.csv")

    df.to_csv(arquivo, index=False, encoding='utf-8-sig')
    logger.info(f"Histórico DI Futuro salvo em {arquivo} ({len(df):,} linhas)")

    return arquivo


def atualizar_historico(meses_vencimento: List[int] = [1], dir_dados: str = "dados") -> pd.DataFrame:
    """
    Atualiza histórico de DI Futuro com dados do dia atual.

    Estratégia:
    1. Carrega histórico existente
    2. Coleta dados da API do dia atual
    3. Remove dados antigos do dia atual (se existirem)
    4. Adiciona novos dados
    5. Salva histórico atualizado

    Args:
        meses_vencimento: Meses de vencimento desejados (1=Jan, 2=Fev, etc)
        dir_dados: Diretório para armazenar dados

    Returns:
        DataFrame consolidado atualizado
    """
    logger.info("=" * 60)
    logger.info("Atualizando histórico de DI Futuro")
    logger.info("=" * 60)

    # Carrega histórico
    df_historico = carregar_historico(dir_dados)

    # Coleta dados da API
    df_hoje = coletar_di_api(meses_vencimento)

    if len(df_hoje) == 0:
        logger.warning("Nenhum dado coletado da API, mantendo histórico anterior")
        return df_historico

    # Adiciona dias úteis/corridos
    df_hoje = adicionar_dias_vencimento(df_hoje)

    # Remove dados do dia atual do histórico (se existirem)
    data_hoje = datetime.now().date()
    if len(df_historico) > 0:
        df_historico = df_historico[df_historico['Data'] != data_hoje]
        logger.info(f"Removidos registros anteriores de {data_hoje}")

    # Consolida
    if len(df_historico) > 0:
        df_final = pd.concat([df_historico, df_hoje], ignore_index=True)
    else:
        df_final = df_hoje

    df_final = df_final.sort_values(['Ticker', 'Data']).reset_index(drop=True)

    # Salva
    salvar_historico(df_final, dir_dados)

    logger.info("=" * 60)
    logger.info(f"Histórico atualizado com sucesso!")
    logger.info(f"Total: {len(df_final):,} registros | Tickers: {df_final['Ticker'].nunique()} | Período: {df_final['Data'].min()} a {df_final['Data'].max()}")
    logger.info("=" * 60)

    return df_final


if __name__ == "__main__":
    # Teste do módulo
    print("=" * 80)
    print("TESTE: Modulo pydifuturo.py")
    print("=" * 80)

    # Atualiza histórico
    df = atualizar_historico(meses_vencimento=[1])  # Apenas Janeiro

    if len(df) > 0:
        print(f"\nTotal de registros: {len(df):,}")
        print(f"Tickers unicos: {df['Ticker'].nunique()}")
        print(f"Periodo: {df['Data'].min()} a {df['Data'].max()}")
        print(f"\nAmostra dos dados (ultimos 10 registros):")
        print(df.tail(10).to_string())
    else:
        print("\nNenhum dado coletado")
