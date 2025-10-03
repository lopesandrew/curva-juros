"""
Módulo para integração com API de Expectativas FOCUS do Banco Central do Brasil.

API: https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata

Coleta expectativas de mercado para:
- IPCA (inflação)
- Selic (taxa básica de juros)

NOTA: A API BCB FOCUS está retornando erros 400 (Bad Request) para todas as consultas OData.
Possíveis causas:
- Mudança na sintaxe OData da API
- Restrições de acesso ou rate limiting
- API temporariamente indisponível
- Necessidade de autenticação ou headers específicos

Autor: Claude Code
Data: 2025-10-02
"""

import logging
import json
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
import os

import pandas as pd
import requests


logger = logging.getLogger(__name__)


# URL base da API do BCB
BCB_API_BASE = "https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata"


def buscar_expectativas_focus(
    indicador: str,
    data_referencia: Optional[date] = None,
    cache_dir: str = "dados"
) -> pd.DataFrame:
    """
    Busca expectativas FOCUS do BCB para um indicador específico.

    Args:
        indicador: Nome do indicador ('IPCA' ou 'Selic')
        data_referencia: Data de referência (default: hoje)
        cache_dir: Diretório para cache

    Returns:
        DataFrame com expectativas
    """
    if data_referencia is None:
        data_referencia = date.today()

    logger.info(f"Buscando expectativas FOCUS para {indicador} em {data_referencia}")

    # Verifica cache
    cache_file = os.path.join(cache_dir, f"focus_{indicador}_{data_referencia.strftime('%Y%m%d')}.csv")

    if os.path.exists(cache_file):
        logger.info(f"Carregando cache: {cache_file}")
        try:
            df = pd.read_csv(cache_file, parse_dates=['Data', 'DataReferencia'])
            return df
        except Exception as e:
            logger.warning(f"Erro ao carregar cache: {e}")

    # Busca da API
    try:
        df = _buscar_api_focus(indicador, data_referencia)

        # Salva cache
        if len(df) > 0:
            os.makedirs(cache_dir, exist_ok=True)
            df.to_csv(cache_file, index=False)
            logger.info(f"Cache salvo: {cache_file}")

        return df

    except Exception as e:
        logger.error(f"Erro ao buscar expectativas FOCUS: {e}")
        return pd.DataFrame()


def _buscar_api_focus(indicador: str, data_referencia: date) -> pd.DataFrame:
    """
    Busca dados da API FOCUS do BCB.

    Endpoint: ExpectativasMercadoAnuais (para projeções anuais)

    Args:
        indicador: Nome do indicador
        data_referencia: Data de referência

    Returns:
        DataFrame com expectativas
    """
    # Endpoints diferentes para diferentes granularidades
    endpoints = {
        'anual': 'ExpectativasMercadoAnuais',
        '12meses': 'ExpectativasMercadoInflacao12Meses',
        'mensal': 'ExpectativasMercadoMensais'
    }

    resultados = []

    # 1. Busca expectativas anuais (próximos 4 anos)
    ano_atual = data_referencia.year
    anos = [ano_atual, ano_atual + 1, ano_atual + 2, ano_atual + 3]

    for ano in anos:
        url = f"{BCB_API_BASE}/{endpoints['anual']}"
        # Não filtra por data específica, pega os mais recentes disponíveis
        # IMPORTANTE: DataReferencia precisa ser string no filtro OData
        params = {
            '$filter': f"Indicador eq '{indicador}' and DataReferencia eq '{ano}'",
            '$format': 'json',
            '$select': 'Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao,Minimo,Maximo',
            '$orderby': 'Data desc',
            '$top': '1'
        }

        logger.debug(f"Buscando {indicador} para {ano}: {url}")

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            valores = data.get('value', [])

            if valores:
                resultados.extend(valores)
                logger.debug(f"  {indicador} {ano}: Mediana={valores[0].get('Mediana')}")

        except Exception as e:
            logger.warning(f"Erro ao buscar {indicador} {ano}: {e}")

    # 2. Busca expectativa 12 meses (rolling)
    if indicador == 'IPCA':
        try:
            url = f"{BCB_API_BASE}/{endpoints['12meses']}"
            # Pega os dados mais recentes disponíveis
            params = {
                '$filter': f"Indicador eq '{indicador}'",
                '$format': 'json',
                '$select': 'Indicador,Data,DataReferencia,Media,Mediana,DesvioPadrao',
                '$orderby': 'Data desc',
                '$top': 1
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()
            valores = data.get('value', [])

            if valores:
                # Adiciona identificador para diferenciar de anuais
                for v in valores:
                    v['DataReferencia'] = '12M'
                resultados.extend(valores)
                logger.debug(f"  {indicador} 12 meses: Mediana={valores[0].get('Mediana')}")

        except Exception as e:
            logger.warning(f"Erro ao buscar {indicador} 12 meses: {e}")

    if len(resultados) == 0:
        logger.warning(f"Nenhum resultado encontrado para {indicador}")
        return pd.DataFrame()

    df = pd.DataFrame(resultados)

    # Converte datas
    df['Data'] = pd.to_datetime(df['Data'])

    # DataReferencia pode ser ano (int) ou string ('12M')
    # Mantém como está para identificação

    logger.info(f"Expectativas FOCUS coletadas: {len(df)} registros para {indicador}")

    return df


def obter_resumo_focus(
    data_referencia: Optional[date] = None,
    cache_dir: str = "dados"
) -> Dict[str, pd.DataFrame]:
    """
    Obtém resumo de todas as expectativas FOCUS relevantes.

    Args:
        data_referencia: Data de referência (default: hoje)
        cache_dir: Diretório para cache

    Returns:
        Dicionário com DataFrames por indicador: {'IPCA': df, 'Selic': df}
    """
    if data_referencia is None:
        data_referencia = date.today()

    logger.info("Obtendo resumo de expectativas FOCUS")

    indicadores = ['IPCA', 'Selic']
    resumo = {}

    for indicador in indicadores:
        df = buscar_expectativas_focus(indicador, data_referencia, cache_dir)
        if len(df) > 0:
            resumo[indicador] = df

    return resumo


def formatar_tabela_focus(resumo: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Formata expectativas FOCUS em tabela consolidada para exibição.

    Args:
        resumo: Dicionário com DataFrames por indicador

    Returns:
        DataFrame formatado para HTML
    """
    if not resumo:
        return pd.DataFrame()

    linhas = []

    for indicador, df in resumo.items():
        if len(df) == 0:
            continue

        for idx, row in df.iterrows():
            ref = row.get('DataReferencia')

            # Formata referência
            if ref == '12M':
                ref_fmt = '12 Meses'
            else:
                try:
                    ref_fmt = str(int(ref))
                except:
                    ref_fmt = str(ref)

            linhas.append({
                'Indicador': indicador,
                'Referência': ref_fmt,
                'Mediana': row.get('Mediana'),
                'Média': row.get('Media'),
                'Desvio': row.get('DesvioPadrao'),
                'Mínimo': row.get('Minimo'),
                'Máximo': row.get('Maximo')
            })

    if len(linhas) == 0:
        return pd.DataFrame()

    df_tabela = pd.DataFrame(linhas)

    # Formata números
    for col in ['Mediana', 'Média', 'Desvio', 'Mínimo', 'Máximo']:
        if col in df_tabela.columns:
            df_tabela[col] = df_tabela[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")

    return df_tabela


if __name__ == "__main__":
    # Teste do módulo
    print("=" * 80)
    print("TESTE: Modulo pyfocus.py - API Expectativas BCB")
    print("=" * 80)

    # Configura logging
    logging.basicConfig(level=logging.DEBUG)

    print("\n1. Testando busca de expectativas IPCA...")
    df_ipca = buscar_expectativas_focus('IPCA')
    if len(df_ipca) > 0:
        print(f"   IPCA: {len(df_ipca)} registros coletados")
        print("\n   Primeiros registros:")
        print(df_ipca.head())
    else:
        print("   ERRO: Nenhum dado IPCA coletado")

    print("\n2. Testando busca de expectativas Selic...")
    df_selic = buscar_expectativas_focus('Selic')
    if len(df_selic) > 0:
        print(f"   Selic: {len(df_selic)} registros coletados")
        print("\n   Primeiros registros:")
        print(df_selic.head())
    else:
        print("   ERRO: Nenhum dado Selic coletado")

    print("\n3. Testando resumo FOCUS...")
    resumo = obter_resumo_focus()
    if resumo:
        print(f"   Resumo obtido: {list(resumo.keys())}")

        print("\n4. Testando formatação de tabela...")
        df_tabela = formatar_tabela_focus(resumo)
        if len(df_tabela) > 0:
            print("\n   Tabela formatada:")
            print(df_tabela.to_string(index=False))
        else:
            print("   ERRO: Tabela vazia")
    else:
        print("   ERRO: Resumo vazio")

    print("\n" + "=" * 80)
    print("Testes concluídos!")
