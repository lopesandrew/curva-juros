"""
Script para coleta e análise de dados de NTN-B.

Este módulo busca dados históricos de NTN-B, gera comparativos entre diferentes
períodos e produz visualizações em PDF, Excel e CSV.
"""

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
import os

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from pandas.api.types import is_numeric_dtype
from matplotlib.ticker import FuncFormatter
import yaml
import base64
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from pyield import ntnb

# Módulos DI Futuro
import pydifuturo
import pydifuturo_relatorio

# Módulos de Análises Avançadas
import pyanalise
import pyfocus

# Módulo Treasuries US
import pyust_data


# ============================================
# CONFIGURAÇÃO E CONSTANTES
# ============================================

def load_config(config_path: str = "config_ntnb.yaml") -> Dict[str, Any]:
    """
    Carrega configurações do arquivo YAML.

    Args:
        config_path: Caminho para o arquivo de configuração

    Returns:
        Dicionário com configurações

    Raises:
        FileNotFoundError: Se o arquivo não existir
        yaml.YAMLError: Se houver erro ao parsear o YAML
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: Dict[str, Any]) -> None:
    """
    Configura o sistema de logging.

    Args:
        config: Dicionário de configuração
    """
    log_config = config.get('logging', {})
    level = getattr(logging, log_config.get('level', 'INFO'))
    format_str = log_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    date_format = log_config.get('date_format', '%Y-%m-%d %H:%M:%S')

    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt=date_format
    )


# ============================================
# CONSTANTES DE FORMATAÇÃO
# ============================================

MESES_PT = ["jan", "fev", "mar", "abr", "mai", "jun",
            "jul", "ago", "set", "out", "nov", "dez"]


# ============================================
# FUNÇÕES DE BUSCA E COLETA
# ============================================

def busca_ultimo_com_dados(
    fetch_fn: Callable[[str], pd.DataFrame],
    data_base: date,
    max_lookback: int
) -> Tuple[date, pd.DataFrame]:
    """
    Busca retroativamente por dados válidos a partir de uma data base.

    Args:
        fetch_fn: Função de fetch que recebe data como string
        data_base: Data inicial de busca
        max_lookback: Número máximo de dias para buscar retroativamente

    Returns:
        Tupla (data_encontrada, dataframe)

    Raises:
        RuntimeError: Se não encontrar dados dentro da janela de busca
    """
    data_busca = data_base

    for _ in range(max_lookback):
        data_str = data_busca.strftime("%d-%m-%Y")
        try:
            df = fetch_fn(data_str)
            if isinstance(df, pd.DataFrame) and not df.empty:
                logging.debug(f"Dados encontrados para {data_str}")
                return data_busca, df
        except Exception as e:
            logging.debug(f"Erro ao buscar dados para {data_str}: {e}")

        data_busca -= timedelta(days=1)

    raise RuntimeError(
        f"Não foi possível encontrar dados dentro da janela de {max_lookback} dias"
    )


def get_ntnb_por_data_alvo(data_alvo: date) -> Tuple[date, pd.DataFrame]:
    """
    Busca dados de NTN-B para uma data alvo, filtrando apenas títulos PRINCIPAL.

    Args:
        data_alvo: Data alvo para busca

    Returns:
        Tupla (data_encontrada, dataframe_filtrado)

    Raises:
        RuntimeError: Se não encontrar dados válidos
    """
    def fetch_filtrado(data_str: str) -> pd.DataFrame:
        try:
            df_raw = ntnb.data(data_str)
        except Exception as e:
            logging.debug(f"Erro ao buscar dados NTN-B: {e}")
            return pd.DataFrame()

        if not isinstance(df_raw, pd.DataFrame):
            return pd.DataFrame()

        df = df_raw.copy()

        # Converter data de vencimento
        df["MaturityDate"] = pd.to_datetime(
            df.get("MaturityDate"),
            dayfirst=True,
            errors="coerce"
        )

        # Filtrar apenas títulos PRINCIPAL
        if "BondType" in df.columns:
            filtro_principal = df["BondType"].str.contains("PRINCIPAL", case=False, na=False)
            if filtro_principal.any():
                df = df[filtro_principal]

        # Selecionar colunas relevantes
        cols = ["MaturityDate", "BidRate", "BondType"]
        cols_existentes = [c for c in cols if c in df.columns]

        if not cols_existentes:
            return pd.DataFrame()

        df = df[cols_existentes]
        df = df.dropna(subset=[c for c in ["MaturityDate", "BidRate"] if c in df.columns])

        return df

    data_encontrada, df = busca_ultimo_com_dados(fetch_filtrado, data_alvo, max_lookback=25)
    return data_encontrada, df


def carregar_cache_historico(maturity_date: date, dir_dados: str) -> pd.DataFrame:
    """Carrega cache de histórico para um vencimento."""
    cache_file = os.path.join(dir_dados, f"cache_historico_{maturity_date.strftime('%Y%m%d')}.csv")
    if os.path.exists(cache_file):
        try:
            df = pd.read_csv(cache_file, parse_dates=['Data'])
            df['Data'] = df['Data'].dt.date
            return df
        except Exception as e:
            logging.warning(f"Erro ao carregar cache {cache_file}: {e}")
    return pd.DataFrame(columns=["Data", "BidRate"])


def salvar_cache_historico(df: pd.DataFrame, maturity_date: date, dir_dados: str):
    """Salva cache de histórico para um vencimento."""
    if df.empty:
        return
    cache_file = os.path.join(dir_dados, f"cache_historico_{maturity_date.strftime('%Y%m%d')}.csv")
    try:
        df.to_csv(cache_file, index=False)
    except Exception as e:
        logging.warning(f"Erro ao salvar cache {cache_file}: {e}")


def coleta_historico_maturity(
    maturity_date: date,
    data_final: date,
    dias_periodo: int,
    bond_type: Optional[str] = None,
    dir_dados: str = "dados"
) -> pd.DataFrame:
    """
    Coleta histórico de taxas para um vencimento específico (com cache).

    Args:
        maturity_date: Data de vencimento do título
        data_final: Data final do período
        dias_periodo: Número de dias do período histórico
        bond_type: Tipo de título para filtrar (opcional)
        dir_dados: Diretório para cache

    Returns:
        DataFrame com colunas ['Data', 'BidRate']
    """
    # Carrega cache existente
    df_cache = carregar_cache_historico(maturity_date, dir_dados)

    limite = data_final - timedelta(days=dias_periodo)

    # Se cache tem dados suficientes, complementa apenas datas novas
    if not df_cache.empty:
        data_max_cache = df_cache['Data'].max()
        # Se cache está atualizado, retorna
        if data_max_cache >= data_final:
            df_result = df_cache[df_cache['Data'] >= limite].copy()
            return df_result.reset_index(drop=True)

        # Coleta apenas datas novas (dos últimos 60 dias)
        dias_atualizar = min(60, (data_final - data_max_cache).days + 5)
    else:
        dias_atualizar = dias_periodo + 120

    dados = []
    vistos = set()
    atingiu_limite = False
    max_offset = dias_atualizar

    for offset in range(max_offset + 1):
        alvo = data_final - timedelta(days=offset)

        # Otimização: parar se já passou 30 dias além do limite ou se já está no cache
        if not df_cache.empty and alvo <= data_max_cache:
            break

        if alvo < limite - timedelta(days=30) and atingiu_limite:
            break

        try:
            data_real, df = get_ntnb_por_data_alvo(alvo)
        except RuntimeError:
            break

        # Evitar duplicatas
        if data_real in vistos:
            continue

        # Filtrar por maturity
        subset_base = df[df["MaturityDate"].dt.date == maturity_date]
        subset = subset_base

        # Filtrar por bond_type se especificado
        if bond_type and "BondType" in subset.columns:
            filtro = subset["BondType"].str.upper().str.contains(bond_type.upper(), na=False)
            if filtro.any():
                subset = subset[filtro]

        if subset.empty:
            continue

        taxa = subset["BidRate"].iloc[0]
        if pd.isna(taxa):
            continue

        dados.append((data_real, float(taxa)))
        vistos.add(data_real)

        if data_real <= limite:
            atingiu_limite = True

    # Combina novos dados com cache
    if dados:
        df_novos = pd.DataFrame(dados, columns=["Data", "BidRate"])
        if not df_cache.empty:
            df_hist = pd.concat([df_cache, df_novos], ignore_index=True)
            df_hist = df_hist.drop_duplicates(subset=['Data'], keep='last')
        else:
            df_hist = df_novos
    else:
        df_hist = df_cache

    if df_hist.empty:
        logging.warning(f"Nenhum dado histórico encontrado para {maturity_date}")
        return pd.DataFrame(columns=["Data", "BidRate"])

    df_hist = df_hist.sort_values("Data")

    # Salva cache atualizado
    salvar_cache_historico(df_hist, maturity_date, dir_dados)

    # Retorna apenas período solicitado
    df_result = df_hist[df_hist["Data"] >= limite].copy()
    return df_result.reset_index(drop=True)


# ============================================
# FUNÇÕES DE FORMATAÇÃO
# ============================================

def format_percent(value: Any, decimals: int = 4) -> str:
    """
    Formata um valor decimal como percentual.

    Args:
        value: Valor a formatar
        decimals: Número de casas decimais

    Returns:
        String formatada como percentual
    """
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.{decimals}f}%".replace(".", ",")


def formata_mes_ano(dt: Any) -> str:
    """
    Formata datetime como mês/ano abreviado em português.

    Args:
        dt: Datetime a formatar

    Returns:
        String no formato "mmm/aa" (ex: "jan/25")
    """
    if pd.isna(dt):
        return ""
    return f"{MESES_PT[dt.month - 1]}/{str(dt.year)[-2:]}"


def tick_mes_ano(x: float, _: Any) -> str:
    """
    Formatter para eixo X de gráficos matplotlib.

    Args:
        x: Valor numérico da data
        _: Posição (não utilizado)

    Returns:
        String formatada
    """
    dt = mdates.num2date(x)
    return formata_mes_ano(dt)


# ============================================
# FUNÇÕES DE CONVERSÃO PARA HTML
# ============================================

def fig_to_base64(fig: plt.Figure, dpi: int = 150) -> str:
    """
    Converte figura matplotlib para string base64.

    Args:
        fig: Figura matplotlib
        dpi: DPI para renderização

    Returns:
        String base64 da imagem PNG
    """
    buffer = BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    return img_base64


def df_to_html_table(df: pd.DataFrame, title: str = "",
                     highlight_bps: bool = False,
                     header_color: str = '#4472C4') -> str:
    """
    Converte DataFrame para tabela HTML estilizada.

    Args:
        df: DataFrame a converter
        title: Título da tabela
        highlight_bps: Se True, colore células de variação (bps)
        header_color: Cor do cabeçalho

    Returns:
        String HTML da tabela
    """
    html = []

    if title:
        html.append(f'<h2 style="color: #2c3e50; margin-top: 30px; margin-bottom: 15px; font-size: 20px; text-align: center; font-weight: bold;">{title}</h2>')

    # Wrapper centralizado para a tabela
    html.append('<div style="width: 100%; overflow-x: auto; text-align: center; margin-bottom: 30px;">')
    html.append('<table cellpadding="0" cellspacing="0" border="0" style="width: auto; min-width: 700px; max-width: 1000px; border-collapse: collapse; margin: 0 auto; font-family: Arial, sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: inline-table;">')

    # Cabeçalho
    html.append('<thead>')
    html.append('<tr>')
    for col in df.columns:
        html.append(f'<th style="background-color: {header_color} !important; color: white !important; padding: 16px 20px !important; text-align: center !important; font-weight: bold !important; border: 1px solid #ddd !important; font-size: 16px !important; font-family: Arial, sans-serif !important; white-space: nowrap !important; mso-line-height-rule: exactly; line-height: 1.4;">{col}</th>')
    html.append('</tr>')
    html.append('</thead>')

    # Corpo
    html.append('<tbody>')
    for idx, row in df.iterrows():
        # Alternar cores de linha (banded rows)
        bg_color = '#F2F2F2' if idx % 2 == 0 else 'white'
        html.append('<tr>')

        for col_idx, (col, value) in enumerate(row.items()):
            cell_bg = bg_color

            # Colorir células de variação bps (apenas a fonte)
            cell_color = '#333'  # Cor padrão
            font_weight = 'normal'

            # Primeira coluna sempre em negrito
            if col_idx == 0:
                font_weight = 'bold'

            if highlight_bps and col.endswith('(bps)'):
                try:
                    # Tentar converter para número
                    if isinstance(value, str):
                        num_value = float(value.replace(',', '.').replace('+', ''))
                    else:
                        num_value = float(value)

                    if num_value > 0:
                        cell_color = '#d32f2f'  # Vermelho (alta)
                        font_weight = 'bold'
                    elif num_value < 0:
                        cell_color = '#388e3c'  # Verde (queda)
                        font_weight = 'bold'
                except:
                    pass

            cell_style = f'padding: 14px 20px !important; text-align: center !important; border: 1px solid #ddd !important; background-color: {cell_bg} !important; color: {cell_color} !important; font-weight: {font_weight} !important; font-size: 16px !important; font-family: Arial, sans-serif !important; white-space: nowrap !important; mso-line-height-rule: exactly; line-height: 1.4;'
            html.append(f'<td style="{cell_style}">{value}</td>')

        html.append('</tr>')
    html.append('</tbody>')
    html.append('</table>')
    html.append('</div>')  # Fechar wrapper

    # Adicionar um comentário para forçar renderização correta
    html.append('<!--[if mso]></td></tr></table><![endif]-->')

    return '\n'.join(html)


def gerar_html_relatorio(
    df_variacoes: pd.DataFrame,
    df_comparativo: pd.DataFrame,
    fig_curvas: Optional[plt.Figure],
    fig_hist_3y: Optional[plt.Figure],
    fig_hist_5y: Optional[plt.Figure],
    data_ref: date,
    config: Dict[str, Any],
    # Parâmetros opcionais para DI Futuro
    df_di_variacoes: Optional[pd.DataFrame] = None,
    fig_di_curva: Optional[plt.Figure] = None,
    fig_di_hist_12m: Optional[plt.Figure] = None,
    fig_di_hist_3y: Optional[plt.Figure] = None,
    fig_di_hist_5y: Optional[plt.Figure] = None,
    # Parâmetros opcionais para Análises Avançadas
    df_bei: Optional[pd.DataFrame] = None,
    df_steepness_di: Optional[pd.DataFrame] = None,
    df_steepness_ntnb: Optional[pd.DataFrame] = None,
    df_stats_di: Optional[pd.DataFrame] = None,
    df_focus: Optional[pd.DataFrame] = None,
    # Parâmetros opcionais para Treasuries US
    df_ust_variacoes: Optional[pd.DataFrame] = None,
    fig_ust_curva: Optional[plt.Figure] = None,
    fig_ust_hist_10y: Optional[plt.Figure] = None
) -> str:
    """
    Gera relatório HTML completo com tabelas e gráficos (NTN-B e DI Futuro).

    Args:
        df_variacoes: DataFrame de variações NTN-B
        df_comparativo: DataFrame comparativo NTN-B
        fig_curvas: Figura de curvas NTN-B
        fig_hist_3y: Figura histórico 3 anos NTN-B
        fig_hist_5y: Figura histórico 5 anos NTN-B
        data_ref: Data de referência
        config: Configuração global
        df_di_variacoes: DataFrame de variações DI Futuro (opcional)
        fig_di_curva: Figura curva DI (opcional)
        fig_di_hist_12m: Figura histórico 12 meses DI (opcional)
        fig_di_hist_3y: Figura histórico 3 anos DI (opcional)
        fig_di_hist_5y: Figura histórico 5 anos DI (opcional)

    Returns:
        String HTML completa
    """
    formato_cfg = config.get('formato', {})
    date_format = formato_cfg.get('date_format', '%d/%m/%Y')
    data_str = data_ref.strftime(date_format)

    html = []

    # Cabeçalho HTML (otimizado para clientes de email)
    html.append('''<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Relatório NTN-B</title>
</head>
<body style="font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #ffffff;">
    <div style="max-width: 1200px; margin: 0 auto; background-color: #ffffff; padding: 20px;">''')

    # Título e metadados
    html.append(f'<h1 style="color: #2c3e50; border-bottom: 3px solid #4472C4; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">Curva de Juros - {data_str}</h1>')
    html.append(f'<div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #4472C4; margin-bottom: 30px;">')
    html.append(f'<p style="margin: 5px 0; font-size: 14px; color: #555;"><strong>Gerado em:</strong> {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>')
    html.append(f'</div>')

    # Seção 1: Tabela de Variações
    html.append('<div style="margin-bottom: 40px;">')
    if not df_variacoes.empty:
        html.append(df_to_html_table(df_variacoes, "Variações de Taxa", highlight_bps=True))
    html.append('</div>')

    # Seção 2: Tabela Comparativa
    html.append('<div style="margin-bottom: 40px;">')
    if not df_comparativo.empty:
        html.append(df_to_html_table(df_comparativo, "Comparativo de Taxas por Vencimento"))
    html.append('</div>')

    # Seção 3: Gráfico de Curvas
    if fig_curvas is not None:
        html.append('<div style="margin-bottom: 40px;">')
        html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Curvas de Taxa</h2>')
        html.append('<div style="text-align: center; margin: 20px 0;">')
        img_b64 = fig_to_base64(fig_curvas, dpi=150)
        html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Gráfico de Curvas" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
        html.append('</div>')
        html.append('</div>')

    # Seção 4: Gráfico Histórico 3 Anos
    if fig_hist_3y is not None:
        html.append('<div style="margin-bottom: 40px;">')
        html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Histórico - 3 Anos</h2>')
        html.append('<div style="text-align: center; margin: 20px 0;">')
        img_b64 = fig_to_base64(fig_hist_3y, dpi=150)
        html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Gráfico Histórico 3 Anos" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
        html.append('</div>')
        html.append('</div>')

    # Seção 5: Gráfico Histórico 5 Anos
    if fig_hist_5y is not None:
        html.append('<div style="margin-bottom: 40px;">')
        html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Histórico - 5 Anos</h2>')
        html.append('<div style="text-align: center; margin: 20px 0;">')
        img_b64 = fig_to_base64(fig_hist_5y, dpi=150)
        html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Gráfico Histórico 5 Anos" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
        html.append('</div>')
        html.append('</div>')

    # ==============================================
    # SEÇÃO DI FUTURO (se habilitada)
    # ==============================================
    if df_di_variacoes is not None and not df_di_variacoes.empty:
        html.append('<hr style="margin: 60px 0; border: none; border-top: 2px solid #4472C4;">')
        html.append('<h1 style="color: #2c3e50; border-bottom: 3px solid #ff7f0e; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">DI Futuro - Análise de Taxas</h1>')

        # Seção DI-1: Tabela de Variações
        html.append('<div style="margin-bottom: 40px;">')
        html.append(df_to_html_table(df_di_variacoes, "Variações de Taxa - DI Futuro (Vencimentos Janeiro)", highlight_bps=True))
        html.append('</div>')

        # Seção DI-2: Gráfico de Curva
        if fig_di_curva is not None:
            html.append('<div style="margin-bottom: 40px;">')
            html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Curva de Juros - DI Futuro</h2>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_di_curva, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Curva DI Futuro" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

        # Seção DI-3: Histórico 12 Meses
        if fig_di_hist_12m is not None:
            html.append('<div style="margin-bottom: 40px;">')
            html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Histórico DI Futuro - 12 Meses</h2>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_di_hist_12m, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Histórico 12 Meses DI" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

        # Seção DI-4: Histórico 3 Anos
        if fig_di_hist_3y is not None:
            html.append('<div style="margin-bottom: 40px;">')
            html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Histórico DI Futuro - 3 Anos</h2>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_di_hist_3y, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Histórico 3 Anos DI" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

        # Seção DI-5: Histórico 5 Anos
        if fig_di_hist_5y is not None:
            html.append('<div style="margin-bottom: 40px;">')
            html.append('<h2 style="color: #2c3e50; margin-bottom: 15px; font-size: 18px; font-weight: bold; text-align: center;">Histórico DI Futuro - 5 Anos</h2>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_di_hist_5y, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Histórico 5 Anos DI" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

    # ============================================
    # SEÇÃO: ANÁLISES AVANÇADAS
    # ============================================

    if df_bei is not None and len(df_bei) > 0:
        html.append('<hr style="margin: 60px 0; border: none; border-top: 2px solid #4472C4;">')
        html.append('<h1 style="color: #2c3e50; border-bottom: 3px solid #28a745; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">Breakeven Inflation (Inflação Implícita)</h1>')
        html.append('<p style="color: #555; margin-bottom: 20px; font-size: 14px;">Inflação implícita calculada a partir da diferença entre taxas nominais (DI) e reais (NTN-B)</p>')

        # Pega dados mais recentes
        data_max = df_bei['Data'].max()
        df_bei_atual = df_bei[df_bei['Data'] == data_max].copy()

        if len(df_bei_atual) > 0:
            html.append('<div style="margin-bottom: 40px;">')

            # Formata tabela BEI
            df_bei_html = df_bei_atual.copy()

            # Formata vencimento
            df_bei_html['Vencimento'] = pd.to_datetime(df_bei_html['Vencimento']).dt.strftime('%d/%m/%Y')

            df_bei_html['NTNB_Real'] = df_bei_html['NTNB_Real'].apply(lambda x: f"{x:.2f}%")
            df_bei_html['DI_Nominal'] = df_bei_html['DI_Nominal'].apply(lambda x: f"{x:.2f}%")
            df_bei_html['BEI'] = df_bei_html['BEI'].apply(lambda x: f"{x:.2f}%")

            # Remove colunas desnecessárias e renomeia
            df_bei_html = df_bei_html[['Vencimento', 'Ticker_DI', 'NTNB_Real', 'DI_Nominal', 'BEI']]
            df_bei_html.columns = ['Vencimento NTN-B', 'DI Futuro', 'Yield Real', 'Taxa Nominal', 'Inflação Implícita']

            html.append(df_to_html_table(df_bei_html, "Breakeven Inflation - Inflação Implícita do Mercado"))
            html.append('</div>')

    if df_steepness_di is not None and len(df_steepness_di) > 0:
        html.append('<hr style="margin: 60px 0; border: none; border-top: 2px solid #4472C4;">')
        html.append('<h1 style="color: #2c3e50; border-bottom: 3px solid #17a2b8; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">Steepness (Inclinação da Curva)</h1>')
        html.append('<p style="color: #555; margin-bottom: 20px; font-size: 14px;">Diferença entre vencimentos longos e curtos (medida de inclinação da curva de juros)</p>')

        # Cria tabela consolidada de steepness
        steepness_data = []

        # Steepness DI
        data_max_di = df_steepness_di['Data'].max()
        row_di = df_steepness_di[df_steepness_di['Data'] == data_max_di].iloc[-1]
        steep_di = row_di['Steepness']

        # Cor: vermelho para positivo, verde para negativo
        cor_di = 'red' if steep_di > 0 else 'green'
        steep_di_formatado = f'<span style="color: {cor_di}; font-weight: bold;">{steep_di:.4f}</span>'

        steepness_data.append({
            'Curva': 'DI Futuro',
            'Vencimento Curto': row_di['Ticker_Curto'],
            'Taxa Curta': f"{row_di['Taxa_Curto']:.2f}%",
            'Vencimento Longo': row_di['Ticker_Longo'],
            'Taxa Longa': f"{row_di['Taxa_Longo']:.2f}%",
            'Steepness (bps)': steep_di_formatado
        })

        # Steepness NTN-B (se disponível)
        if df_steepness_ntnb is not None and len(df_steepness_ntnb) > 0:
            data_max_ntnb = df_steepness_ntnb['Data'].max()
            row_ntnb = df_steepness_ntnb[df_steepness_ntnb['Data'] == data_max_ntnb].iloc[-1]
            steep_ntnb = row_ntnb['Steepness']

            # Formata vencimentos
            venc_curto = pd.to_datetime(row_ntnb['Ticker_Curto']).strftime('%d/%m/%Y')
            venc_longo = pd.to_datetime(row_ntnb['Ticker_Longo']).strftime('%d/%m/%Y')

            # Cor: vermelho para positivo, verde para negativo
            cor_ntnb = 'red' if steep_ntnb > 0 else 'green'
            steep_ntnb_formatado = f'<span style="color: {cor_ntnb}; font-weight: bold;">{steep_ntnb:.4f}</span>'

            steepness_data.append({
                'Curva': 'NTN-B',
                'Vencimento Curto': venc_curto,
                'Taxa Curta': f"{row_ntnb['Taxa_Curto']:.2f}%",
                'Vencimento Longo': venc_longo,
                'Taxa Longa': f"{row_ntnb['Taxa_Longo']:.2f}%",
                'Steepness (bps)': steep_ntnb_formatado
            })

        df_steep_html = pd.DataFrame(steepness_data)

        html.append('<div style="margin-bottom: 40px;">')
        html.append(df_to_html_table(df_steep_html, "Steepness - Inclinação das Curvas de Juros"))
        html.append('</div>')

    # Estatísticas Históricas - Comentado a pedido do usuário
    # if df_stats_di is not None and len(df_stats_di) > 0:
    #     html.append('<hr style="margin: 60px 0; border: none; border-top: 2px solid #4472C4;">')
    #     html.append('<h1 style="color: #2c3e50; border-bottom: 3px solid #ffc107; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">Estatísticas Históricas - DI Futuro</h1>')
    #
    #     # Formata tabela de estatísticas
    #     df_stats_html = df_stats_di.copy()
    #
    #     # Formata números
    #     for col in df_stats_html.columns:
    #         if 'Taxa' in col or 'High' in col or 'Low' in col or 'Media' in col:
    #             df_stats_html[col] = df_stats_html[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else "-")
    #         elif 'Percentil' in col:
    #             df_stats_html[col] = df_stats_html[col].apply(lambda x: f"{x:.0f}%" if pd.notna(x) else "-")
    #
    #     html.append(df_to_html_table(df_stats_html, "Estatísticas - High/Low e Posição no Range"))

    if df_focus is not None and len(df_focus) > 0:
        html.append('<hr style="margin: 60px 0; border: none; border-top: 2px solid #4472C4;">')
        html.append('<h1 style="color: #2c3e50; border-bottom: 3px solid #dc3545; padding-bottom: 15px; margin-bottom: 30px; font-size: 24px; font-weight: bold;">Expectativas FOCUS - Banco Central</h1>')

        html.append(df_to_html_table(df_focus, "Expectativas de Mercado - IPCA e Selic"))

    # ============================================
    # SEÇÃO: Treasuries US
    # ============================================
    if df_ust_variacoes is not None and not df_ust_variacoes.empty:
        html.append('<div style="page-break-before: always; margin-top: 40px; padding-top: 30px; border-top: 3px solid #4472C4;">')
        html.append('<h2 style="color: #2c3e50; margin-bottom: 20px; font-size: 20px; font-weight: bold;">📊 US Treasuries</h2>')

        # Tabela de Variações
        html.append(df_to_html_table(df_ust_variacoes, "Variações de Taxa - Treasuries", highlight_bps=True))

        # Gráfico de Curvas
        if fig_ust_curva is not None:
            html.append('<div style="margin: 30px 0;">')
            html.append('<h3 style="color: #2c3e50; margin-bottom: 15px; font-size: 16px; font-weight: bold; text-align: center;">Curvas de Taxa - Treasuries</h3>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_ust_curva, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Gráfico de Curvas Treasuries" style="width: 100%; max-width: 900px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

        # Gráfico Histórico Treasuries
        if fig_ust_hist_10y is not None:
            html.append('<div style="margin: 30px 0;">')
            html.append('<h3 style="color: #2c3e50; margin-bottom: 15px; font-size: 16px; font-weight: bold; text-align: center;">Histórico - US Treasuries (5Y, 7Y, 10Y, 20Y, 30Y)</h3>')
            html.append('<div style="text-align: center; margin: 20px 0;">')
            img_b64 = fig_to_base64(fig_ust_hist_10y, dpi=150)
            html.append(f'<img src="data:image/png;base64,{img_b64}" alt="Histórico US Treasuries" style="width: 100%; max-width: 1000px; height: auto; border: 1px solid #ddd; display: block; margin: 0 auto;">')
            html.append('</div>')
            html.append('</div>')

        html.append('</div>')

    # ============================================
    # FOOTER: Fontes de Dados e Tecnologias
    # ============================================
    html.append('<div style="margin-top: 60px; padding: 30px; border-top: 3px solid #4472C4; background: linear-gradient(to bottom, #f8f9fa 0%, #ffffff 100%);">')

    # Fontes de Dados
    html.append('<h3 style="color: #2c3e50; font-size: 16px; font-weight: bold; margin-bottom: 20px; border-bottom: 2px solid #4472C4; padding-bottom: 10px;">📊 Fontes de Dados</h3>')
    html.append('<ul style="color: #555; font-size: 12px; line-height: 2.0; margin: 0 0 25px 20px; list-style-type: none;">')
    html.append('<li style="padding: 5px 0;">📈 <strong>NTN-B:</strong> ANBIMA (Associação Brasileira das Entidades dos Mercados Financeiro e de Capitais)</li>')
    html.append('<li style="padding: 5px 0;">💹 <strong>DI Futuro:</strong> B3 (Brasil, Bolsa, Balcão) via API</li>')
    html.append('<li style="padding: 5px 0;">🏦 <strong>Expectativas FOCUS:</strong> Banco Central do Brasil (BCB)</li>')
    html.append('<li style="padding: 5px 0;">🇺🇸 <strong>US Treasuries:</strong> Federal Reserve Economic Data (FRED) - St. Louis Fed</li>')
    html.append('</ul>')

    # Tecnologias
    html.append('<h3 style="color: #2c3e50; font-size: 16px; font-weight: bold; margin-bottom: 20px; border-bottom: 2px solid #4472C4; padding-bottom: 10px;">🛠️ Tecnologias Utilizadas</h3>')
    html.append('<p style="color: #555; font-size: 12px; line-height: 1.8; margin: 0 0 10px 0;"><strong>Python 3.x</strong> com os seguintes pacotes:</p>')
    html.append('<ul style="color: #555; font-size: 12px; line-height: 1.8; margin: 0 0 0 20px; list-style-type: disc;">')
    html.append('<li><strong>pandas</strong> - Manipulação e análise de dados</li>')
    html.append('<li><strong>numpy</strong> - Computação numérica</li>')
    html.append('<li><strong>matplotlib</strong> - Visualização de dados e gráficos</li>')
    html.append('<li><strong>requests</strong> - Requisições HTTP para APIs</li>')
    html.append('<li><strong>PyYAML</strong> - Configuração em formato YAML</li>')
    html.append('<li><strong>pyield</strong> - Biblioteca para dados de NTN-B</li>')
    html.append('<li><strong>fredapi</strong> - API para dados do Federal Reserve (FRED)</li>')
    html.append('<li><strong>feedparser</strong> - Parser de feeds RSS/Atom para notícias</li>')
    html.append('</ul>')

    # Copyright
    html.append('<div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #ddd; text-align: center;">')
    html.append('<p style="color: #999; font-size: 11px; margin: 0;">Relatório gerado automaticamente | Desenvolvido com Python 3.x</p>')
    html.append('</div>')

    html.append('</div>')

    html.append('</div></body></html>')

    return '\n'.join(html)


# ============================================
# FUNÇÕES DE EMAIL
# ============================================

def enviar_email_html(
    html_content: str,
    subject: str,
    config: Dict[str, Any]
) -> bool:
    """
    Envia relatório HTML por email.

    Args:
        html_content: Conteúdo HTML do relatório
        subject: Assunto do email
        config: Configuração global

    Returns:
        True se enviado com sucesso, False caso contrário
    """
    email_cfg = config.get('email', {})

    # Verificar se email está habilitado
    if not email_cfg.get('enabled', False):
        logging.info("Envio de email desabilitado na configuração")
        return False

    # Validar configurações
    sender_email = email_cfg.get('sender_email', '')
    sender_password = email_cfg.get('sender_password', '')
    recipients = email_cfg.get('recipients', [])
    smtp_server = email_cfg.get('smtp_server', 'smtp.gmail.com')
    smtp_port = email_cfg.get('smtp_port', 587)

    if not sender_email or not sender_password:
        logging.error("Email ou senha não configurados. Verifique o arquivo config_ntnb.yaml")
        return False

    if not recipients:
        logging.error("Nenhum destinatário configurado")
        return False

    try:
        # Criar mensagem
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = sender_email
        msg['To'] = ', '.join(recipients)

        # Adicionar HTML
        html_part = MIMEText(html_content, 'html', 'utf-8')
        msg.attach(html_part)

        # Conectar ao servidor SMTP
        logging.info(f"Conectando ao servidor SMTP {smtp_server}:{smtp_port}...")
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(sender_email, sender_password)

            # Enviar email
            logging.info(f"Enviando email para {len(recipients)} destinatário(s)...")
            server.send_message(msg)

        logging.info(f"Email enviado com sucesso para: {', '.join(recipients)}")
        return True

    except smtplib.SMTPAuthenticationError:
        logging.error("Erro de autenticação. Verifique email e senha (use senha de aplicativo do Gmail)")
        return False
    except smtplib.SMTPException as e:
        logging.error(f"Erro SMTP ao enviar email: {e}")
        return False
    except Exception as e:
        logging.error(f"Erro inesperado ao enviar email: {e}")
        return False


# ============================================
# FUNÇÕES DE SALVAMENTO
# ============================================

def salvar_excel(
    dfs: Dict[str, pd.DataFrame],
    filepath: str,
    config: Dict[str, Any]
) -> None:
    """
    Salva múltiplos DataFrames em um arquivo Excel.

    Args:
        dfs: Dicionário {nome_sheet: dataframe}
        filepath: Caminho do arquivo
        config: Configuração global
    """
    formato_cfg = config.get('formato', {})

    with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        # Formatação
        workbook = writer.book
        formato_data = workbook.add_format({"num_format": "dd/mm/yyyy"})
        formato_texto = workbook.add_format({"align": "left"})

        for sheet_name in dfs.keys():
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column(0, 0, 12, formato_data)
            worksheet.set_column(1, 1, 14, formato_texto)

    logging.info(f"Arquivo Excel salvo: {os.path.abspath(filepath)}")


def salvar_csv(
    df: pd.DataFrame,
    filepath: str,
    config: Dict[str, Any]
) -> None:
    """
    Salva DataFrame em CSV com configurações específicas.

    Args:
        df: DataFrame a salvar
        filepath: Caminho do arquivo
        config: Configuração global
    """
    formato_cfg = config.get('formato', {})

    df.to_csv(
        filepath,
        index=False,
        encoding=formato_cfg.get('csv_encoding', 'utf-8-sig'),
        sep=formato_cfg.get('csv_separator', ';')
    )

    logging.info(f"Arquivo CSV salvo: {os.path.abspath(filepath)}")


# ============================================
# FUNÇÕES DE ANÁLISE E VARIAÇÕES
# ============================================

def calcular_tabela_variacoes(
    series: List[Tuple[date, pd.DataFrame]],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calcula tabela de variações para cada vencimento.

    Args:
        series: Lista de tuplas (data, dataframe) com dados históricos
        config: Configuração global

    Returns:
        DataFrame com colunas: Vencimento, Taxa Atual, D-1, Var D-1 (bps),
        D-7, Var D-7 (bps), D-30, Var D-30 (bps)
    """
    if not series:
        return pd.DataFrame()

    # Pegar os dados de hoje, D-1, D-7 e D-30
    data_hoje, df_hoje = series[0]

    # Mapear os períodos esperados
    dados_por_periodo = {0: (data_hoje, df_hoje)}

    for data_ref, df_ref in series[1:]:
        dias_diff = (data_hoje - data_ref).days
        # Mapear para os períodos mais próximos
        if 1 <= dias_diff <= 3 and 1 not in dados_por_periodo:
            dados_por_periodo[1] = (data_ref, df_ref)
        elif 5 <= dias_diff <= 10 and 7 not in dados_por_periodo:
            dados_por_periodo[7] = (data_ref, df_ref)
        elif 25 <= dias_diff <= 35 and 30 not in dados_por_periodo:
            dados_por_periodo[30] = (data_ref, df_ref)

    # Criar tabela de variações
    registros = []

    for _, row in df_hoje.iterrows():
        maturity = row["MaturityDate"]
        taxa_hoje = row["BidRate"]

        if pd.isna(taxa_hoje):
            continue

        registro = {
            "Vencimento": maturity,
            "Taxa Atual": taxa_hoje,
        }

        # D-1
        if 1 in dados_por_periodo:
            data_d1, df_d1 = dados_por_periodo[1]
            registro["Data D-1"] = data_d1
            taxa_d1 = df_d1[df_d1["MaturityDate"] == maturity]["BidRate"]
            if not taxa_d1.empty and not pd.isna(taxa_d1.iloc[0]):
                registro["Taxa D-1"] = taxa_d1.iloc[0]
                registro["Var D-1 (bps)"] = (taxa_hoje - taxa_d1.iloc[0]) * 10000
            else:
                registro["Taxa D-1"] = None
                registro["Var D-1 (bps)"] = None

        # D-7
        if 7 in dados_por_periodo:
            data_d7, df_d7 = dados_por_periodo[7]
            registro["Data D-7"] = data_d7
            taxa_d7 = df_d7[df_d7["MaturityDate"] == maturity]["BidRate"]
            if not taxa_d7.empty and not pd.isna(taxa_d7.iloc[0]):
                registro["Taxa D-7"] = taxa_d7.iloc[0]
                registro["Var D-7 (bps)"] = (taxa_hoje - taxa_d7.iloc[0]) * 10000
            else:
                registro["Taxa D-7"] = None
                registro["Var D-7 (bps)"] = None

        # D-30
        if 30 in dados_por_periodo:
            data_d30, df_d30 = dados_por_periodo[30]
            registro["Data D-30"] = data_d30
            taxa_d30 = df_d30[df_d30["MaturityDate"] == maturity]["BidRate"]
            if not taxa_d30.empty and not pd.isna(taxa_d30.iloc[0]):
                registro["Taxa D-30"] = taxa_d30.iloc[0]
                registro["Var D-30 (bps)"] = (taxa_hoje - taxa_d30.iloc[0]) * 10000
            else:
                registro["Taxa D-30"] = None
                registro["Var D-30 (bps)"] = None

        registros.append(registro)

    if not registros:
        return pd.DataFrame()

    df_variacoes = pd.DataFrame(registros)
    return df_variacoes


# ============================================
# FUNÇÕES DE GERAÇÃO DE HISTÓRICOS
# ============================================

def gerar_historicos_ntnb(
    maturities_info: List[Dict[str, Any]],
    data_hoje: date,
    nome_base: str,
    config: Dict[str, Any],
    output_dir: str = '.'
) -> Dict[str, pd.DataFrame]:
    """
    Gera histórico consolidado de 3 anos para todas as maturities.

    Args:
        maturities_info: Lista de dicionários com info das maturities
        data_hoje: Data de referência
        nome_base: Prefixo para nomes de arquivo
        config: Configuração global
        output_dir: Diretório de saída

    Returns:
        Dicionário {label: dataframe_historico_3anos}
    """
    registros_grafico = {}
    hist_cfg = config.get('historico', {})
    dias_3_anos = hist_cfg.get('dias_3_anos', 1095)

    # DataFrame consolidado
    df_consolidado = None

    for info in maturities_info:
        maturity_date = datetime.strptime(info["date"], "%Y-%m-%d").date()
        label = info["label"]
        bond_type = info.get("bond_type")

        logging.info(f"Coletando histórico para {label}...")

        historico_3_anos = coleta_historico_maturity(
            maturity_date, data_hoje, dias_3_anos, bond_type=bond_type, dir_dados=output_dir
        )

        if historico_3_anos.empty:
            logging.warning(f"Não foi possível obter histórico para {label}")
            continue

        # Adicionar ao consolidado
        if df_consolidado is None:
            df_consolidado = historico_3_anos[["Data"]].copy()

        # Renomear coluna BidRate com o label
        df_temp = historico_3_anos[["Data", "BidRate"]].copy()
        df_temp = df_temp.rename(columns={"BidRate": label})

        # Merge com o consolidado
        df_consolidado = df_consolidado.merge(df_temp, on="Data", how="outer")

        registros_grafico[label] = historico_3_anos

    # Salvar CSV consolidado
    if df_consolidado is not None and not df_consolidado.empty:
        df_consolidado = df_consolidado.sort_values("Data").reset_index(drop=True)

        # Formatar para display
        formato_cfg = config.get('formato', {})
        date_format = formato_cfg.get('date_format', '%d/%m/%Y')
        percent_decimals = formato_cfg.get('percent_decimals', 4)

        df_consolidado["Data"] = pd.to_datetime(df_consolidado["Data"]).dt.strftime(date_format)

        # Formatar colunas de yield
        for col in df_consolidado.columns:
            if col != "Data":
                df_consolidado[col] = df_consolidado[col].apply(
                    lambda x: format_percent(x, percent_decimals)
                )

        nome_csv_consolidado = os.path.join(
            output_dir,
            f"{nome_base}_historico_3anos_consolidado.csv"
        )
        salvar_csv(df_consolidado, nome_csv_consolidado, config)

    return registros_grafico


# ============================================
# FUNÇÃO PRINCIPAL
# ============================================

def main() -> None:
    """Função principal de execução do script."""
    # Carregar configuração
    try:
        config = load_config()
    except Exception as e:
        print(f"Erro ao carregar configuração: {e}")
        return

    # Setup logging
    setup_logging(config)

    logging.info("=" * 60)
    logging.info("Iniciando coleta de dados NTN-B")
    logging.info("=" * 60)

    # Configurações
    tz = ZoneInfo(config.get('timezone', 'America/Sao_Paulo'))
    formato_cfg = config.get('formato', {})
    graficos_cfg = config.get('graficos', {})

    # Data de referência
    hoje_ref = datetime.now(tz).date() - timedelta(days=1)

    # Criar diretório de saída se não existir
    output_dir = config.get('output_directory', '.')
    if output_dir != '.' and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logging.info(f"Diretório de saída criado: {output_dir}")

    try:
        data_hoje, df_hoje = get_ntnb_por_data_alvo(hoje_ref)
        logging.info(f"Data de referência: {data_hoje.strftime('%d/%m/%Y')}")
    except RuntimeError as e:
        logging.error(f"Erro ao buscar dados: {e}")
        return

    # Coletar séries temporais
    deslocamentos = config.get('deslocamentos', [7, 14, 30, 90, 180, 360, 720])

    series = [(data_hoje, df_hoje)]
    for dias in deslocamentos:
        alvo = data_hoje - timedelta(days=dias)
        try:
            data_k, df_k = get_ntnb_por_data_alvo(alvo)
            series.append((data_k, df_k))
            logging.info(f"Coletado: {data_k.strftime('%d/%m/%Y')} (T-{dias})")
        except RuntimeError as e:
            logging.warning(f"Não foi possível coletar T-{dias}: {e}")

    # Montar DataFrame comparativo
    df_final = None
    labels = []
    labels_display = {}

    date_format_file = formato_cfg.get('date_format_file', '%Y%m%d')
    date_format = formato_cfg.get('date_format', '%d/%m/%Y')

    for data_ref, df in series:
        if df.empty:
            continue

        lbl = data_ref.strftime(date_format_file)
        labels.append(lbl)
        labels_display[lbl] = data_ref.strftime(date_format)

        df_ren = df[["MaturityDate", "BidRate"]].rename(
            columns={"BidRate": f"BidRate_{lbl}"}
        )

        if df_final is None:
            df_final = df_ren
        else:
            df_final = df_final.merge(df_ren, on="MaturityDate", how="outer")

    if df_final is None:
        logging.error("Não foi possível montar o DataFrame final")
        return

    # Filtrar vencimentos
    ano_minimo = config.get('filtros', {}).get('ano_minimo_vencimento', 2030)
    mask_ano = df_final["MaturityDate"].dt.year >= ano_minimo
    df_final = df_final[mask_ano].sort_values("MaturityDate").reset_index(drop=True)

    if df_final.empty:
        logging.warning(f"Nenhum vencimento >= {ano_minimo} encontrado")

    # Calcular tabela de variações
    logging.info("Calculando variações...")
    df_variacoes = calcular_tabela_variacoes(series, config)

    # Salvar CSV de variações
    if not df_variacoes.empty:
        df_variacoes_display = df_variacoes.copy()
        percent_decimals = formato_cfg.get('percent_decimals', 4)

        # Formatar datas e percentuais
        df_variacoes_display["Vencimento"] = pd.to_datetime(
            df_variacoes_display["Vencimento"]
        ).dt.strftime(date_format)

        for col in df_variacoes_display.columns:
            if col.startswith("Taxa") and not col.endswith("(bps)"):
                df_variacoes_display[col] = df_variacoes_display[col].apply(
                    lambda x: format_percent(x, percent_decimals)
                )
            elif col.startswith("Data D-"):
                df_variacoes_display[col] = pd.to_datetime(
                    df_variacoes_display[col]
                ).dt.strftime(date_format)
            elif col.endswith("(bps)"):
                df_variacoes_display[col] = df_variacoes_display[col].apply(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )

        nome_csv_variacoes = os.path.join(output_dir, f"ntnb_variacoes_{data_hoje.strftime(date_format_file)}.csv")
        salvar_csv(df_variacoes_display, nome_csv_variacoes, config)

    # Salvar CSV comparativo
    nome_base = f"ntnb_comparativo_{data_hoje.strftime(date_format_file)}"
    nome_csv = os.path.join(output_dir, f"{nome_base}.csv")

    salvar_csv(df_final, nome_csv, config)

    # Gerar PDF
    logging.info("Gerando PDF comparativo...")
    nome_pdf = os.path.join(output_dir, f"{nome_base}.pdf")
    df_para_tabela = df_final.copy()

    with PdfPages(nome_pdf) as pdf:
        # Página 1: Tabela de Variações
        if not df_variacoes.empty:
            df_var_pdf = df_variacoes.copy()

            # Formatar para PDF
            df_var_pdf["Vencimento"] = pd.to_datetime(
                df_var_pdf["Vencimento"]
            )

            # Filtrar vencimentos >= 2030
            df_var_pdf = df_var_pdf[df_var_pdf["Vencimento"].dt.year >= 2030]

            df_var_pdf["Vencimento"] = df_var_pdf["Vencimento"].dt.strftime(date_format)

            # Selecionar colunas principais para o PDF
            colunas_pdf = ["Vencimento", "Taxa Atual"]
            if "Taxa D-1" in df_var_pdf.columns:
                colunas_pdf.extend(["Taxa D-1", "Var D-1 (bps)"])
            if "Taxa D-7" in df_var_pdf.columns:
                colunas_pdf.extend(["Taxa D-7", "Var D-7 (bps)"])
            if "Taxa D-30" in df_var_pdf.columns:
                colunas_pdf.extend(["Taxa D-30", "Var D-30 (bps)"])

            df_var_pdf = df_var_pdf[colunas_pdf]

            # Formatar valores
            for col in df_var_pdf.columns:
                if col.startswith("Taxa") and not col.endswith("(bps)"):
                    df_var_pdf[col] = df_var_pdf[col].apply(
                        lambda x: format_percent(x, percent_decimals)
                    )
                elif col.endswith("(bps)"):
                    df_var_pdf[col] = df_var_pdf[col].apply(
                        lambda x: f"{x:+.2f}" if pd.notna(x) else ""
                    )

            # Criar tabela de variações (orientação vertical)
            figsize_cfg = graficos_cfg.get('figsize', {})
            linhas_var = len(df_var_pdf.index)
            altura_var = max(
                figsize_cfg.get('tabela_min_height', 3),
                figsize_cfg.get('tabela_height_per_row', 0.4) * linhas_var + 1
            )
            largura_var = 8.5  # Largura padrão para orientação vertical

            fig_var, ax_var = plt.subplots(figsize=(largura_var, altura_var))
            ax_var.axis("off")
            ax_var.set_title(
                f"NTN-B - Variações (ref: {data_hoje.strftime(date_format)})",
                pad=20,
                fontsize=14,
                fontweight='bold'
            )

            tabela_var = ax_var.table(
                cellText=df_var_pdf.values,
                colLabels=df_var_pdf.columns,
                loc="center"
            )

            estilo_cfg = graficos_cfg.get('estilo', {})
            tabela_var.auto_set_font_size(False)
            tabela_var.set_fontsize(estilo_cfg.get('fontsize_table', 8))
            tabela_var.scale(1, estilo_cfg.get('table_scale', 1.4))

            # Formatar cabeçalho (azul com texto branco)
            for j in range(len(df_var_pdf.columns)):
                cell = tabela_var[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
                cell.set_text_props(ha='center')

            # Colorir células e centralizar
            for i in range(1, len(df_var_pdf) + 1):
                # Banded rows (cinza claro para linhas pares)
                if i % 2 == 0:
                    cor_base = '#F2F2F2'
                else:
                    cor_base = 'white'

                for j, col in enumerate(df_var_pdf.columns):
                    cell = tabela_var[(i, j)]
                    cell.set_text_props(ha='center')

                    if col.endswith("(bps)"):
                        try:
                            valor = df_variacoes.iloc[i-1][col]
                            if pd.notna(valor):
                                if valor > 0:
                                    cell.set_facecolor('#ffcccc')  # Vermelho claro
                                elif valor < 0:
                                    cell.set_facecolor('#ccffcc')  # Verde claro
                                else:
                                    cell.set_facecolor(cor_base)
                            else:
                                cell.set_facecolor(cor_base)
                        except:
                            cell.set_facecolor(cor_base)
                    else:
                        cell.set_facecolor(cor_base)

            pdf.savefig(fig_var, bbox_inches="tight")
            plt.close(fig_var)

        # Página 2: Tabela Comparativa
        if df_para_tabela.empty:
            fig_table, ax_table = plt.subplots(figsize=(8, 3))
            ax_table.axis("off")
            ax_table.text(0.5, 0.5, "Sem dados para exibir na tabela.",
                         ha="center", va="center")
            ax_table.set_title(
                f"NTN-B - Comparativo de Taxas (ref: {data_hoje.strftime(date_format)})",
                pad=20,
                fontsize=14,
                fontweight='bold'
            )
            pdf.savefig(fig_table, bbox_inches="tight")
            plt.close(fig_table)
        else:
            df_display = df_para_tabela.copy()
            df_display["MaturityDate"] = df_display["MaturityDate"].dt.strftime(date_format)

            # Renomear colunas
            col_renames = {"MaturityDate": "Vencimento"}
            for lbl, col_label in labels_display.items():
                col = f"BidRate_{lbl}"
                if col in df_display.columns:
                    col_renames[col] = col_label
            df_display = df_display.rename(columns=col_renames)

            # Formatar percentuais
            percent_decimals = formato_cfg.get('percent_decimals', 4)
            for coluna in df_display.columns:
                if coluna == "Vencimento":
                    df_display[coluna] = df_display[coluna].fillna("")
                else:
                    serie = df_display[coluna]
                    if is_numeric_dtype(serie):
                        df_display[coluna] = serie.apply(
                            lambda x: format_percent(x, percent_decimals)
                        )
                    else:
                        df_display[coluna] = serie.fillna("").astype(str)

            # Criar figura combinada: tabela + gráfico de curvas
            figsize_cfg = graficos_cfg.get('figsize', {})

            fig_combined = plt.figure(figsize=(8.5, 11))  # Página vertical

            # Subplot 1: Tabela comparativa (metade superior)
            ax_table = plt.subplot(2, 1, 1)
            ax_table.axis("off")
            ax_table.set_title(
                f"NTN-B - Comparativo de Taxas (ref: {data_hoje.strftime(date_format)})",
                pad=10,
                fontsize=12,
                fontweight='bold'
            )

            tabela = ax_table.table(
                cellText=df_display.values,
                colLabels=df_display.columns,
                loc="center"
            )

            estilo_cfg = graficos_cfg.get('estilo', {})
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(7)
            tabela.scale(1, 1.2)

            # Formatar cabeçalho (azul com texto branco)
            for j in range(len(df_display.columns)):
                cell = tabela[(0, j)]
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
                cell.set_text_props(ha='center')

            # Banded rows e centralizar valores
            for i in range(1, len(df_display) + 1):
                # Banded rows (cinza claro para linhas pares)
                if i % 2 == 0:
                    cor_linha = '#F2F2F2'
                else:
                    cor_linha = 'white'

                for j in range(len(df_display.columns)):
                    cell = tabela[(i, j)]
                    cell.set_facecolor(cor_linha)
                    cell.set_text_props(ha='center')

            # Subplot 2: Gráfico de curvas (metade inferior)
            ax_plot = plt.subplot(2, 1, 2)

            if not df_final.empty:
                estilo_cfg = graficos_cfg.get('estilo', {})
                fontsize_label = estilo_cfg.get('fontsize_label', 8)
                linewidth = estilo_cfg.get('linewidth', 2.0)

                # Filtrar apenas as colunas de 7, 14, 30 dias e atual
                periodos_desejados = []
                for i, (data_ref, _) in enumerate(series):
                    if i == 0:  # Atual
                        periodos_desejados.append((labels[i], "Atual"))
                    else:
                        dias_diff = (data_hoje - data_ref).days
                        if 5 <= dias_diff <= 10:  # ~7 dias (janela mais ampla)
                            periodos_desejados.append((labels[i], f"7 dias atrás"))
                        elif 12 <= dias_diff <= 18:  # ~14 dias (janela mais ampla para pegar finais de semana)
                            periodos_desejados.append((labels[i], f"14 dias atrás"))
                        elif 28 <= dias_diff <= 35:  # ~30 dias (janela mais ampla)
                            periodos_desejados.append((labels[i], f"30 dias atrás"))

                # Plotar apenas os períodos desejados
                for lbl, periodo_label in periodos_desejados:
                    col = f"BidRate_{lbl}"
                    if col in df_final.columns:
                        ax_plot.plot(
                            df_final["MaturityDate"],
                            df_final[col] * 100,
                            marker="o",
                            markersize=5,
                            linewidth=linewidth,
                            label=f"{periodo_label} ({labels_display.get(lbl, lbl)})",
                            alpha=0.8
                        )

                        # Anotar último ponto
                        serie = df_final[["MaturityDate", col]].dropna()
                        if not serie.empty:
                            x_ultimo = serie["MaturityDate"].iloc[-1]
                            y_ultimo = serie[col].iloc[-1] * 100
                            ax_plot.annotate(
                                f"{y_ultimo:.2f}%",
                                xy=(x_ultimo, y_ultimo),
                                xytext=(8, 0),
                                textcoords="offset points",
                                fontsize=fontsize_label,
                                ha="left",
                                va="center",
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                            )

                ax_plot.set_title(
                    f"NTN-B Bid Rate por Maturity Date",
                    fontsize=12,
                    fontweight='bold',
                    pad=15
                )
                ax_plot.set_xlabel("Data de Vencimento", fontsize=10)
                ax_plot.set_ylabel("Taxa Bid (%)", fontsize=10)
                ax_plot.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

                locator_cfg = graficos_cfg.get('date_locator', {})
                ax_plot.xaxis.set_major_locator(
                    mdates.AutoDateLocator(
                        minticks=locator_cfg.get('minticks', 6),
                        maxticks=locator_cfg.get('maxticks', 10)
                    )
                )
                ax_plot.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

                rotation = estilo_cfg.get('rotation_xaxis', 45)
                plt.setp(ax_plot.get_xticklabels(), rotation=rotation, ha='right', fontsize=9)
                plt.setp(ax_plot.get_yticklabels(), fontsize=9)
                ax_plot.legend(title="Datas", fontsize=8, loc='best', framealpha=0.9)

            fig_combined.tight_layout()
            pdf.savefig(fig_combined, bbox_inches="tight")
            plt.close(fig_combined)

        # Gerar históricos individuais antes de adicionar ao PDF
        logging.info("Gerando históricos individuais...")
        maturities_config = config.get('maturities', [])

        registros = gerar_historicos_ntnb(maturities_config, data_hoje, nome_base, config, output_dir)

        # Página 3: Gráfico histórico consolidado
        if registros:
            logging.info("Adicionando gráfico histórico ao PDF...")

            fig_hist = plt.figure(figsize=(11, 8.5))  # Página horizontal
            ax_hist = plt.gca()

            estilo_cfg = graficos_cfg.get('estilo', {})
            linewidth = estilo_cfg.get('linewidth', 1.5)
            fontsize_label = estilo_cfg.get('fontsize_label', 8)

            for label, df_hist in registros.items():
                if df_hist.empty:
                    continue

                ax_hist.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                            linewidth=linewidth, label=label)

                # Anotar último ponto
                serie_hist = df_hist.dropna(subset=["BidRate"])
                if not serie_hist.empty:
                    x_last = serie_hist["Data"].iloc[-1]
                    y_last = serie_hist["BidRate"].iloc[-1] * 100
                    ax_hist.annotate(
                        f"{y_last:.2f}%",
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=fontsize_label,
                        ha="left",
                        va="center"
                    )

            ax_hist.set_title(
                "Yield (%) - Histórico (Últimos 3 anos)",
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            ax_hist.set_xlabel("Data", fontsize=10)
            ax_hist.set_ylabel("Yield (%)", fontsize=10)
            ax_hist.grid(True, alpha=0.3)

            locator_cfg = graficos_cfg.get('date_locator', {})
            ax_hist.xaxis.set_major_locator(
                mdates.AutoDateLocator(
                    minticks=locator_cfg.get('minticks', 6),
                    maxticks=locator_cfg.get('maxticks', 10)
                )
            )
            ax_hist.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

            rotation = estilo_cfg.get('rotation_xaxis', 45)
            plt.setp(ax_hist.get_xticklabels(), rotation=rotation, ha="right")
            ax_hist.legend(fontsize=8, loc='best')
            fig_hist.tight_layout()

            pdf.savefig(fig_hist, bbox_inches="tight")
            plt.close(fig_hist)

            # Salvar PNG também
            nome_png = os.path.join(output_dir, f"{nome_base}_historico_ntnb.png")

            fig_png = plt.figure(figsize=(14, 6))
            ax_png = plt.gca()

            for label, df_hist in registros.items():
                if df_hist.empty:
                    continue
                ax_png.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                            linewidth=linewidth, label=label)
                serie_hist = df_hist.dropna(subset=["BidRate"])
                if not serie_hist.empty:
                    x_last = serie_hist["Data"].iloc[-1]
                    y_last = serie_hist["BidRate"].iloc[-1] * 100
                    ax_png.annotate(
                        f"{y_last:.2f}%",
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=fontsize_label,
                        ha="left",
                        va="center"
                    )

            ax_png.set_title("Yield (%) - Histórico (Últimos 3 anos)")
            ax_png.set_xlabel("Data")
            ax_png.set_ylabel("Yield (%)")
            ax_png.grid(False)
            ax_png.xaxis.set_major_locator(
                mdates.AutoDateLocator(
                    minticks=locator_cfg.get('minticks', 6),
                    maxticks=locator_cfg.get('maxticks', 10)
                )
            )
            ax_png.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))
            plt.setp(ax_png.get_xticklabels(), rotation=rotation, ha="right")
            ax_png.legend()
            plt.tight_layout()

            dpi = graficos_cfg.get('dpi', 200)
            plt.savefig(nome_png, dpi=dpi)
            logging.info(f"Gráfico histórico PNG salvo: {os.path.abspath(nome_png)}")
            plt.close(fig_png)

        # Página adicional: Gráfico histórico de 5 anos
        logging.info("Gerando históricos de 5 anos...")

        dias_5_anos = 1825  # 5 anos aproximadamente
        registros_5_anos = {}

        for info in maturities_config:
            maturity_date = datetime.strptime(info["date"], "%Y-%m-%d").date()
            label = info["label"]
            bond_type = info.get("bond_type")

            logging.info(f"Coletando histórico de 5 anos para {label}...")

            historico_5_anos = coleta_historico_maturity(
                maturity_date, data_hoje, dias_5_anos, bond_type=bond_type, dir_dados=output_dir
            )

            if historico_5_anos.empty:
                logging.warning(f"Não foi possível obter histórico de 5 anos para {label}")
                continue

            registros_5_anos[label] = historico_5_anos

        if registros_5_anos:
            logging.info("Adicionando gráfico histórico de 5 anos ao PDF...")

            fig_hist_5y = plt.figure(figsize=(11, 8.5))  # Página horizontal
            ax_hist_5y = plt.gca()

            estilo_cfg = graficos_cfg.get('estilo', {})
            linewidth = estilo_cfg.get('linewidth', 1.5)
            fontsize_label = estilo_cfg.get('fontsize_label', 8)

            for label, df_hist in registros_5_anos.items():
                if df_hist.empty:
                    continue

                ax_hist_5y.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                            linewidth=linewidth, label=label)

                # Anotar último ponto
                serie_hist = df_hist.dropna(subset=["BidRate"])
                if not serie_hist.empty:
                    x_last = serie_hist["Data"].iloc[-1]
                    y_last = serie_hist["BidRate"].iloc[-1] * 100
                    ax_hist_5y.annotate(
                        f"{y_last:.2f}%",
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=fontsize_label,
                        ha="left",
                        va="center"
                    )

            ax_hist_5y.set_title(
                "Yield (%) - Histórico (Últimos 5 anos)",
                fontsize=14,
                fontweight='bold',
                pad=20
            )
            ax_hist_5y.set_xlabel("Data", fontsize=10)
            ax_hist_5y.set_ylabel("Yield (%)", fontsize=10)
            ax_hist_5y.grid(True, alpha=0.3)

            locator_cfg = graficos_cfg.get('date_locator', {})
            ax_hist_5y.xaxis.set_major_locator(
                mdates.AutoDateLocator(
                    minticks=locator_cfg.get('minticks', 6),
                    maxticks=locator_cfg.get('maxticks', 10)
                )
            )
            ax_hist_5y.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

            rotation = estilo_cfg.get('rotation_xaxis', 45)
            plt.setp(ax_hist_5y.get_xticklabels(), rotation=rotation, ha="right")
            ax_hist_5y.legend(fontsize=8, loc='best')
            fig_hist_5y.tight_layout()

            pdf.savefig(fig_hist_5y, bbox_inches="tight")
            plt.close(fig_hist_5y)

            # Salvar PNG também
            nome_png_5y = os.path.join(output_dir, f"{nome_base}_historico_5anos_ntnb.png")

            fig_png_5y = plt.figure(figsize=(14, 6))
            ax_png_5y = plt.gca()

            for label, df_hist in registros_5_anos.items():
                if df_hist.empty:
                    continue
                ax_png_5y.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                            linewidth=linewidth, label=label)
                serie_hist = df_hist.dropna(subset=["BidRate"])
                if not serie_hist.empty:
                    x_last = serie_hist["Data"].iloc[-1]
                    y_last = serie_hist["BidRate"].iloc[-1] * 100
                    ax_png_5y.annotate(
                        f"{y_last:.2f}%",
                        xy=(x_last, y_last),
                        xytext=(6, 0),
                        textcoords="offset points",
                        fontsize=fontsize_label,
                        ha="left",
                        va="center"
                    )

            ax_png_5y.set_title("Yield (%) - Histórico (Últimos 5 anos)")
            ax_png_5y.set_xlabel("Data")
            ax_png_5y.set_ylabel("Yield (%)")
            ax_png_5y.grid(False)
            ax_png_5y.xaxis.set_major_locator(
                mdates.AutoDateLocator(
                    minticks=locator_cfg.get('minticks', 6),
                    maxticks=locator_cfg.get('maxticks', 10)
                )
            )
            ax_png_5y.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))
            plt.setp(ax_png_5y.get_xticklabels(), rotation=rotation, ha="right")
            ax_png_5y.legend()
            plt.tight_layout()

            dpi = graficos_cfg.get('dpi', 200)
            plt.savefig(nome_png_5y, dpi=dpi)
            logging.info(f"Gráfico histórico 5 anos PNG salvo: {os.path.abspath(nome_png_5y)}")
            plt.close(fig_png_5y)

    logging.info(f"Arquivo PDF salvo: {os.path.abspath(nome_pdf)}")

    # ============================================
    # GERAR RELATÓRIO HTML
    # ============================================
    logging.info("Gerando relatório HTML...")

    # Preparar DataFrames para HTML
    df_var_html = None
    df_comp_html = None

    # Preparar tabela de variações para HTML
    if not df_variacoes.empty:
        df_var_html = df_variacoes.copy()

        # Formatar para HTML
        df_var_html["Vencimento"] = pd.to_datetime(df_var_html["Vencimento"])
        df_var_html = df_var_html[df_var_html["Vencimento"].dt.year >= 2030]
        df_var_html["Vencimento"] = df_var_html["Vencimento"].dt.strftime(date_format)

        # Selecionar e formatar colunas (incluindo Taxa D-1 entre Taxa Atual e Var D-1)
        colunas_html = ["Vencimento", "Taxa Atual"]
        if "Taxa D-1" in df_var_html.columns:
            colunas_html.append("Taxa D-1")
        if "Var D-1 (bps)" in df_var_html.columns:
            colunas_html.append("Var D-1 (bps)")
        if "Taxa D-7" in df_var_html.columns:
            colunas_html.append("Taxa D-7")
        if "Var D-7 (bps)" in df_var_html.columns:
            colunas_html.append("Var D-7 (bps)")
        if "Taxa D-30" in df_var_html.columns:
            colunas_html.append("Taxa D-30")
        if "Var D-30 (bps)" in df_var_html.columns:
            colunas_html.append("Var D-30 (bps)")

        df_var_html = df_var_html[colunas_html]

        # Formatar valores
        for col in df_var_html.columns:
            if col.startswith("Taxa") and not col.endswith("(bps)"):
                df_var_html[col] = df_var_html[col].apply(
                    lambda x: format_percent(x, percent_decimals)
                )
            elif col.endswith("(bps)"):
                df_var_html[col] = df_var_html[col].apply(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else ""
                )

    # Preparar tabela comparativa para HTML
    if not df_final.empty:
        df_comp_html = df_final.copy()
        df_comp_html["MaturityDate"] = df_comp_html["MaturityDate"].dt.strftime(date_format)

        # Renomear colunas
        col_renames = {"MaturityDate": "Vencimento"}
        for lbl, col_label in labels_display.items():
            col = f"BidRate_{lbl}"
            if col in df_comp_html.columns:
                col_renames[col] = col_label
        df_comp_html = df_comp_html.rename(columns=col_renames)

        # Formatar percentuais
        for coluna in df_comp_html.columns:
            if coluna == "Vencimento":
                df_comp_html[coluna] = df_comp_html[coluna].fillna("")
            else:
                serie = df_comp_html[coluna]
                if is_numeric_dtype(serie):
                    df_comp_html[coluna] = serie.apply(
                        lambda x: format_percent(x, percent_decimals)
                    )
                else:
                    df_comp_html[coluna] = serie.fillna("").astype(str)

    # Criar figuras para HTML (reutilizando lógica do PDF mas criando novas figuras)

    # Figura 1: Gráfico de curvas
    fig_curvas_html = None
    if not df_final.empty:
        fig_curvas_html = plt.figure(figsize=(12, 6))
        ax_curvas = plt.gca()

        estilo_cfg = graficos_cfg.get('estilo', {})
        fontsize_label = estilo_cfg.get('fontsize_label', 8)
        linewidth = estilo_cfg.get('linewidth', 2.0)

        # Filtrar períodos desejados (atual, 7, 14, 30 dias)
        periodos_desejados = []
        for i, (data_ref, _) in enumerate(series):
            if i == 0:
                periodos_desejados.append((labels[i], "Atual"))
            else:
                dias_diff = (data_hoje - data_ref).days
                if 5 <= dias_diff <= 10:
                    periodos_desejados.append((labels[i], f"7 dias atrás"))
                elif 12 <= dias_diff <= 18:
                    periodos_desejados.append((labels[i], f"14 dias atrás"))
                elif 28 <= dias_diff <= 35:
                    periodos_desejados.append((labels[i], f"30 dias atrás"))

        for lbl, periodo_label in periodos_desejados:
            col = f"BidRate_{lbl}"
            if col in df_final.columns:
                ax_curvas.plot(
                    df_final["MaturityDate"],
                    df_final[col] * 100,
                    marker="o",
                    markersize=5,
                    linewidth=linewidth,
                    label=f"{periodo_label} ({labels_display.get(lbl, lbl)})",
                    alpha=0.8
                )

                # Anotar último ponto
                serie = df_final[["MaturityDate", col]].dropna()
                if not serie.empty:
                    x_ultimo = serie["MaturityDate"].iloc[-1]
                    y_ultimo = serie[col].iloc[-1] * 100
                    ax_curvas.annotate(
                        f"{y_ultimo:.2f}%",
                        xy=(x_ultimo, y_ultimo),
                        xytext=(8, 0),
                        textcoords="offset points",
                        fontsize=fontsize_label,
                        ha="left",
                        va="center",
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='none')
                    )

        ax_curvas.set_title("NTN-B Bid Rate por Maturity Date", fontsize=14, fontweight='bold')
        ax_curvas.set_xlabel("Data de Vencimento", fontsize=11)
        ax_curvas.set_ylabel("Taxa Bid (%)", fontsize=11)
        ax_curvas.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        locator_cfg = graficos_cfg.get('date_locator', {})
        ax_curvas.xaxis.set_major_locator(
            mdates.AutoDateLocator(
                minticks=locator_cfg.get('minticks', 6),
                maxticks=locator_cfg.get('maxticks', 10)
            )
        )
        ax_curvas.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

        rotation = estilo_cfg.get('rotation_xaxis', 45)
        plt.setp(ax_curvas.get_xticklabels(), rotation=rotation, ha='right')
        ax_curvas.legend(title="Datas", fontsize=9, loc='best', framealpha=0.9)
        fig_curvas_html.tight_layout()

    # Figura 2: Histórico 3 anos
    fig_hist_3y_html = None
    if registros:
        fig_hist_3y_html = plt.figure(figsize=(14, 6))
        ax_3y = plt.gca()

        estilo_cfg = graficos_cfg.get('estilo', {})
        linewidth = estilo_cfg.get('linewidth', 1.5)
        fontsize_label = estilo_cfg.get('fontsize_label', 8)

        for label, df_hist in registros.items():
            if df_hist.empty:
                continue

            ax_3y.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                      linewidth=linewidth, label=label)

            # Anotar último ponto
            serie_hist = df_hist.dropna(subset=["BidRate"])
            if not serie_hist.empty:
                x_last = serie_hist["Data"].iloc[-1]
                y_last = serie_hist["BidRate"].iloc[-1] * 100
                ax_3y.annotate(
                    f"{y_last:.2f}%",
                    xy=(x_last, y_last),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=fontsize_label,
                    ha="left",
                    va="center"
                )

        ax_3y.set_title("Yield (%) - Histórico (Últimos 3 anos)", fontsize=14, fontweight='bold')
        ax_3y.set_xlabel("Data", fontsize=11)
        ax_3y.set_ylabel("Yield (%)", fontsize=11)
        ax_3y.grid(True, alpha=0.3)

        locator_cfg = graficos_cfg.get('date_locator', {})
        ax_3y.xaxis.set_major_locator(
            mdates.AutoDateLocator(
                minticks=locator_cfg.get('minticks', 6),
                maxticks=locator_cfg.get('maxticks', 10)
            )
        )
        ax_3y.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

        rotation = estilo_cfg.get('rotation_xaxis', 45)
        plt.setp(ax_3y.get_xticklabels(), rotation=rotation, ha="right")
        ax_3y.legend(fontsize=9, loc='best')
        fig_hist_3y_html.tight_layout()

    # Figura 3: Histórico 5 anos
    fig_hist_5y_html = None
    if registros_5_anos:
        fig_hist_5y_html = plt.figure(figsize=(14, 6))
        ax_5y = plt.gca()

        estilo_cfg = graficos_cfg.get('estilo', {})
        linewidth = estilo_cfg.get('linewidth', 1.5)
        fontsize_label = estilo_cfg.get('fontsize_label', 8)

        for label, df_hist in registros_5_anos.items():
            if df_hist.empty:
                continue

            ax_5y.plot(df_hist["Data"], df_hist["BidRate"] * 100,
                      linewidth=linewidth, label=label)

            # Anotar último ponto
            serie_hist = df_hist.dropna(subset=["BidRate"])
            if not serie_hist.empty:
                x_last = serie_hist["Data"].iloc[-1]
                y_last = serie_hist["BidRate"].iloc[-1] * 100
                ax_5y.annotate(
                    f"{y_last:.2f}%",
                    xy=(x_last, y_last),
                    xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=fontsize_label,
                    ha="left",
                    va="center"
                )

        ax_5y.set_title("Yield (%) - Histórico (Últimos 5 anos)", fontsize=14, fontweight='bold')
        ax_5y.set_xlabel("Data", fontsize=11)
        ax_5y.set_ylabel("Yield (%)", fontsize=11)
        ax_5y.grid(True, alpha=0.3)

        locator_cfg = graficos_cfg.get('date_locator', {})
        ax_5y.xaxis.set_major_locator(
            mdates.AutoDateLocator(
                minticks=locator_cfg.get('minticks', 6),
                maxticks=locator_cfg.get('maxticks', 10)
            )
        )
        ax_5y.xaxis.set_major_formatter(FuncFormatter(tick_mes_ano))

        rotation = estilo_cfg.get('rotation_xaxis', 45)
        plt.setp(ax_5y.get_xticklabels(), rotation=rotation, ha="right")
        ax_5y.legend(fontsize=9, loc='best')
        fig_hist_5y_html.tight_layout()

    # ============================================
    # PROCESSAR DI FUTURO (se habilitado)
    # ============================================
    di_cfg = config.get('di_futuro', {})
    di_habilitado = di_cfg.get('enabled', False)

    df_di_var_html = None
    fig_di_curva_html = None
    fig_di_hist_12m_html = None
    fig_di_hist_3y_html = None
    fig_di_hist_5y_html = None

    if di_habilitado:
        logging.info("=" * 60)
        logging.info("Processando DI Futuro")
        logging.info("=" * 60)

        try:
            # 1. Atualizar histórico de DI Futuro (coleta da API + salva CSV)
            meses_venc = di_cfg.get('meses_vencimento', [1])
            df_di_historico = pydifuturo.atualizar_historico(meses_vencimento=meses_venc, dir_dados=output_dir)

            if len(df_di_historico) > 0:
                # 2. Calcular variações
                deslocamentos_di = di_cfg.get('deslocamentos', [1, 7, 30])
                df_di_var = pydifuturo_relatorio.calcular_variacoes_di(df_di_historico, deslocamentos_di)

                # 3. Filtrar vencimentos para exibição (27, 28, 29, 30, 32, 35)
                vencimentos_graficos = di_cfg.get('vencimentos_graficos', [])

                # Se não especificou vencimentos, usa filtro padrão
                if not vencimentos_graficos:
                    vencimentos_graficos = ['DI1F27', 'DI1F28', 'DI1F29', 'DI1F30', 'DI1F32', 'DI1F35']

                # Filtra a tabela de variações pelos vencimentos configurados
                if vencimentos_graficos:
                    df_di_var = df_di_var[df_di_var['Ticker'].isin(vencimentos_graficos)]

                # 4. Preparar tabela HTML
                df_di_var_html = pydifuturo_relatorio.preparar_tabela_html_di(df_di_var)

                # 5. Gerar gráficos
                graficos_di_cfg = di_cfg.get('graficos', {})

                # Gráfico de Curva
                try:
                    fig_di_curva_html = pydifuturo_relatorio.gerar_grafico_curva_di(
                        df_di_historico,
                        data_hoje,
                        deslocamentos=deslocamentos_di,
                        figsize=tuple(graficos_cfg.get('figsize', {}).get('grafico_curva', [11, 7])),
                        dpi=graficos_cfg.get('dpi', 150)
                    )
                    logging.info("Gráfico de curva DI gerado")
                except Exception as e:
                    logging.warning(f"Erro ao gerar gráfico de curva DI: {e}")

                # Gráfico Histórico 12 Meses
                try:
                    dias_12m = graficos_di_cfg.get('historico_12_meses', 365)
                    fig_di_hist_12m_html = pydifuturo_relatorio.gerar_grafico_historico_di(
                        df_di_historico,
                        vencimentos_graficos,
                        dias_12m,
                        titulo="Histórico de Taxas DI Futuro - 12 Meses",
                        figsize=tuple(graficos_cfg.get('figsize', {}).get('grafico_historico', [14, 6])),
                        dpi=graficos_cfg.get('dpi', 150)
                    )
                    logging.info("Gráfico histórico 12 meses DI gerado")
                except Exception as e:
                    logging.warning(f"Erro ao gerar gráfico 12 meses DI: {e}")

                # Gráfico Histórico 3 Anos
                try:
                    dias_3y = graficos_di_cfg.get('historico_3_anos', 1095)
                    fig_di_hist_3y_html = pydifuturo_relatorio.gerar_grafico_historico_di(
                        df_di_historico,
                        vencimentos_graficos,
                        dias_3y,
                        titulo="Histórico de Taxas DI Futuro - 3 Anos",
                        figsize=tuple(graficos_cfg.get('figsize', {}).get('grafico_historico', [14, 6])),
                        dpi=graficos_cfg.get('dpi', 150)
                    )
                    logging.info("Gráfico histórico 3 anos DI gerado")
                except Exception as e:
                    logging.warning(f"Erro ao gerar gráfico 3 anos DI: {e}")

                # Gráfico Histórico 5 Anos
                try:
                    dias_5y = graficos_di_cfg.get('historico_5_anos', 1825)
                    fig_di_hist_5y_html = pydifuturo_relatorio.gerar_grafico_historico_di(
                        df_di_historico,
                        vencimentos_graficos,
                        dias_5y,
                        titulo="Histórico de Taxas DI Futuro - 5 Anos",
                        figsize=tuple(graficos_cfg.get('figsize', {}).get('grafico_historico', [14, 6])),
                        dpi=graficos_cfg.get('dpi', 150)
                    )
                    logging.info("Gráfico histórico 5 anos DI gerado")
                except Exception as e:
                    logging.warning(f"Erro ao gerar gráfico 5 anos DI: {e}")

                logging.info(f"DI Futuro processado com sucesso ({len(df_di_var)} contratos)")
            else:
                logging.warning("Nenhum dado DI Futuro coletado")

        except Exception as e:
            logging.error(f"Erro ao processar DI Futuro: {e}")
    else:
        logging.info("DI Futuro desabilitado (configure 'di_futuro.enabled: true' para habilitar)")

    # ============================================
    # ANÁLISES AVANÇADAS
    # ============================================

    df_bei = None
    df_steepness_di = None
    df_steepness_ntnb = None
    df_stats_di = None
    df_focus = None

    analises_cfg = config.get('analises_avancadas', {})
    analises_habilitado = analises_cfg.get('enabled', False)

    if analises_habilitado:
        logging.info("=" * 60)
        logging.info("Processando Análises Avançadas")
        logging.info("=" * 60)

        # 1. Breakeven Inflation
        breakeven_cfg = analises_cfg.get('breakeven', {})
        if breakeven_cfg.get('enabled', False):
            try:
                logging.info("Calculando Breakeven Inflation...")
                pares = breakeven_cfg.get('pares_vencimento', [])

                # Consolida dados históricos NTN-B dos registros
                if len(pares) > 0 and 'registros' in locals() and 'df_di_historico' in locals():
                    # Combina dados de todos os vencimentos
                    dfs_ntnb = []
                    for label, df_hist in registros.items():
                        if not df_hist.empty and 'BidRate' in df_hist.columns:
                            # Extrai vencimento do label
                            import re
                            match = re.search(r'(\d{2}/\d{2}/\d{4})', label)
                            if match:
                                venc_str = match.group(1)
                                # Converte para formato YYYY-MM-DD
                                venc_date = datetime.strptime(venc_str, '%d/%m/%Y').strftime('%Y-%m-%d')

                                df_temp = df_hist[['Data', 'BidRate']].copy()
                                df_temp['Vencimento'] = venc_date
                                dfs_ntnb.append(df_temp)

                    if len(dfs_ntnb) > 0:
                        df_ntnb_consolidado = pd.concat(dfs_ntnb, ignore_index=True)

                        df_bei = pyanalise.calcular_breakeven_inflation(
                            df_ntnb_consolidado,
                            df_di_historico,
                            pares
                        )
                        if len(df_bei) > 0:
                            logging.info(f"Breakeven calculado para {len(pares)} pares")
            except Exception as e:
                logging.error(f"Erro ao calcular Breakeven: {e}")

        # 2. Steepness
        steepness_cfg = analises_cfg.get('steepness', {})
        if steepness_cfg.get('enabled', False):
            try:
                # Steepness DI
                di_steep_cfg = steepness_cfg.get('di', {})
                if 'df_di_historico' in locals() and len(df_di_historico) > 0:
                    logging.info("Calculando Steepness DI...")
                    df_steepness_di = pyanalise.calcular_steepness(
                        df_di_historico,
                        di_steep_cfg.get('ticker_curto', 'DI1F27'),
                        di_steep_cfg.get('ticker_longo', 'DI1F35'),
                        coluna_taxa='Taxa'
                    )
                    if len(df_steepness_di) > 0:
                        logging.info(f"Steepness DI: {df_steepness_di['Steepness'].iloc[-1]:.0f} bps")

                # Steepness NTN-B
                ntnb_steep_cfg = steepness_cfg.get('ntnb', {})
                if 'registros' in locals() and len(registros) > 0:
                    logging.info("Calculando Steepness NTN-B...")

                    # Consolida dados NTN-B
                    dfs_ntnb = []
                    for label, df_hist in registros.items():
                        if not df_hist.empty and 'BidRate' in df_hist.columns:
                            import re
                            match = re.search(r'(\d{2}/\d{2}/\d{4})', label)
                            if match:
                                venc_str = match.group(1)
                                venc_date = datetime.strptime(venc_str, '%d/%m/%Y').strftime('%Y-%m-%d')
                                df_temp = df_hist[['Data', 'BidRate']].copy()
                                df_temp['Vencimento'] = venc_date
                                dfs_ntnb.append(df_temp)

                    if len(dfs_ntnb) > 0:
                        df_ntnb_consolidado = pd.concat(dfs_ntnb, ignore_index=True)

                        df_steepness_ntnb = pyanalise.calcular_steepness(
                            df_ntnb_consolidado,
                            ntnb_steep_cfg.get('venc_curto', '2030-08-15'),
                            ntnb_steep_cfg.get('venc_longo', '2060-08-15'),
                            coluna_taxa='BidRate'
                        )
                        if len(df_steepness_ntnb) > 0:
                            logging.info(f"Steepness NTN-B: {df_steepness_ntnb['Steepness'].iloc[-1]:.0f} bps")

            except Exception as e:
                logging.error(f"Erro ao calcular Steepness: {e}")

        # 3. Estatísticas Históricas
        stats_cfg = analises_cfg.get('estatisticas', {})
        if stats_cfg.get('enabled', False):
            try:
                if 'df_di_historico' in locals() and len(df_di_historico) > 0:
                    logging.info("Calculando Estatísticas Históricas DI...")
                    periodos = stats_cfg.get('periodos', [30, 90, 365])
                    df_stats_di = pyanalise.calcular_estatisticas_historicas(
                        df_di_historico,
                        periodos=periodos,
                        coluna_taxa='Taxa'
                    )
                    if len(df_stats_di) > 0:
                        logging.info(f"Estatísticas calculadas para {len(df_stats_di)} tickers")
            except Exception as e:
                logging.error(f"Erro ao calcular Estatísticas: {e}")

    # 4. Expectativas FOCUS BCB
    focus_cfg = config.get('focus_bcb', {})
    if focus_cfg.get('enabled', False):
        try:
            logging.info("Coletando Expectativas FOCUS BCB...")
            resumo_focus = pyfocus.obter_resumo_focus(data_referencia=data_hoje, cache_dir=output_dir)

            if resumo_focus:
                df_focus = pyfocus.formatar_tabela_focus(resumo_focus)
                if len(df_focus) > 0:
                    logging.info(f"Expectativas FOCUS coletadas: {len(df_focus)} registros")
        except Exception as e:
            logging.error(f"Erro ao coletar FOCUS: {e}")

    # ============================================
    # 5. US Treasuries
    # ============================================
    df_ust_variacoes = None
    fig_ust_curva_html = None
    fig_ust_hist_10y_html = None

    ust_cfg = config.get('treasuries', {})
    if ust_cfg.get('enabled', False):
        try:
            logging.info("=" * 60)
            logging.info("Coletando dados de US Treasuries")
            logging.info("=" * 60)

            # Buscar dados do FRED
            df_ust = pyust_data.buscar_treasuries(config)

            if df_ust is not None and not df_ust.empty:
                logging.info(f"Dados de Treasuries coletados: {len(df_ust)} registros")

                # Calcular variações
                treasury = pyust_data.TreasuryData(ust_cfg.get('fred_api_key'))
                df_ust_var = treasury.calcular_variacoes(df_ust, periodos=[1, 5, 21])

                # Preparar tabela de variações para HTML
                ultimo_dia = df_ust.iloc[-1]
                var_1d = df_ust_var.iloc[-1] if len(df_ust_var) > 0 else None
                var_1w = df_ust_var.iloc[-5] if len(df_ust_var) >= 5 else None
                var_1m = df_ust_var.iloc[-21] if len(df_ust_var) >= 21 else None

                # Montar DataFrame de variações
                tenores = ["2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
                variacoes_data = []

                for tenor in tenores:
                    if tenor in df_ust.columns:
                        row = {
                            "Vencimento": tenor,
                            "Taxa Atual": f"{ultimo_dia[tenor]:.2f}%"
                        }

                        # Variação 1D
                        if var_1d is not None and f"{tenor}_1D" in df_ust_var.columns:
                            var = df_ust_var[f"{tenor}_1D"].iloc[-1]
                            row["Taxa D-1"] = f"{(ultimo_dia[tenor] - var):.2f}%" if pd.notna(var) else ""
                            row["Var D-1 (bps)"] = f"{(var * 100):+.2f}" if pd.notna(var) else ""

                        # Variação 1W
                        if var_1w is not None and f"{tenor}_1W" in df_ust_var.columns:
                            var = df_ust_var[f"{tenor}_1W"].iloc[-5] if len(df_ust_var) >= 5 else None
                            row["Taxa D-7"] = f"{(ultimo_dia[tenor] - var):.2f}%" if pd.notna(var) else ""
                            row["Var D-7 (bps)"] = f"{(var * 100):+.2f}" if pd.notna(var) else ""

                        # Variação 1M
                        if var_1m is not None and f"{tenor}_1M" in df_ust_var.columns:
                            var = df_ust_var[f"{tenor}_1M"].iloc[-21] if len(df_ust_var) >= 21 else None
                            row["Taxa D-30"] = f"{(ultimo_dia[tenor] - var):.2f}%" if pd.notna(var) else ""
                            row["Var D-30 (bps)"] = f"{(var * 100):+.2f}" if pd.notna(var) else ""

                        variacoes_data.append(row)

                df_ust_variacoes = pd.DataFrame(variacoes_data)

                # Gerar gráfico de curvas
                fig_ust_curva_html = plt.figure(figsize=(10, 6))
                ax = fig_ust_curva_html.add_subplot(111)

                # Plotar curva atual e períodos anteriores
                periodos_plot = [0, 7, 30]  # Atual, 7 dias, 30 dias
                labels_plot = ["Atual", "7 dias atrás", "30 dias atrás"]
                cores_curva = ['#1f77b4', '#ff7f0e', '#2ca02c']

                for i, (periodo, label) in enumerate(zip(periodos_plot, labels_plot)):
                    idx = -1 - periodo if periodo > 0 else -1
                    if abs(idx) <= len(df_ust):
                        row = df_ust.iloc[idx]
                        tenores_x = ["2Y", "5Y", "7Y", "10Y", "20Y", "30Y"]
                        valores_y = [row[t] if t in row and pd.notna(row[t]) else None for t in tenores_x]

                        ax.plot(tenores_x, valores_y, marker='o', label=label, linewidth=2, markersize=6, color=cores_curva[i], alpha=0.85)

                        # Adicionar label no último ponto para todas as curvas
                        for j, (tenor, valor) in enumerate(zip(tenores_x, valores_y)):
                            if valor is not None and j == len(tenores_x) - 1:  # Último ponto (30Y)
                                ax.annotate(f'{valor:.2f}%',
                                           xy=(tenor, valor),
                                           xytext=(8, 0),
                                           textcoords='offset points',
                                           fontsize=9,
                                           fontweight='bold',
                                           va='center',
                                           color=cores_curva[i],
                                           bbox=dict(boxstyle='round,pad=0.4',
                                                   facecolor='white',
                                                   alpha=0.9,
                                                   edgecolor=cores_curva[i],
                                                   linewidth=1.5))

                ax.set_title("Curva de Treasuries US", fontsize=14, fontweight='bold')
                ax.set_xlabel("Tenor", fontsize=10)
                ax.set_ylabel("Taxa (%)", fontsize=10)
                ax.grid(True, alpha=0.3, linestyle='--')
                ax.legend(loc='best', fontsize=10, framealpha=0.9)

                # Gerar gráfico histórico com múltiplos vencimentos
                tenores_hist = ["5Y", "7Y", "10Y", "20Y", "30Y"]
                tenores_disponiveis = [t for t in tenores_hist if t in df_ust.columns]

                if len(tenores_disponiveis) > 0:
                    fig_ust_hist_10y_html = plt.figure(figsize=(14, 7))
                    ax = fig_ust_hist_10y_html.add_subplot(111)

                    # Cores para cada tenor
                    cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

                    for i, tenor in enumerate(tenores_disponiveis):
                        # Plotar linha
                        ax.plot(df_ust.index, df_ust[tenor],
                               linewidth=2.5,
                               color=cores[i % len(cores)],
                               label=tenor,
                               alpha=0.85)

                        # Adicionar label do último valor
                        ultimo_idx = df_ust[tenor].last_valid_index()
                        if ultimo_idx is not None:
                            ultimo_valor = df_ust.loc[ultimo_idx, tenor]
                            ax.annotate(f'{ultimo_valor:.2f}%',
                                       xy=(ultimo_idx, ultimo_valor),
                                       xytext=(8, 0),
                                       textcoords='offset points',
                                       fontsize=9,
                                       fontweight='bold',
                                       va='center',
                                       color=cores[i % len(cores)],
                                       bbox=dict(boxstyle='round,pad=0.4',
                                               facecolor='white',
                                               alpha=0.8,
                                               edgecolor=cores[i % len(cores)],
                                               linewidth=1.5))

                    ax.set_title("Histórico - US Treasuries", fontsize=14, fontweight='bold')
                    ax.set_xlabel("Data", fontsize=10)
                    ax.set_ylabel("Taxa (%)", fontsize=10)
                    ax.grid(True, alpha=0.3, linestyle='--')
                    ax.legend(loc='best', fontsize=10, framealpha=0.9)

                    # Formatar eixo x
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

                logging.info("Gráficos de Treasuries gerados com sucesso")

        except Exception as e:
            logging.error(f"Erro ao processar Treasuries: {e}")

    # ============================================
    # Gerar HTML (NTN-B + DI Futuro + Análises + Treasuries)
    # ============================================
    html_content = gerar_html_relatorio(
        df_var_html if df_var_html is not None else pd.DataFrame(),
        df_comp_html if df_comp_html is not None else pd.DataFrame(),
        fig_curvas_html,
        fig_hist_3y_html,
        fig_hist_5y_html,
        data_hoje,
        config,
        # Dados DI Futuro
        df_di_variacoes=df_di_var_html,
        fig_di_curva=fig_di_curva_html,
        fig_di_hist_12m=fig_di_hist_12m_html,
        fig_di_hist_3y=fig_di_hist_3y_html,
        fig_di_hist_5y=fig_di_hist_5y_html,
        # Análises Avançadas
        df_bei=df_bei,
        df_steepness_di=df_steepness_di,
        df_steepness_ntnb=df_steepness_ntnb,
        df_stats_di=df_stats_di,
        df_focus=df_focus,
        # Treasuries US
        df_ust_variacoes=df_ust_variacoes,
        fig_ust_curva=fig_ust_curva_html,
        fig_ust_hist_10y=fig_ust_hist_10y_html
    )

    # Salvar HTML
    nome_html = os.path.join(output_dir, f"{nome_base}.html")
    with open(nome_html, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logging.info(f"Arquivo HTML salvo: {os.path.abspath(nome_html)}")

    # Fechar figuras HTML (NTN-B)
    if fig_curvas_html is not None:
        plt.close(fig_curvas_html)
    if fig_hist_3y_html is not None:
        plt.close(fig_hist_3y_html)
    if fig_hist_5y_html is not None:
        plt.close(fig_hist_5y_html)

    # Fechar figuras HTML (DI Futuro)
    if fig_di_curva_html is not None:
        plt.close(fig_di_curva_html)
    if fig_di_hist_12m_html is not None:
        plt.close(fig_di_hist_12m_html)
    if fig_di_hist_3y_html is not None:
        plt.close(fig_di_hist_3y_html)
    if fig_di_hist_5y_html is not None:
        plt.close(fig_di_hist_5y_html)

    # Fechar figuras HTML (Treasuries)
    if fig_ust_curva_html is not None:
        plt.close(fig_ust_curva_html)
    if fig_ust_hist_10y_html is not None:
        plt.close(fig_ust_hist_10y_html)

    # ============================================
    # ENVIAR EMAIL
    # ============================================
    email_cfg = config.get('email', {})
    if email_cfg.get('enabled', False):
        logging.info("Enviando relatório por email...")

        # Montar assunto do email
        subject_template = email_cfg.get('subject', 'Relatório NTN-B - {date}')
        subject = subject_template.format(date=data_hoje.strftime(date_format))

        # Enviar email
        sucesso = enviar_email_html(html_content, subject, config)

        if not sucesso:
            logging.warning("Não foi possível enviar o email. Verifique as configurações.")
    else:
        logging.info("Envio de email desabilitado. Para habilitar, configure 'email.enabled: true' no config_ntnb.yaml")

    logging.info("=" * 60)
    logging.info("Processo concluído com sucesso!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
