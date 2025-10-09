"""
Módulo de análise macroeconômica via ChatGPT/OpenAI.

Integra com a API do OpenAI para gerar análises contextualizadas das curvas de juros
brasileiras relacionando com dados macroeconômicos.

Autor: Claude Code
Data: 2025-10-03
"""

import logging
from typing import Dict, Optional, Any, List
import pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def preparar_resumo_dados(
    df_variacoes_ntnb: pd.DataFrame,
    df_variacoes_di: pd.DataFrame,
    df_steepness_di: Optional[pd.DataFrame],
    df_steepness_ntnb: Optional[pd.DataFrame],
    df_bei: Optional[pd.DataFrame],
    df_focus: Optional[pd.DataFrame]
) -> str:
    """
    Prepara resumo estruturado dos dados para envio ao GPT.

    Args:
        df_variacoes_ntnb: Variações NTN-B
        df_variacoes_di: Variações DI Futuro
        df_steepness_di: Steepness DI
        df_steepness_ntnb: Steepness NTN-B
        df_bei: Breakeven Inflation
        df_focus: Expectativas FOCUS

    Returns:
        String formatada com resumo dos dados
    """
    resumo = []

    # Variações NTN-B
    if not df_variacoes_ntnb.empty:
        resumo.append("## Variações NTN-B (Títulos Indexados à Inflação)")
        try:
            taxa_media = df_variacoes_ntnb['Taxa Atual'].mean()
            resumo.append(f"Taxa atual média: {taxa_media:.2f}%")
        except:
            pass

        try:
            if 'Var D-1 (bps)' in df_variacoes_ntnb.columns:
                var_d1 = df_variacoes_ntnb['Var D-1 (bps)'].dropna()
                if len(var_d1) > 0:
                    resumo.append(f"Variação média D-1: {var_d1.mean():.1f} bps")
        except:
            pass

        try:
            if 'Var D-30 (bps)' in df_variacoes_ntnb.columns:
                var_d30 = df_variacoes_ntnb['Var D-30 (bps)'].dropna()
                if len(var_d30) > 0:
                    resumo.append(f"Variação média D-30: {var_d30.mean():.1f} bps")
        except:
            pass

    # Variações DI - Simplificado para evitar erros
    if not df_variacoes_di.empty:
        resumo.append("\n## Variações DI Futuro (Taxas Nominais)")
        # Apenas conta número de contratos
        resumo.append(f"Contratos analisados: {len(df_variacoes_di)}")

    # Steepness
    if df_steepness_di is not None and len(df_steepness_di) > 0:
        steep_di = df_steepness_di['Steepness'].iloc[-1]
        resumo.append(f"\n## Steepness DI Futuro: {steep_di:.2f} bps")
        if steep_di > 0:
            resumo.append("(Curva inclinada positiva - expectativa de juros mais altos no longo prazo)")
        elif steep_di < 0:
            resumo.append("(Curva invertida - expectativa de queda de juros)")
        else:
            resumo.append("(Curva plana)")

    if df_steepness_ntnb is not None and len(df_steepness_ntnb) > 0:
        steep_ntnb = df_steepness_ntnb['Steepness'].iloc[-1]
        resumo.append(f"\n## Steepness NTN-B: {steep_ntnb:.2f} bps")

    # Breakeven Inflation
    if df_bei is not None and len(df_bei) > 0:
        bei_atual = df_bei['BEI'].iloc[-1]
        resumo.append(f"\n## Breakeven Inflation: {bei_atual:.2f}%")
        resumo.append("(Inflação implícita esperada pelo mercado)")

    # Expectativas FOCUS
    if df_focus is not None and len(df_focus) > 0:
        resumo.append("\n## Expectativas FOCUS (Banco Central)")
        for _, row in df_focus.iterrows():
            ind = row['Indicador']
            ref = row['Referência']
            med = row['Mediana']
            resumo.append(f"- {ind} {ref}: {med}%")

    return "\n".join(resumo)


def buscar_noticias_google(queries: List[str] = None, max_por_query: int = 3, dias_atras: int = 7) -> str:
    """
    Busca notícias recentes no Google News via RSS.

    Args:
        queries: Lista de termos de busca
        max_por_query: Máximo de notícias por query
        dias_atras: Filtrar notícias dos últimos N dias

    Returns:
        String formatada com resumo das notícias
    """
    if queries is None:
        queries = [
            "Brasil economia",
            "Banco Central Brasil Selic",
            "inflação Brasil IPCA",
            "política fiscal Brasil"
        ]

    try:
        import feedparser
    except ImportError:
        logger.error("Pacote 'feedparser' não instalado. Execute: pip install feedparser")
        return ""

    noticias = []
    data_limite = datetime.now() - timedelta(days=dias_atras)

    for query in queries:
        try:
            # Construir URL do Google News RSS
            url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

            logger.info(f"Buscando notícias: '{query}'")
            feed = feedparser.parse(url)

            count = 0
            for entry in feed.entries:
                if count >= max_por_query:
                    break

                # Tentar parsear data (Google News usa formato struct_time)
                try:
                    pub_date = datetime(*entry.published_parsed[:6])

                    # Filtrar notícias antigas
                    if pub_date < data_limite:
                        continue

                    data_formatada = pub_date.strftime("%d/%m/%Y")
                except:
                    data_formatada = "Data desconhecida"

                titulo = entry.title

                # Extrair snippet se disponível
                snippet = ""
                if hasattr(entry, 'summary'):
                    snippet = entry.summary[:150] + "..." if len(entry.summary) > 150 else entry.summary

                noticias.append({
                    'titulo': titulo,
                    'data': data_formatada,
                    'snippet': snippet
                })

                count += 1

        except Exception as e:
            logger.warning(f"Erro ao buscar notícias para '{query}': {e}")
            continue

    if not noticias:
        logger.warning("Nenhuma notícia encontrada")
        return ""

    # Formatar notícias para o GPT
    resumo = [f"\n## NOTÍCIAS RECENTES ({len(noticias)} últimas)"]
    for i, noticia in enumerate(noticias[:10], 1):  # Limitar a 10 notícias
        resumo.append(f"\n{i}. **{noticia['titulo']}** ({noticia['data']})")
        if noticia['snippet']:
            resumo.append(f"   {noticia['snippet']}")

    logger.info(f"Coletadas {len(noticias)} notícias recentes")
    return "\n".join(resumo)


def gerar_analise_gpt(
    df_variacoes_ntnb: pd.DataFrame,
    df_variacoes_di: pd.DataFrame,
    df_steepness_di: Optional[pd.DataFrame] = None,
    df_steepness_ntnb: Optional[pd.DataFrame] = None,
    df_bei: Optional[pd.DataFrame] = None,
    df_focus: Optional[pd.DataFrame] = None,
    config: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Gera análise macroeconômica usando ChatGPT/OpenAI.

    Args:
        df_variacoes_ntnb: DataFrame com variações NTN-B
        df_variacoes_di: DataFrame com variações DI
        df_steepness_di: DataFrame com steepness DI
        df_steepness_ntnb: DataFrame com steepness NTN-B
        df_bei: DataFrame com breakeven inflation
        df_focus: DataFrame com expectativas FOCUS
        config: Configuração global

    Returns:
        Texto da análise ou None se desabilitado/erro
    """
    if config is None:
        logger.warning("Configuração não fornecida para análise GPT")
        return None

    gpt_cfg = config.get('gpt_analise', {})

    # Verificar se está habilitado
    if not gpt_cfg.get('enabled', False):
        logger.info("Análise GPT desabilitada na configuração")
        return None

    api_key = gpt_cfg.get('api_key', '')
    if not api_key or api_key == 'sua_api_key_openai_aqui':
        logger.warning("API Key OpenAI não configurada. Análise GPT desabilitada.")
        return None

    try:
        # Importar OpenAI (lazy import)
        try:
            from openai import OpenAI
        except ImportError:
            logger.error("Pacote 'openai' não instalado. Execute: pip install openai")
            return None

        # Preparar resumo dos dados
        resumo_dados = preparar_resumo_dados(
            df_variacoes_ntnb,
            df_variacoes_di,
            df_steepness_di,
            df_steepness_ntnb,
            df_bei,
            df_focus
        )

        # Buscar notícias recentes (se habilitado)
        noticias_texto = ""
        if gpt_cfg.get('buscar_noticias', True):
            logger.info("Buscando notícias recentes do Google News...")
            queries = gpt_cfg.get('queries_noticias', None)
            noticias_texto = buscar_noticias_google(
                queries=queries,
                max_por_query=3,
                dias_atras=7
            )

        # Configurações do modelo
        model = gpt_cfg.get('model', 'gpt-4o')
        max_tokens = gpt_cfg.get('max_tokens', 600)  # Aumentado para acomodar notícias
        temperature = gpt_cfg.get('temperature', 0.7)

        # Prompt customizável
        prompt_template = gpt_cfg.get('prompt_template', """
Você é um analista de mercado financeiro brasileiro especializado em renda fixa.

Analise os dados das curvas de juros brasileiras abaixo e forneça uma análise concisa (máximo 250 palavras) considerando:
- Movimentação das taxas NTN-B (títulos indexados à inflação) e DI Futuro (taxas nominais)
- Steepness (inclinação da curva) e seu significado
- Breakeven Inflation (inflação implícita)
- Expectativas FOCUS do Banco Central
- Contexto macroeconômico atual do Brasil (política monetária, fiscal, inflação)
- Notícias econômicas recentes (se disponíveis abaixo)

Seja objetivo, profissional e forneça insights relevantes para investidores.

DADOS DO MERCADO:
{dados}

{noticias}

ANÁLISE:
        """)

        prompt = prompt_template.replace('{dados}', resumo_dados).replace('{noticias}', noticias_texto)

        # Chamar API OpenAI
        logger.info(f"Gerando análise com {model}...")

        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você é um analista financeiro especializado em mercado brasileiro de renda fixa."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        analise = response.choices[0].message.content.strip()

        logger.info(f"Análise GPT gerada com sucesso ({len(analise)} caracteres)")

        return analise

    except Exception as e:
        logger.error(f"Erro ao gerar análise GPT: {e}")
        return None


if __name__ == "__main__":
    # Teste básico
    print("=" * 80)
    print("TESTE: Módulo pygpt_analise.py")
    print("=" * 80)

    # Dados fictícios para teste
    df_ntnb = pd.DataFrame({
        'Vencimento': ['2030', '2035', '2040'],
        'Taxa Atual': [6.5, 6.7, 6.9],
        'Var D-1 (bps)': [5, 7, 8],
        'Var D-30 (bps)': [25, 30, 35]
    })

    df_di = pd.DataFrame({
        'Vencimento': ['DI1F27', 'DI1F30', 'DI1F35'],
        'Taxa Atual': ['13,50%', '13,75%', '14,00%'],
        'Var D-1 (bps)': ['2,50', '3,00', '3,50']
    })

    resumo = preparar_resumo_dados(df_ntnb, df_di, None, None, None, None)
    print("\nResumo dos dados:")
    print(resumo)

    print("\n" + "=" * 80)
    print("Nota: Para testar a API OpenAI, configure a chave em config_ntnb.yaml")
