"""Ferramentas de linha de comando usando a biblioteca `mercados`.

Este script oferece utilitarios para:
  * Consultar series temporais do Banco Central (Selic, CDI, IPCA etc.);
  * Ajustar valores pela Selic (base diaria ou mensal);
  * Baixar negociacoes do mercado de balcao da B3 e aplicar filtros basicos.

Exemplos:
    python mercados_cli.py bcb-series "Selic diaria" --inicio 2024-01-01 --fim 2024-03-31
    python mercados_cli.py ajusta-selic dia 2024-01-01 2024-03-31 100000
    python mercados_cli.py b3-balcao 2024-10-04 --instrumento CRA --summary
"""

from __future__ import annotations

import argparse
import datetime as dt
import unicodedata
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from mercados.b3 import B3
from mercados.bcb import BancoCentral


def parse_iso_date(value: str) -> dt.date:
    """Converte uma string AAAA-MM-DD para date, validando o formato."""
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Data invalida: {value!r}. Use o formato AAAA-MM-DD.") from exc


def _normalize(text: str) -> str:
    """Remove acentos e coloca em minusculas para facilitar comparacoes."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def resolve_bcb_series_name(requested: str, available: dict[str, int]) -> str:
    """Resolve o nome amigavel informado pelo usuario para o nome real da serie do Banco Central."""
    normalized_available = {_normalize(name): name for name in available.keys()}
    key = _normalize(requested)
    if key in normalized_available:
        return normalized_available[key]
    options = ", ".join(sorted(available.keys()))
    raise ValueError(f"Serie '{requested}' nao encontrada. Series disponiveis: {options}")


def _decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    """Converte Decimal para float preservando nulos."""
    if value is None:
        return None
    return float(value)


def _emit_dataframe(df: pd.DataFrame, output: Optional[Path], preview_rows: int) -> None:
    """Mostra um preview do DataFrame e, se solicitado, salva em CSV."""
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=True)

    if df.empty:
        print("Nenhum dado encontrado.")
    else:
        preview = df if preview_rows <= 0 else df.head(preview_rows)
        print(preview.to_string())
        print(f"\nTotal de linhas: {len(df)}")

    if output:
        print(f"Arquivo salvo em: {output}")


def handle_bcb_series(args: argparse.Namespace) -> None:
    bc = BancoCentral()
    try:
        serie_nome = resolve_bcb_series_name(args.serie, bc.series)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    registros = bc.serie_temporal(serie_nome, inicio=args.inicio, fim=args.fim)
    if not registros:
        print("Nenhum dado retornado pelo Banco Central.")
        return

    df = pd.DataFrame(
        {
            "data": [item.data for item in registros],
            serie_nome: [float(item.valor) for item in registros],
        }
    ).set_index("data")
    df.index = pd.to_datetime(df.index)
    df.index.name = "data"

    _emit_dataframe(df, args.output, args.limit)

    if not df.empty:
        serie_col = df.columns[0]
        print(
            f"\nPeriodo: {df.index.min().date()} a {df.index.max().date()} | "
            f"Minimo: {df[serie_col].min():.4f} | "
            f"Maximo: {df[serie_col].max():.4f} | "
            f"Media: {df[serie_col].mean():.4f}"
        )


def handle_adjust_selic(args: argparse.Namespace) -> None:
    if args.data_inicial > args.data_final:
        raise SystemExit("data_inicial deve ser anterior ou igual a data_final.")

    bc = BancoCentral()
    if args.tipo == "dia":
        ajustado = bc.ajustar_selic_por_dia(args.data_inicial, args.data_final, args.valor)
    else:
        ajustado = bc.ajustar_selic_por_mes(args.data_inicial, args.data_final, args.valor)

    fator = (ajustado / args.valor) if args.valor else None
    print(f"Valor ajustado: {ajustado:.2f}")
    if fator is not None:
        print(f"Fator acumulado: {fator:.6f}")


def _prepare_b3_rows(registros: Iterable) -> pd.DataFrame:
    rows = []
    for item in registros:
        row = asdict(item)
        row["datahora"] = row["datahora"].isoformat()
        row["data_liquidacao"] = row["data_liquidacao"].isoformat() if row["data_liquidacao"] else None
        row["quantidade"] = _decimal_to_float(row["quantidade"])
        row["preco"] = _decimal_to_float(row["preco"])
        row["volume"] = _decimal_to_float(row["volume"])
        row["taxa"] = _decimal_to_float(row["taxa"])
        rows.append(row)
    return pd.DataFrame(rows)


def handle_b3_balcao(args: argparse.Namespace) -> None:
    b3_client = B3()
    registros = list(b3_client.negociacao_balcao(args.data))
    if not registros:
        print("Nenhum registro retornado pela B3.")
        return

    df = _prepare_b3_rows(registros)

    if args.instrumento:
        instrumentos = {inst.upper().strip() for inst in args.instrumento}
        df = df[df["instrumento"].str.upper().isin(instrumentos)]
    if args.emissor:
        df = df[df["emissor"].str.contains(args.emissor, case=False, na=False)]
    if args.codigo_if:
        df = df[df["codigo_if"].str.upper() == args.codigo_if.upper()]
    if args.isin:
        df = df[df["codigo_isin"].str.upper() == args.isin.upper()]
    if args.situacao:
        df = df[df["situacao"].fillna("").str.upper() == args.situacao.upper()]
    if args.origem:
        df = df[df["origem"].fillna("").str.upper() == args.origem.upper()]

    if df.empty:
        print("Nenhum dado encontrado com os filtros informados.")
        return

    df = df.sort_values("volume", ascending=False, na_position="last")
    _emit_dataframe(df, args.output, args.limit)

    if args.summary:
        summary = (
            df.groupby("instrumento")
            .agg(
                negocios=("codigo", "count"),
                volume=("volume", "sum"),
                quantidade=("quantidade", "sum"),
                preco_medio=("preco", "mean"),
                taxa_media=("taxa", "mean"),
            )
            .sort_values("volume", ascending=False)
        )
        summary["volume"] = summary["volume"].round(2)
        summary["quantidade"] = summary["quantidade"].round(2)
        summary["preco_medio"] = summary["preco_medio"].round(4)
        summary["taxa_media"] = summary["taxa_media"].round(4)
        print("\nResumo por instrumento:")
        print(summary.to_string())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilitarios para acessar dados de mercado com a biblioteca mercados."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_bcb = subparsers.add_parser("bcb-series", help="Baixa uma serie temporal do Banco Central.")
    p_bcb.add_argument("serie", help="Nome da serie (ex: 'Selic diaria', 'CDI', 'IPCA mensal').")
    p_bcb.add_argument("--inicio", type=parse_iso_date, help="Data inicial (AAAA-MM-DD).")
    p_bcb.add_argument("--fim", type=parse_iso_date, help="Data final (AAAA-MM-DD).")
    p_bcb.add_argument("--output", type=Path, help="Arquivo CSV de saida.")
    p_bcb.add_argument("--limit", type=int, default=10, help="Linhas mostradas no preview (0 para ilimitado).")
    p_bcb.set_defaults(func=handle_bcb_series)

    p_selic = subparsers.add_parser("ajusta-selic", help="Ajusta um valor pela Selic.")
    p_selic.add_argument("tipo", choices=["dia", "mes"], help="Base de calculo: diaria ou mensal.")
    p_selic.add_argument("data_inicial", type=parse_iso_date, help="Data inicial (AAAA-MM-DD).")
    p_selic.add_argument("data_final", type=parse_iso_date, help="Data final (AAAA-MM-DD).")
    p_selic.add_argument("valor", type=Decimal, help="Valor base a ser ajustado.")
    p_selic.set_defaults(func=handle_adjust_selic)

    p_balcao = subparsers.add_parser("b3-balcao", help="Negociacoes do mercado de balcao da B3 para uma data.")
    p_balcao.add_argument("data", type=parse_iso_date, help="Data do pregrao (AAAA-MM-DD).")
    p_balcao.add_argument(
        "--instrumento",
        "-i",
        action="append",
        help="Filtra por instrumento (ex: CRA, CRI, DEB). Pode ser usado multiplas vezes.",
    )
    p_balcao.add_argument("--emissor", "-e", help="Filtro de substring para o emissor (case insensitive).")
    p_balcao.add_argument("--codigo-if", "-c", help="Filtro exato para o campo codigo_if.")
    p_balcao.add_argument("--isin", help="Filtro exato para o codigo ISIN.")
    p_balcao.add_argument("--situacao", help="Filtra pela situacao do negocio (ex: Confirmado).")
    p_balcao.add_argument("--origem", help="Filtra pela origem do negocio (ex: Registro).")
    p_balcao.add_argument("--output", type=Path, help="Arquivo CSV de saida.")
    p_balcao.add_argument("--limit", type=int, default=10, help="Linhas mostradas no preview (0 para ilimitado).")
    p_balcao.add_argument("--summary", action="store_true", help="Exibe tambem um resumo agregado por instrumento.")
    p_balcao.set_defaults(func=handle_b3_balcao)

    return parser


def main(args: Optional[list[str]] = None) -> None:
    parser = build_parser()
    parsed = parser.parse_args(args)
    parsed.func(parsed)


if __name__ == "__main__":
    main()
