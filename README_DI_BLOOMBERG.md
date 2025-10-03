# Integração DI Futuro - Dados Bloomberg

Este documento explica como preparar e integrar dados históricos de DI Futuro da Bloomberg ao sistema de relatórios.

---

## 📋 Formato do CSV da Bloomberg

O arquivo `di_historico.csv` deve ter as seguintes colunas (em ordem):

```
Data,Ticker,Vencimento,Taxa,Volume,Contratos_Abertos,Num_Negocios,Dias_Corridos,Dias_Uteis
```

### Colunas Obrigatórias:

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `Data` | date | Data do registro | `2024-01-15` |
| `Ticker` | string | Código do contrato (padrão BMF) | `DI1F26` |
| `Vencimento` | date | Data de vencimento do contrato | `2026-01-02` |
| `Taxa` | float | Taxa anual em % (não em decimal!) | `13.45` |
| `Volume` | float | Volume financeiro negociado | `1500000000.50` |
| `Contratos_Abertos` | int | Contratos em aberto (open interest) | `850000` |
| `Num_Negocios` | int | Número de negócios | `1250` |
| `Dias_Corridos` | int | Dias corridos até vencimento | `450` |
| `Dias_Uteis` | int | Dias úteis até vencimento | `315` |

### ⚠️ Observações Importantes:

1. **Data**: Formato `YYYY-MM-DD` (padrão ISO)
2. **Taxa**: Em **percentual** (13.45 = 13.45%), **não** em decimal (0.1345)
3. **Ticker**: Usar nomenclatura BMF (DI1F26, DI1F27, etc.)
4. **Encoding**: UTF-8 com BOM (`utf-8-sig`) ou sem BOM (`utf-8`)
5. **Separador**: Vírgula (`,`)
6. **Decimais**: Ponto (`.`) não vírgula

---

## 🔄 Processo de Integração

### Passo 1: Preparar CSV da Bloomberg

1. Exporte dados históricos da Bloomberg para Excel/CSV
2. Certifique-se de ter **apenas contratos com vencimento em Janeiro** (DI1F)
3. Ajuste nomes de colunas para o formato esperado
4. Converta datas para formato ISO (`YYYY-MM-DD`)
5. Garanta que taxas estão em **percentual** (não decimal)

### Passo 2: Validar Formato

Execute este script Python para validar o CSV:

```python
import pandas as pd

# Carrega CSV
df = pd.read_csv('di_historico.csv')

# Verifica colunas
colunas_esperadas = ['Data', 'Ticker', 'Vencimento', 'Taxa', 'Volume',
                     'Contratos_Abertos', 'Num_Negocios', 'Dias_Corridos', 'Dias_Uteis']
print(f"Colunas no arquivo: {list(df.columns)}")
print(f"Colunas esperadas:  {colunas_esperadas}")
print(f"OK: {list(df.columns) == colunas_esperadas}")

# Verifica dados
print(f"\nTotal de registros: {len(df):,}")
print(f"Período: {df['Data'].min()} a {df['Data'].max()}")
print(f"Tickers únicos: {sorted(df['Ticker'].unique())}")
print(f"Taxa média: {df['Taxa'].mean():.2f}%")
print(f"Taxa mínima: {df['Taxa'].min():.2f}%")
print(f"Taxa máxima: {df['Taxa'].max():.2f}%")

# Amostra
print(f"\nAmostra dos dados:")
print(df.head())
```

### Passo 3: Copiar para Diretório de Dados

```bash
# Windows
copy di_historico.csv "C:\Users\Andrew Lopes\OneDrive - BCP Securities\Área de Trabalho\Python NTN-B\dados\"

# Linux/Mac
cp di_historico.csv "dados/"
```

### Passo 4: Teste Inicial

Execute o sistema para verificar se tudo funciona:

```bash
python pyieldntnb.py
```

Verifique:
- ✓ Tabela de variações DI com valores históricos
- ✓ Gráfico de curva DI
- ✓ Gráficos históricos DI com linhas visíveis
- ✓ Email enviado com sucesso

---

## 🔄 Funcionamento Diário (Automático)

Após a integração inicial, o sistema funciona assim:

1. **Carrega** histórico existente (`di_historico.csv` com dados Bloomberg)
2. **Coleta** dados do dia atual via API B3
3. **Remove** duplicatas do dia atual (se já existirem)
4. **Adiciona** novos dados ao histórico
5. **Salva** arquivo atualizado
6. **Gera** relatório HTML + envia email

**Nenhuma ação manual necessária!** O sistema atualiza automaticamente a cada execução.

---

## 📊 Exemplo de CSV da Bloomberg

Arquivo: `di_historico.csv`

```csv
Data,Ticker,Vencimento,Taxa,Volume,Contratos_Abertos,Num_Negocios,Dias_Corridos,Dias_Uteis
2024-01-15,DI1F26,2026-01-02,13.450,1500000000.50,850000,1250,718,503
2024-01-15,DI1F27,2027-01-04,13.320,980000000.75,625000,980,1084,759
2024-01-15,DI1F28,2028-01-03,13.180,450000000.25,380000,650,1449,1015
2024-01-16,DI1F26,2026-01-02,13.475,1520000000.30,852000,1280,717,502
2024-01-16,DI1F27,2027-01-04,13.340,995000000.60,628000,1005,1083,758
2024-01-16,DI1F28,2028-01-03,13.200,465000000.80,382000,670,1448,1014
```

---

## 🔧 Colunas Opcionais

Se sua base Bloomberg não tiver todas as colunas, você pode preencher com valores padrão:

```python
import pandas as pd

# Carrega dados parciais da Bloomberg
df = pd.read_csv('bloomberg_export.csv')

# Adiciona colunas faltantes com valores padrão
if 'Dias_Corridos' not in df.columns:
    df['Dias_Corridos'] = 0
if 'Dias_Uteis' not in df.columns:
    df['Dias_Uteis'] = 0
if 'Volume' not in df.columns:
    df['Volume'] = 0.0
if 'Contratos_Abertos' not in df.columns:
    df['Contratos_Abertos'] = 0
if 'Num_Negocios' not in df.columns:
    df['Num_Negocios'] = 0

# Salva com todas as colunas
df.to_csv('di_historico.csv', index=False, encoding='utf-8-sig')
```

---

## ❓ Troubleshooting

### Erro: "UnicodeDecodeError"
**Solução**: Salve o CSV com encoding UTF-8

### Erro: "KeyError: 'Taxa'"
**Solução**: Verifique se os nomes das colunas estão corretos (case-sensitive)

### Erro: Taxas muito altas (>100%)
**Solução**: Suas taxas podem estar em decimal. Multiplique por 100:
```python
df['Taxa'] = df['Taxa'] * 100
```

### Erro: Datas não reconhecidas
**Solução**: Converta para formato ISO:
```python
df['Data'] = pd.to_datetime(df['Data']).dt.strftime('%Y-%m-%d')
df['Vencimento'] = pd.to_datetime(df['Vencimento']).dt.strftime('%Y-%m-%d')
```

### Gráficos vazios
**Solução**: Certifique-se de ter pelo menos 10-20 dias de dados para gráficos visíveis

---

## 📝 Manutenção

### Backup Periódico
Recomenda-se fazer backup semanal do `di_historico.csv`:

```bash
# Windows
copy dados\di_historico.csv dados\backup\di_historico_%date:~-4,4%%date:~-7,2%%date:~-10,2%.csv

# Linux/Mac
cp dados/di_historico.csv dados/backup/di_historico_$(date +%Y%m%d).csv
```

### Verificação de Integridade

Execute periodicamente para verificar qualidade dos dados:

```python
import pandas as pd

df = pd.read_csv('dados/di_historico.csv')

# Verifica datas duplicadas
duplicatas = df.groupby(['Data', 'Ticker']).size()
duplicatas = duplicatas[duplicatas > 1]
if len(duplicatas) > 0:
    print(f"⚠️  ATENÇÃO: {len(duplicatas)} registros duplicados encontrados!")
    print(duplicatas)
else:
    print("✓ Nenhum registro duplicado")

# Verifica gaps nas datas
df['Data'] = pd.to_datetime(df['Data'])
datas_unicas = sorted(df['Data'].unique())
gaps = []
for i in range(1, len(datas_unicas)):
    diff = (datas_unicas[i] - datas_unicas[i-1]).days
    if diff > 3:  # Gap de mais de 3 dias (considera fins de semana)
        gaps.append((datas_unicas[i-1], datas_unicas[i], diff))

if gaps:
    print(f"\n⚠️  ATENÇÃO: {len(gaps)} gaps encontrados nas datas:")
    for d1, d2, diff in gaps[:5]:  # Mostra primeiros 5
        print(f"  {d1.date()} → {d2.date()} ({diff} dias)")
else:
    print("✓ Nenhum gap significativo nas datas")
```

---

## 📞 Suporte

Se tiver dúvidas ou problemas na integração:

1. Verifique este README
2. Execute os scripts de validação acima
3. Consulte os logs do sistema em `pyieldntnb.py`
4. Verifique o arquivo `di_historico.csv` manualmente

---

**Última atualização**: 2025-10-02
