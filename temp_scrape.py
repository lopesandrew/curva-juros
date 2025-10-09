import requests, pandas as pd
from pathlib import Path
urls = {
    'boi': 'https://www.cepea.org.br/br/indicador/series/boi-gordo.aspx?id=2',
    'bezerro': 'https://www.cepea.org.br/br/indicador/series/bezerro.aspx?id=8',
    'acucar': 'https://www.cepea.org.br/br/indicador/series/acucar.aspx?id=53',
    'algodao': 'https://www.cepea.org.br/br/indicador/series/algodao.aspx?id=54',
    'trigo': 'https://www.cepea.org.br/br/indicador/series/trigo.aspx?id=178',
    'arroz': 'https://www.cepea.org.br/br/indicador/series/arroz.aspx?id=91',
    'frango': 'https://www.cepea.org.br/br/indicador/series/frango.aspx?id=181'
}
headers = {'User-Agent': 'Mozilla/5.0'}
for name, url in urls.items():
    path = Path(f'series_{name}.xls')
    path.write_bytes(requests.get(url, headers=headers, timeout=120).content)
    df = pd.read_excel(path, engine='xlrd', skiprows=3, engine_kwargs={'ignore_workbook_corruption': True})
    print(name, df.columns.tolist(), df.head(1).to_dict(orient='records'), df.tail(1).to_dict(orient='records'))
