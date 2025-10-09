# 🤖 Análise GPT - Integração ChatGPT/OpenAI

## 📋 Descrição

Funcionalidade que adiciona análise automatizada via ChatGPT/OpenAI ao relatório de curvas de juros NTN-B e DI Futuro. A IA analisa os dados e gera comentário contextualizado relacionando com o cenário macroeconômico brasileiro.

## ✨ Funcionalidades

- **Análise automática** das curvas NTN-B e DI Futuro
- **Contexto macroeconômico** brasileiro (inflação, Selic, política monetária)
- **Interpretação de steepness** e breakeven inflation
- **Comentário profissional** no topo do relatório HTML
- **Totalmente configurável** via YAML

## 🚀 Como Usar

### 1. Obter API Key OpenAI

1. Acesse: https://platform.openai.com/api-keys
2. Faça login ou crie uma conta
3. Clique em **"Create new secret key"**
4. Dê um nome (ex: "NTN-B Analysis")
5. **Copie a chave** (ela aparece apenas uma vez!)

### 2. Configurar no arquivo YAML

Edite seu `config_ntnb.yaml` (não o .example):

```yaml
# Análise GPT (ChatGPT/OpenAI)
gpt_analise:
  enabled: true  # Mude para true para habilitar
  api_key: "sk-proj-xxxxxxxxxxxxxxxxxxxx"  # Cole sua API key aqui
  model: "gpt-4o"  # Modelo recomendado
  max_tokens: 500
  temperature: 0.7
```

### 3. Instalar dependência

```bash
pip install openai
```

Ou instale todas as dependências:

```bash
pip install -r requirements.txt
```

### 4. Executar normalmente

```bash
python pyieldntnb.py
```

O relatório HTML agora incluirá uma seção de análise GPT no topo!

## 💰 Custos

### Modelos Disponíveis

| Modelo | Custo Input | Custo Output | Custo por Análise | Recomendação |
|--------|-------------|--------------|-------------------|--------------|
| **gpt-4o** | $2.50/1M tokens | $10/1M tokens | ~R$ 0.05-0.10 | ⭐ Recomendado |
| gpt-4.1 | ~2x gpt-4o | ~2x gpt-4o | ~R$ 0.10-0.20 | Análises complexas |
| gpt-4o-mini | $0.15/1M tokens | $0.60/1M tokens | ~R$ 0.01 | Mais econômico |

### Estimativa Mensal

- **20 dias úteis/mês**: ~R$ 1-2 (gpt-4o)
- Custo desprezível comparado ao valor da análise

## ⚙️ Configurações Avançadas

### Personalizar Prompt

Edite o `prompt_template` no config YAML:

```yaml
gpt_analise:
  prompt_template: |
    Você é um analista sênior especializado em renda fixa brasileira.

    [Seu prompt customizado aqui]

    DADOS:
    {dados}
```

### Ajustar Parâmetros

- **max_tokens**: Tamanho máximo da resposta (500 = ~300 palavras)
- **temperature**:
  - `0.0` = Determinístico, conservador
  - `0.7` = Balanceado (padrão)
  - `1.0` = Criativo, variado

## 📊 Exemplo de Análise Gerada

```
🤖 Análise de Mercado Automatizada (IA)

O mercado de renda fixa brasileiro apresentou movimentação significativa
com alta de 25 bps nas NTN-Bs em relação ao mês anterior, refletindo
aumento nas expectativas inflacionárias. O steepness positivo de 45 bps
na curva DI indica que o mercado precifica manutenção da Selic em
patamar elevado no curto prazo.

O breakeven de 4.8% está acima do centro da meta (3%), sinalizando
preocupação com pressões inflacionárias. As expectativas FOCUS apontam
IPCA de 4.5% para 2025, corroborando este cenário.

Recomenda-se cautela com duration longa até maior clareza sobre trajetória
da política monetária.
```

## ⚠️ Observações Importantes

1. **Não substitui análise humana**: Use como complemento
2. **API Key é sensível**: Nunca compartilhe ou versione no Git
3. **Pode ser desabilitada**: Basta `enabled: false` no config
4. **Requer internet**: Para chamar API OpenAI
5. **Adiciona ~2-5s**: Ao tempo de execução

## 🔒 Segurança

- ✅ API Key armazenada apenas em `config_ntnb.yaml` (local)
- ✅ Arquivo local ignorado no `.gitignore`
- ✅ Exemplo sem credenciais em `config_ntnb.yaml.example`
- ✅ Nenhuma chave exposta no repositório

## 🆘 Troubleshooting

### Erro: "Pacote 'openai' não instalado"
```bash
pip install openai
```

### Erro: "API Key OpenAI não configurada"
- Verifique se `api_key` está preenchida no `config_ntnb.yaml`
- Certifique-se que não está usando "sua_api_key_openai_aqui"

### Erro: "401 Unauthorized"
- API Key inválida ou expirada
- Gere nova chave em https://platform.openai.com/api-keys

### Análise não aparece no relatório
- Verifique `enabled: true` no config
- Veja os logs para erros na geração

## 📝 Logs

O sistema registra:
- `"Gerando análise GPT..."` - Início da chamada
- `"Análise GPT gerada com sucesso"` - Sucesso
- `"Análise GPT desabilitada"` - Se `enabled: false`
- Erros detalhados em caso de falha

## 🔄 Desabilitar Temporariamente

No `config_ntnb.yaml`:

```yaml
gpt_analise:
  enabled: false  # Volta ao comportamento normal
```

## 📚 Referências

- OpenAI API: https://platform.openai.com/docs
- Pricing: https://openai.com/pricing
- Python SDK: https://github.com/openai/openai-python
