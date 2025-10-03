# Configuração de Envio de Email - Relatório NTN-B

## Como configurar o envio automático de email

### 1. Criar Senha de Aplicativo do Gmail

Para usar o Gmail com este script, você precisa criar uma **senha de aplicativo**:

1. Acesse: https://myaccount.google.com/apppasswords
2. Faça login na sua conta Gmail
3. Crie uma nova senha de aplicativo:
   - Nome: "Python NTN-B" (ou qualquer nome descritivo)
   - Copie a senha gerada (16 caracteres)

**IMPORTANTE**: Não use sua senha normal do Gmail! Use apenas a senha de aplicativo.

### 2. Configurar o arquivo `config_ntnb.yaml`

Edite o arquivo `config_ntnb.yaml` e preencha a seção `email`:

```yaml
email:
  enabled: true  # Mude para true para habilitar envio automático
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "seu-email@gmail.com"  # Seu email do Gmail
  sender_password: "xxxx xxxx xxxx xxxx"  # Cole aqui a senha de aplicativo de 16 dígitos
  recipients:
    - "alopes@bcpsecurities.com"
    - "outro@email.com"  # Adicione mais destinatários se necessário
  subject: "Relatório NTN-B - {date}"  # {date} será substituído pela data de referência
```

### 3. Executar o script

Execute normalmente:

```bash
python pyieldntnb.py
```

Se tudo estiver configurado corretamente, você verá no log:

```
INFO - Enviando relatório por email...
INFO - Conectando ao servidor SMTP smtp.gmail.com:587...
INFO - Enviando email para 1 destinatário(s)...
INFO - Email enviado com sucesso para: alopes@bcpsecurities.com
```

### 4. Testar o envio

Para testar sem modificar o config principal, você pode:

1. Fazer uma cópia de `config_ntnb.yaml` → `config_teste.yaml`
2. Modificar o `config_teste.yaml` com suas credenciais de teste
3. No código, temporariamente mudar `load_config()` para `load_config('config_teste.yaml')`

## Formato do Email

O email será enviado com:

- **Assunto**: "Relatório NTN-B - [Data de Referência]"
- **Corpo**: HTML completo inline com:
  - Tabela de Variações (com cores)
  - Tabela Comparativa de Taxas
  - Gráfico de Curvas
  - Gráfico Histórico 3 anos
  - Gráfico Histórico 5 anos

## Segurança

⚠️ **ATENÇÃO**:
- Nunca commite o arquivo `config_ntnb.yaml` com senha preenchida em repositórios públicos
- Mantenha a senha de aplicativo segura
- Se a senha vazar, revogue-a imediatamente em: https://myaccount.google.com/apppasswords

## Troubleshooting

### Erro: "Erro de autenticação"
- Verifique se está usando a senha de aplicativo (não a senha normal)
- Confirme que a verificação em 2 etapas está habilitada na conta Gmail
- Recrie a senha de aplicativo se necessário

### Erro: "Email ou senha não configurados"
- Verifique se preencheu `sender_email` e `sender_password` no YAML
- Confira se não há espaços extras ou caracteres especiais

### Email não está sendo enviado
- Verifique se `enabled: true` está configurado
- Confira os logs para ver se há mensagens de erro específicas
- Teste conectividade: `telnet smtp.gmail.com 587`

## Desabilitar o envio

Para desabilitar o envio automático, basta configurar:

```yaml
email:
  enabled: false
```

O script continuará gerando o HTML normalmente, apenas não enviará por email.
