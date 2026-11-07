# Relatório

O relatório final está em [`relatorio.md`](relatorio.md). Antes da entrega:

1. preencha o número do grupo e os nomes faltantes;
2. confirme que a Figura 1 é exibida corretamente;
3. exporte o documento para PDF preservando tabelas, fórmulas e links; e
4. revise a paginação do sumário após a conversão.

Uma opção de conversão, caso Pandoc esteja instalado, é:

```bash
pandoc report/relatorio.md \
  --from gfm \
  --standalone \
  --toc \
  --number-sections \
  -o report/relatorio.pdf
```

O PDF deve ser aberto e conferido visualmente antes de compor o arquivo compactado da entrega.
