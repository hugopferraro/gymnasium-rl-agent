# Contexto de Aprendizagem — Parada Obrigatória 2

Disciplina: Inteligência Artificial II\
Prazo de entrega: 09/11/2026, às 23h59\
Pontuação máxima: 7 pontos

## Orientações gerais

- Esta atividade deverá ser realizada em equipes de até 4 componentes;
- Os temas contemplados são: aplicações de redes neurais e aprendizado por reforço (Reinforcement Learning);
- Todos os membros da equipe devem realizar a postagem do trabalho finalizado no AVA;
- A correta identificação de todos os estudantes envolvidos deve constar na atividade;
- O prazo máximo para envio via AVA encontra-se cadastrado na tarefa.

## Proposta

No contexto da atividade prática de avaliação, pretende-se que, utilizando as bibliotecas Python, em especial TensorFlow, sejam desenvolvidos modelos baseados em deep learning que permitam a um agente desempenhar, com sucesso, a tarefa de jogar um determinado jogo de forma autônoma.

Este sucesso (boas jogadas) é medido pela eficácia do agente em atingir os objetivos do jogo.

Os grupos de trabalho devem recorrer ao toolkit Gymnasium da OpenAI e tirar partido dos vários ambientes jogáveis fornecidos por esta ferramenta.

Além de compatível com TensorFlow, este toolkit fornece informação detalhada sobre o ambiente a cada instante, assim como o impacto de cada ação tomada (Figura 1).

Figura 1: ciclo Gym OpenAI.

Após implementação do ambiente, recorrendo ao toolkit Gym, deverão desenvolver e otimizar modelos de deep learning que permitam a um agente ter a capacidade de, autonomamente, jogar o jogo.

Para atingir este objetivo, terão que percorrer todo um processo que passa, invariavelmente, pela coleção do conjunto de dados até à concepção e treino do modelo em si.

O agente deverá então usar o modelo desenvolvido para obter toda a informação necessária para executar, com sucesso, a sua tarefa.

Os grupos, divididos entre pares e ímpares, deverão conceber um agente capaz de jogar em pelo menos um dos seguintes ambientes jogáveis, seja este simples ou complexo (agentes capazes de jogar em ambientes complexos serão valorizados).

### Grupos de número par

Ambientes simples:

- CartPole-v1
- Acrobot-v1

Ambientes complexos:

- LunarLander-v2
- Qualquer ambiente Atari

### Grupos de número ímpar

Ambientes simples:

- CartPole-v1
- Pendulum-v0

Ambientes complexos:

- BipedalWalker-v2
- Qualquer ambiente Atari

Todo o processo deverá ser acompanhado de exemplos e indicações que permitam reproduzir todos os passos realizados, assim como os resultados obtidos.

## Etapas do projeto

### Etapa 1

Entrar na plataforma indicada Gymnasium OpenAI e escolher o ambiente.

### Etapa 2

Modelar o problema e aplicar conhecimentos teóricos para criação do agente inteligente.

### Etapa 3

Criar código em Python com o protótipo em funcionamento.

## Relatório

Os resultados obtidos deverão ser objeto de um relatório que contenha, entre outros:

1. Quais os domínios a tratar, quais os objetivos e como se propõe a atingi-los;
2. Descrição do ambiente jogável implementado, o porquê da sua escolha e quais as suas características;
3. Descrição do modelo de deep learning desenvolvido, quais as suas características, como e sobre que parâmetros foi realizado o tuning do modelo, entre outros detalhes que seja oportuno fornecer;
4. Descrição do dataset utilizado para treino do modelo e descrição do treino em si;
5. Sumário dos resultados obtidos e respectiva análise crítica;
6. Apresentação de sugestões e recomendações para melhoria do modelo desenvolvido.

## Entrega

As submissões deverão ser no AVA, enviando, em um único arquivo compactado, todos os elementos produzidos.

Tanto o assunto da mensagem como o arquivo submetido deverão ser identificados na forma `IAII_PO2GX`, em que X designa o número do grupo de trabalho.

A sessão de apresentação do trabalho decorrerá através de vídeo, que será gravado e terá um resumo dos principais desafios e resultados alcançados.

Cada grupo disporá de 10 minutos para realizar a apresentação, utilizando os meios que considerar mais adequados.

## Critérios de avaliação

### 1. Parte escrita — 1,5 pontos

| Nível      | Descrição                                                                                                                                                  | Pontos |
| ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Excelente  | Escrita clara e objetiva, seguindo padrões científicos, tais como: sumário, introdução, corpo do trabalho, conclusão, referências, anexos (se existentes). | 1,5    |
| Muito bom  | Discussão dos resultados + aplicação de hiperparametrização + conclusões.                                                                                  | 1,3    |
| Bom        | Descrição da arquitetura do modelo com adição de algoritmos de busca + resultados parciais.                                                                | 1,0    |
| Razoável   | Escreveu e detalhou a base de dados + não descreveu a arquitetura do modelo.                                                                               | 0,5    |
| Não atende | Sem entrega, cópia, fora do contexto e/ou não atendimento às especificações do trabalho.                                                                   | 0      |

### 2. Apresentação em PowerPoint + vídeo — 1,5 pontos

| Nível      | Descrição                                                                                                                                                         | Pontos |
| ---------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| Excelente  | Apresentação clara e objetiva, seguindo padrões científicos, tais como: sumário, introdução, corpo do trabalho, conclusões, bibliografia, anexos (se existentes). | 1,5    |
| Muito bom  | Discutiu os resultados com aplicação de hiperparametrização + conclusões.                                                                                         | 1,3    |
| Bom        | Apresentou a descrição da arquitetura do modelo com adição de algoritmos de busca + resultados parciais.                                                          | 1,0    |
| Razoável   | Apresentou e detalhou a base de dados e descreveu a arquitetura do modelo.                                                                                        | 0,5    |
| Não atende | Sem entrega, cópia, fora do contexto e/ou não atendimento às especificações do trabalho.                                                                          | 0      |

### 3. Código em Python — 2 pontos

| Nível      | Descrição                                                                                                          | Pontos |
| ---------- | ------------------------------------------------------------------------------------------------------------------ | ------ |
| Excelente  | Código funcionando, documentado e com demonstração/justificativa do uso dos melhores parâmetros.                   | 2,0    |
| Muito bom  | Discutiu os resultados com aplicação de hiperparametrização + conclusões em Python.                                | 1,8    |
| Bom        | Apresentou a descrição da arquitetura do modelo com adição de algoritmos de busca + resultados parciais em Python. | 1,5    |
| Razoável   | Apresentou e detalhou a base de dados e descreveu a arquitetura do modelo em Python.                               | 0,5    |
| Não atende | Sem entrega, cópia, fora do contexto e/ou não atendimento às especificações do trabalho.                           | 0      |

### 4. Implementação e funcionamento do protótipo — 2 pontos

| Nível      | Descrição                                                                                                                                                          | Pontos |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| Excelente  | Pleno funcionamento e instruções de uso (incluindo dependência) do código devidamente documentado e com demonstração/justificativa do uso dos melhores parâmetros. | 2,0    |
| Muito bom  | Pleno funcionamento da descrição dos resultados com aplicação de hiperparametrização + conclusões em Python.                                                       | 1,8    |
| Bom        | Pleno funcionamento da descrição da arquitetura do modelo com adição de algoritmos de busca + resultados parciais em Python.                                       | 1,5    |
| Razoável   | Pleno funcionamento da apresentação e detalhamento da base de dados + descrição da arquitetura do modelo em Python.                                                | 0,5    |
| Não atende | Sem entrega, cópia, fora do contexto e/ou não atendimento às especificações do trabalho.                                                                           | 0      |

Pontuação total: 7 pontos.