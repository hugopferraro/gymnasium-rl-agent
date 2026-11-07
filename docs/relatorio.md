# Agente Double DQN para o ambiente LunarLander-v3

> Substituir os campos entre colchetes e inserir apenas resultados produzidos pelas execuções.

## Identificação

- Disciplina: Inteligência Artificial II
- Grupo: [número]
- Integrantes: [nomes]
- Data: [data]

## Resumo

[Apresentar objetivo, método, melhor configuração e resultado final em até 200 palavras.]

## 1. Introdução e objetivos

Este trabalho investiga a aplicação de aprendizado por reforço profundo ao controle autônomo
do módulo lunar no ambiente `LunarLander-v3`. O objetivo mensurável é obter retorno médio de
pelo menos 200 pontos em 100 episódios de teste não utilizados no treinamento ou tuning.

## 2. Ambiente

Descrever estados, quatro ações discretas, recompensas, condições de término e truncamento.
Explicar que Gymnasium, mantido pela Farama Foundation, é o sucessor do OpenAI Gym e que
`LunarLander-v3` atualiza o identificador `LunarLander-v2` citado no enunciado.

## 3. Modelagem e algoritmo

Descrever a rede neural, política epsilon-greedy, replay buffer, rede-alvo, Huber loss e a
diferença entre DQN e Double DQN. Incluir pseudocódigo e diagrama da arquitetura.

## 4. Dataset de experiência

O conjunto de dados é produzido durante a interação. Cada amostra contém estado, ação,
recompensa, próximo estado e indicador de término. Informar capacidade do replay buffer,
estratégia de amostragem e quantidade final de transições.

## 5. Hiperparametrização

Apresentar o espaço de busca, as 12 tentativas, seeds de validação, critério de seleção e tabela
de `tuning_results.csv`. Justificar a configuração vencedora.

## 6. Resultados

Incluir curvas de aprendizado, comparação com política aleatória e avaliação final em 100 seeds.
Reportar média, mediana, desvio-padrão, intervalo de confiança, mínimo, máximo e taxa de retorno
igual ou superior a 200.

## 7. Análise crítica

Discutir estabilidade entre seeds, eficiência amostral, custo computacional, possíveis vieses de
seleção e limitações da política treinada.

## 8. Conclusão e trabalhos futuros

Responder se o objetivo foi atingido. Sugerir replay priorizado, arquitetura dueling, n-step
returns e comparação com PPO como extensões.

## Referências

- Documentação oficial do Gymnasium.
- Mnih et al. (2015), *Human-level control through deep reinforcement learning*.
- van Hasselt, Guez e Silver (2016), *Deep Reinforcement Learning with Double Q-learning*.
- Sutton e Barto (2018), *Reinforcement Learning: An Introduction*.
