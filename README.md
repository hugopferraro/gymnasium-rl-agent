# Gymnasium RL Agent

Projeto da disciplina **Inteligência Artificial II**: um agente Double DQN, implementado com
TensorFlow/Keras, para resolver o ambiente `LunarLander-v3` do Gymnasium.

## Tecnologia escolhida

O Gymnasium é o sucessor mantido do OpenAI Gym. A atividade cita o identificador histórico
`LunarLander-v2`; este projeto usa o identificador atual `LunarLander-v3` e a API moderna, que
distingue `terminated` de `truncated`.

## Requisitos

- Python 3.12 e [uv](https://docs.astral.sh/uv/).
- WSL2 Ubuntu para treinamento com GPU NVIDIA.
- Driver NVIDIA compatível com CUDA no WSL2.

No WSL2:

```bash
uv sync --extra gpu --extra video --group dev
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Para execução somente em CPU, use `uv sync --extra video --group dev`.

## Uso

```bash
# Treinamento padrão
uv run gymnasium-rl-agent train --config configs/lunar_lander.yaml

# Busca aleatória de hiperparâmetros
uv run gymnasium-rl-agent tune --config configs/tuning.yaml

# Avaliação final em 100 episódios e comparação aleatória
uv run gymnasium-rl-agent evaluate \
  --config configs/lunar_lander.yaml \
  --model artifacts/final/model.keras \
  --episodes 100 \
  --include-random-baseline \
  --output artifacts/final/evaluation.json

# Gravação de demonstração
uv run gymnasium-rl-agent demo \
  --config configs/lunar_lander.yaml \
  --model artifacts/final/model.keras \
  --record
```

Cada treinamento registra configuração, versões, métricas CSV, TensorBoard, gráfico, resumo e
checkpoint em `artifacts/runs/`. Apenas os resultados selecionados devem ser promovidos para
`artifacts/final/`.

## Estrutura

- `src/gymnasium_rl_agent/`: pacote modular do agente.
- `configs/`: configurações reproduzíveis de treinamento e tuning.
- `tests/`: testes unitários e de integração.
- `docs/`: relatório, apresentação, roteiro e enunciado original.
- `artifacts/`: resultados temporários e entrega final.

## Qualidade

```bash
uv run ruff check .
uv run pytest
uv run mypy src
```

Os resultados finais só devem ser descritos no relatório após a execução dos experimentos;
nenhum valor de desempenho é presumido pelo código.
