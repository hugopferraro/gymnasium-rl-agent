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
export UV_PROJECT_ENVIRONMENT=.venv-wsl
uv sync --extra gpu --extra video --group dev
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Para execução somente em CPU, use `uv sync --extra video --group dev`.

## Fluxo completo da atividade

Os comandos abaixo estão na ordem em que devem ser executados. Primeiro selecionamos os
hiperparâmetros; somente depois treinamos e avaliamos o modelo final.

### 1. Encontrar os melhores hiperparâmetros

```bash
uv run gymnasium-rl-agent tune --config configs/tuning.yaml
```

O comando executa 12 trials e, ao terminar, informa a pasta criada. Dentro dela estarão
`tuning_results.csv`, `best_trial.json` e `best_config.yaml`. O último arquivo combina os
hiperparâmetros vencedores com o orçamento completo de 1.000 episódios para o treinamento final.

### 2. Treinar o modelo final

Use exatamente o `next_command` mostrado ao final do tuning. Exemplo:

```bash
uv run gymnasium-rl-agent train \
  --config "artifacts/runs/<PASTA-DO-TUNING>/best_config.yaml"
```

Não use `configs/lunar_lander.yaml` nesta etapa: ele representa apenas a configuração inicial,
antes do tuning.

Depois do treinamento, copie os arquivos selecionados para a área final:

```bash
mkdir -p artifacts/final
cp "artifacts/runs/<PASTA-DO-TREINO>/best_model.keras" artifacts/final/model.keras
cp "artifacts/runs/<PASTA-DO-TREINO>/config.yaml" artifacts/final/config.yaml
cp "artifacts/runs/<PASTA-DO-TREINO>/training.png" artifacts/final/training.png
cp "artifacts/runs/<PASTA-DO-TUNING>/tuning_results.csv" artifacts/final/tuning_results.csv
```

### 3. Avaliar o modelo final

Use a configuração final copiada na etapa anterior:

```bash
uv run gymnasium-rl-agent evaluate \
  --config artifacts/final/config.yaml \
  --model artifacts/final/model.keras \
  --episodes 100 \
  --include-random-baseline \
  --output artifacts/final/evaluation.json
```

### 4. Gravar a demonstração

```bash
uv run gymnasium-rl-agent demo \
  --config artifacts/final/config.yaml \
  --model artifacts/final/model.keras \
  --episodes 3 \
  --record \
  --video-dir artifacts/final/videos
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
