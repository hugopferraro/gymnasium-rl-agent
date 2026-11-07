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

### Preparação do WSL2 e da GPU

Todos os comandos de instalação, tuning e treinamento devem ser executados no Ubuntu/WSL2. Não
execute `uv sync` pelo PowerShell na mesma cópia do projeto, pois ambientes virtuais do Windows
e do Linux não são compatíveis.

Primeiro, confirme que o driver do Windows está expondo a GPU ao WSL:

```bash
nvidia-smi
```

Instale as dependências do projeto, incluindo CUDA e gravação de vídeo:

```bash
export UV_LINK_MODE=copy
uv sync --extra gpu --extra video --group dev
```

O modo `copy` evita o aviso de hardlink quando o projeto está em `/mnt/c` e o cache do `uv` está
no sistema de arquivos do Linux.

O TensorFlow 2.21 possui um problema conhecido de descoberta das bibliotecas CUDA instaladas
por `tensorflow[and-cuda]`. Depois do `uv sync`, crie os links recomendados pela documentação do
TensorFlow. Os comandos abaixo descobrem automaticamente o ambiente virtual usado pelo `uv`:

```bash
TF_DIR=$(uv run python -c 'import os, tensorflow; print(os.path.dirname(tensorflow.__file__))')
ENV_BIN=$(dirname "$(uv run python -c 'import sys; print(sys.executable)')")
NVIDIA_DIR=$(uv run python -c 'import nvidia; print(next(iter(nvidia.__path__)))')

pushd "$TF_DIR"
ln -svf ../nvidia/*/lib/*.so* .
popd

PTXAS=$(find "$NVIDIA_DIR/cuda_nvcc" -type f -name ptxas -print -quit)
ln -sf "$PTXAS" "$ENV_BIN/ptxas"
```

No TensorFlow 2.21, os links podem não ser suficientes porque o `RPATH` do wheel não contém
todos os diretórios NVIDIA. Ainda na raiz do projeto, adicione as bibliotecas da virtualenv ao
loader do Linux:

```bash
NVIDIA_DIR=$(uv run python -c 'import nvidia; print(next(iter(nvidia.__path__)))')

LIB_PATHS=$(find "$NVIDIA_DIR" \
  -mindepth 2 \
  -maxdepth 2 \
  -type d \
  -name lib \
  -print | paste -sd: -)

export LD_LIBRARY_PATH="$LIB_PATHS:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

O `export LD_LIBRARY_PATH` deve ser repetido em cada novo terminal usado para tuning,
treinamento, avaliação ou gravação. Confira os caminhos carregados com:

```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n'
```

Verifique se o TensorFlow registrou a GPU:

```bash
uv run python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

O resultado deve conter `PhysicalDevice` com `device_type='GPU'`. Confirme também que uma
operação real foi alocada na GPU:

```bash
uv run python -c "import tensorflow as tf; a=tf.random.normal((4096,4096)); b=tf.matmul(a,a); print(b.device)"
```

O dispositivo impresso deve terminar em `/device:GPU:0`. Se o primeiro teste não mostrar a GPU,
não inicie o tuning: confirme novamente `nvidia-smi`, os pacotes com `uv pip list | grep nvidia`
e repita a criação dos links.

Nas GPUs RTX série 50, o TensorFlow 2.21 pode informar que não possui kernels CUDA nativos para
`compute capability 12.0a`. Nesse caso, os kernels serão compilados de PTX na primeira execução,
que pode demorar vários minutos. O aviso não indica uso de CPU se o teste mostrar
`/device:GPU:0`; aguarde a primeira compilação sem interromper o processo.

Os links ficam dentro da virtualenv e precisam ser recriados caso ela seja removida. Consulte a
[instalação oficial do TensorFlow](https://www.tensorflow.org/install/pip) para detalhes.

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
