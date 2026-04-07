# Laboratório 7: Especialização de LLMs com LoRA e QLoRA 

Pipeline completo de fine-tuning de um modelo de linguagem fundacional (TinyLlama 1.1B) no domínio de programação, utilizando geração de dataset sintético via API do Groq, técnicas de eficiência de parâmetros (PEFT/LoRA) e treinamento supervisionado com SFTTrainer.

---

## Pré-requisitos

- Python 3.8+
- Conta no [Groq](https://console.groq.com) com chave de API
- Google Colab com GPU T4
- Hugging Face `transformers`, `peft`, `trl`, `datasets`, `groq`, `accelerate`

Instale as dependências com:

```bash
pip install transformers==4.46.0 trl==0.11.0 datasets peft accelerate groq pyarrow
```

## Como rodar

Abra o notebook no Google Colab com GPU T4 habilitada e execute as células em ordem. A chave da API do Groq deve estar configurada nos **Secrets** do Colab com o nome `GROQ_API_KEY`.

---

## O que o código faz

### Componentes implementados

| Componente | Descrição |
|---|---|
| `gerar_par(topico)` | Chama a API do Groq (Llama 3.3 70B) para gerar um par pergunta/resposta sobre um tópico de programação e retorna um dicionário JSON |
| Cache do dataset | Verifica se os arquivos `.jsonl` já existem antes de gerar, evitando chamadas desnecessárias à API |
| `formatar_prompt(exemplo)` | Formata cada amostra no template `### Pergunta / ### Resposta` e remove as colunas originais |
| `LoraConfig` | Configura os adaptadores LoRA com rank, alpha e dropout |
| `SFTTrainer` | Orquestra o loop de treinamento supervisionado com suporte a PEFT |

### Fluxo de execução

**1. Geração do dataset sintético (Passo 1)**
Um conjunto de 25 tópicos de programação é expandido para 50 amostras e enviado um a um para o modelo `llama-3.3-70b-versatile` via API do Groq. Cada resposta é parseada como JSON e acumulada em uma lista. Ao final, o dataset é embaralhado e dividido em 90% treino / 10% teste, salvos em `dataset_treino.jsonl` e `dataset_teste.jsonl`.

**2. Carregamento do modelo (Passo 2)**
O modelo `TinyLlama/TinyLlama-1.1B-Chat-v1.0` é carregado em `float16` com `device_map="auto"` para aproveitar a GPU disponível. O token de padding é definido como o token EOS.

**3. Configuração do LoRA (Passo 3)**
Os adaptadores LoRA são configurados via `LoraConfig` com os hiperparâmetros definidos pelo enunciado. O LoRA congela os pesos originais do modelo e treina apenas as matrizes de decomposição de baixo rank, reduzindo drasticamente o número de parâmetros treináveis.

**4. Treinamento supervisionado (Passo 4)**
O dataset é carregado e formatado com o template de instrução. O `SFTTrainer` gerencia o loop de treinamento com o otimizador AdamW nativo do PyTorch, scheduler cosseno e warmup. Ao final, o adaptador LoRA é salvo em `./lora_adaptador`.

---

## Configuração padrão

```python
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# LoRA
r            = 16     # Rank das matrizes de decomposição
lora_alpha   = 16     # Fator de escala dos novos pesos
lora_dropout = 0.1    # Dropout para evitar overfitting

# Treinamento
num_train_epochs          = 1
per_device_train_batch_size = 2
learning_rate             = 2e-4
lr_scheduler_type         = "cosine"
warmup_steps              = 10
max_seq_length            = 512
```

---

## Saída esperada

O adaptador adapter_model.safetensors é salvo

---

## Observações

- O enunciado original pede uso da api do GPT, porém por questão de custo, foi usada uma api com custo inicial gratuito do Groq
---

## Uso de IA generativa

- Geração do dataset sintético via API do Groq (Llama 3.3 70B).
- Auxílio na depuração de compatibilidade de versões entre `trl`, `transformers`, `bitsandbytes` e CUDA.
- Geração deste README.
