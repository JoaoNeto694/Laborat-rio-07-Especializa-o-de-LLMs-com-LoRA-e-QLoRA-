import os
import json
import random
from groq import Groq
from google.colab import userdata
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Passo 1: Engenharia de Dados Sintéticos
# Usei a api do Groq porque ela incialmente não tem custo
GROQ_API_KEY = userdata.get('GROQ_API_KEY')
client = Groq(api_key=GROQ_API_KEY)

# Lista de tópicos de programação (gerada por IA)
TOPICS = [
    "listas em Python", "dicionários em Python", "funções recursivas",
    "orientação a objetos", "herança em Python", "decorators",
    "generators e iterators", "list comprehension", "tratamento de exceções",
    "manipulação de arquivos", "expressões regulares", "algoritmos de ordenação",
    "busca binária", "estruturas de dados", "complexidade de algoritmos",
    "APIs REST", "requisições HTTP com requests", "JSON em Python",
    "testes unitários com pytest", "virtual environments",
    "git básico", "SQL básico", "banco de dados com SQLite",
    "programação funcional", "lambda functions",
]

# Função para gerar um par de pergunta e resposta usando a API do Groq
def gerar_par(topico: str) -> dict:
    prompt = (
        f"Gere um par de pergunta e resposta sobre o tópico de programação: '{topico}'. "
        "Responda APENAS com um JSON válido no formato: "
        '{"prompt": "<pergunta>", "response": "<resposta detalhada>"}. '
        "Sem texto adicional fora do JSON."
    )
    chat = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    content = chat.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


# Garantir que ele não gere o dataset toda vez que rodar o código
if os.path.exists("dataset_treino.jsonl") and os.path.exists("dataset_teste.jsonl"):
    print("Dataset já existe, pulando geração...\n")
else:
    dataset = []
    topicos_expandidos = (TOPICS * 3)[:50]

    for i, topico in enumerate(topicos_expandidos):
        try:
            par = gerar_par(topico)
            dataset.append(par)
            print(f"  [{i+1:02d}/50] OK — {topico}")
        except Exception as e:
            print(f"  [{i+1:02d}/50] ERRO — {topico}: {e}")

    # Divisão 90/10
    random.shuffle(dataset)
    split = int(len(dataset) * 0.9)
    treino = dataset[:split]
    teste = dataset[split:]

    with open("dataset_treino.jsonl", "w", encoding="utf-8") as f:
        for item in treino:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open("dataset_teste.jsonl", "w", encoding="utf-8") as f:
        for item in teste:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# Passo 2: Configuração da Quantização (QLoRA)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# Passo 3: Arquitetura do LoRA
lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
)

# Passo 4: Pipeline de Treinamento
dataset_treino = load_dataset("json", data_files="dataset_treino.jsonl", split="train")

def formatar_prompt(exemplo):
    return {"text": f"### Pergunta:\n{exemplo['prompt']}\n\n### Resposta:\n{exemplo['response']}"}

dataset_treino = dataset_treino.map(formatar_prompt, remove_columns=["prompt", "response"])


training_args = SFTConfig(
    output_dir="./resultados",
    num_train_epochs=1,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=1,
    optim="adamw_torch",
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    fp16=True,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",
    dataset_text_field="text",
    max_seq_length=512,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset_treino,
    peft_config=lora_config,
    args=training_args,
)
trainer.train()

trainer.model.save_pretrained("./lora_adaptador")
