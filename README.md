# LIGIA - Visão Computacional  
## Detecção de Pneumonia em Raios-X de Tórax

Implementação completa do pipeline de classificação binária (NORMAL vs PNEUMONIA) usando Transfer Learning com EfficientNet-B0, validação cruzada (5-fold) e ensemble.  
O repositório contém os **artefatos finais do modelo** (`.pth`) e instruções para **reproduzir a geração do submission.csv**.

---

## 📁 Estrutura do Repositório

```
Ligia_CV/
├── src/                    # Código-fonte modularizado
│   ├── cli.py              # Interface de linha de comando para inferência
│   ├── data.py             # Dataset, DataLoaders e validação de estrutura
│   ├── inference.py        # Funções de predição e ensemble
│   ├── model.py            # Arquitetura do modelo (EfficientNet-B0)
│   └── transforms.py       # Transformações de imagem (augmentation/normalização)
├── models/                 # Checkpoints dos 5 folds treinados
│   ├── best_model_fold0.pth
│   ├── best_model_fold1.pth
│   ├── best_model_fold2.pth
│   ├── best_model_fold3.pth
│   └── best_model_fold4.pth
├── notebooks/              # Notebooks de desenvolvimento e competição
│   ├── competicao.ipynb    # Notebook principal da competição
│   └── train.ipynb         # Notebook de treinamento
├── ligia-compviz/          # Dataset (não versionado)
│   ├── train.csv
│   ├── test.csv
│   ├── train/train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test_images/test_images/
├── requirements.txt        # Dependências do projeto
└── README.md
```

---

## ⬇️ Obtendo o Projeto

### 1️⃣ Clonar o Repositório

```bash
git clone <LINK_DO_SEU_REPOSITORIO>
cd Ligia_CV
```

### 2️⃣ Preparar o Dataset

Baixe o dataset da competição e extraia na pasta `ligia-compviz/` dentro do repositório, mantendo a estrutura esperada.

⚠️ **Verificações importantes:**
- O nome da pasta deve ser **exatamente** `ligia-compviz` (tudo minúsculo, com hífen)
- Verifique se não há espaços ou caracteres invisíveis no nome da pasta
- Certifique-se de que os arquivos dentro do dataset também não contêm espaços nos nomes

---

## 🐍 Configuração do Ambiente

### Requisitos

⚠️ **Atenção:** Este projeto requer **Python 3.11**. As bibliotecas utilizadas (PyTorch, torchvision) podem não ter suporte para versões mais recentes como Python 3.13.

### 1️⃣ Verificar a Versão do Python

```bash
python --version
```

Se você possui múltiplas versões instaladas, especifique a versão correta nos comandos:

```bash
python3.11 --version
```

### 2️⃣ Criar o Ambiente Virtual

```bash
# Se python aponta para 3.11:
python -m venv .venv

# Ou, se precisar especificar a versão:
python3.11 -m venv .venv
```

### 3️⃣ Ativar o Ambiente Virtual

**Linux/macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.\.venv\Scripts\activate
```

### 4️⃣ Instalar as Dependências

```bash
pip install -r requirements.txt
```

---

## ▶️ Execução do CLI

O CLI permite gerar o arquivo `submission.csv` diretamente pela linha de comando.

### Uso Básico

```bash
python -m src.cli --data-dir ./ligia-compviz --models-dir ./models
```

### Uso Completo (com todas as opções)

```bash
python -m src.cli \
    --data-dir ./ligia-compviz \
    --models-dir ./models \
    --output submission.csv \
    --batch-size 32 \
    --num-folds 5 \
    --device cuda
```

### Parâmetros Disponíveis

| Parâmetro | Descrição | Default |
|-----------|-----------|---------|
| `--data-dir` | Diretório raiz do dataset | *obrigatório* |
| `--models-dir` | Diretório com os checkpoints (.pth) | *obrigatório* |
| `--output` | Caminho do arquivo de saída | `submission.csv` |
| `--batch-size` | Tamanho do batch para inferência | `32` |
| `--num-folds` | Número de folds para ensemble | `5` |
| `--num-workers` | Workers do DataLoader | `2` |
| `--device` | Dispositivo (cuda/cpu) | auto-detecta |
| `--img-size` | Tamanho da imagem de entrada | `224` |

### Exemplo de Saída

```
============================================================
Geração de Submission - Classificação de Pneumonia
============================================================
Data dir:    /home/user/Ligia_CV/ligia-compviz
Models dir:  /home/user/Ligia_CV/models
Output:      /home/user/Ligia_CV/submission.csv
Batch size:  32
Num folds:   5
Image size:  224
Device:      cuda
============================================================

[1/4] Verificando checkpoints...
✅ 5 checkpoints encontrados

[2/4] Carregando dataset de teste...
✅ X amostras de teste carregadas

[3/4] Preparando DataLoader...
✅ DataLoader pronto (Y batches)

[4/4] Rodando inferência (ensemble de 5 folds)...
✅ Inferência concluída (X predições)

============================================================
Salvando submissão...

============================================================
✅ Processo concluído com sucesso!
Arquivo salvo em: /home/user/Ligia_CV/submission.csv
============================================================
```

---

## 🔧 Uso de GPU

O uso de GPU é recomendado para acelerar a inferência:
- O CLI detecta automaticamente a disponibilidade de CUDA
- Para forçar CPU: `--device cpu`
- Para forçar GPU: `--device cuda`

---

## 🏋️ Treinamento dos Modelos (Opcional - Google Colab)

Os checkpoints (`.pth`) já estão incluídos no repositório. Esta seção é **opcional** e serve apenas para quem deseja **retreinar os modelos do zero** para verificação.

⚠️ **Atenção:** O notebook `train.ipynb` foi desenvolvido para execução no **Google Colab com GPU**. Localmente, sem GPU, o treinamento pode ser extremamente lento.

### Configuração no Google Colab

1. **Criar uma pasta no Google Drive:**
   - Acesse seu Google Drive e crie uma pasta (ex: `ligia-cv`)
   - Recomendação: use nome **minúsculo** e **sem espaços**
   - Verifique se não existe outra pasta com o mesmo nome em `MyDrive/`

2. **Copiar os arquivos necessários:**
   - Copie o arquivo `notebooks/train.ipynb` para a pasta criada
   - Copie a pasta `ligia-compviz/` (dataset) para o **mesmo nível** da pasta
   
   Estrutura esperada:
   ```
   MyDrive/
   └── ligia-cv/              # sua pasta
       ├── train.ipynb        # notebook de treinamento
       └── ligia-compviz/     # dataset
           ├── train.csv
           ├── test.csv
           ├── train/
           └── test_images/
   ```

3. **Ajustar o caminho no notebook:**
   - Abra o `train.ipynb` no Colab
   - Na seção **0.5.2**, ajuste a variável `PROJECT_DIR` para o nome da sua pasta:
   ```python
   PROJECT_DIR = "/content/drive/MyDrive/ligia-cv"  # <- ajuste para sua pasta
   ```

4. **Habilitar GPU:**
   - No Colab: `Ambiente de execução` → `Alterar tipo de ambiente de execução` → `GPU T4`

5. **Executar o notebook:**
   - Execute todas as células sequencialmente
   - Os checkpoints serão salvos automaticamente na pasta do Colab

### Tempo de Execução

Com a GPU T4 gratuita do Google Colab, o treinamento completo (5 folds) leva aproximadamente **23 minutos**.

