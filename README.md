# LIGIA - Visão Computacional  
## Detecção de Pneumonia em Raios-X de Tórax

Implementação completa do pipeline de classificação binária (NORMAL vs PNEUMONIA) usando Transfer Learning, validação cruzada (5-fold), ensemble e interpretabilidade (Grad-CAM).  
O repositório contém os **artefatos finais do modelo** (`.pth`) e instruções para **reproduzir a geração do submission.csv**.

---

## 📁 Estrutura do Repositório

O projeto está organizado da seguinte forma para facilitar a reprodutibilidade e a organização do pipeline de Visão Computacional:

* **`notebooks/`**: Notebook principal contendo toda a implementação da solução, incluindo preparação dos dados, definição do modelo, treinamento com validação cruzada (5-fold), avaliação, interpretabilidade com Grad-CAM e geração do arquivo de submissão.

* **`models/`**: Contém os modelos treinados serializados (.pth), correspondentes aos pesos finais de cada fold utilizados para gerar as previsões finais.

* **`ligia-compviz/`**: (não versionado): Pasta esperada para o dataset extraído, contendo imagens de treino/teste e arquivos CSV fornecidos pela competição.

* **`requirements.txt`**: Arquivo de configuração contendo as bibliotecas necessárias para execução do projeto.

---

## ▶️ Reprodução dos Experimentos (Google Colab + Google Drive)

O projeto foi estruturado para execução no Google Colab utilizando o Google Drive para armazenamento do dataset e geração dos resultados.

Esta é a forma recomendada para reprodução integral dos experimentos.

---

### 1️⃣ Preparação do Dataset no Google Drive

1. Faça o download do dataset da competição.
2. Extraia o conteúdo.
3. No Google Drive, crie a seguinte estrutura:

```
MyDrive/Ligia_compviz/
├── competicao.ipynb
├── ligia-compviz/
│   ├── train.csv
│   ├── test.csv
│   ├── train/
│   │   ├── NORMAL/
│   │   └── PNEUMONIA/
│   └── test_images/
│       └── test_images/
```
Recomenda-se criar a pasta no Google Drive com o nome: Ligia_compviz

Entretanto, caso utilize outro nome ou outro local no Drive, basta ajustar manualmente a variável no início do notebook:

```python
PROJECT_DIR = "/content/drive/MyDrive/Ligia_compviz"  # ajuste para sua pasta
DATA_DIR = f"{PROJECT_DIR}/ligia-compviz"
```
---

### 2️⃣ Abrir o Notebook no Colab

1. Acesse o Google Colab.
2. Faça upload do arquivo:
   notebooks/competicao.ipynb
3. Ative GPU (opcional, mas recomendado):
   Ambiente de execução → Alterar o tipo de Ambiente de Execução → GPU

#### 🔧 Uso de GPU

O projeto foi executado utilizando **Google Colab com GPU T4**.

- ⏱ Tempo médio de execução completa: aproximadamente **20 minutos**
- 💻 Em CPU, o tempo de execução pode aumentar consideravelmente
- 🚀 O uso de GPU é fortemente recomendado para reduzir o tempo de treinamento

Caso a execução seja realizada apenas em CPU, o pipeline continuará funcionando normalmente, porém com maior tempo de processamento.

---

### 3️⃣ Montar o Google Drive

Execute a célula inicial responsável por montar o Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```
Ao executar essa célula:

* Será solicitado que você autorize o acesso ao seu Google Drive

* Após a autorização, o notebook continuará a execução normalmente

---

### 4️⃣ Execução do Notebook

Após montar o Drive, você pode:

Executar célula por célula, acompanhando cada etapa do pipeline
ou

Executar tudo de uma vez em:
```
Runtime → Run all
```
⚠️ Recomendação:
Caso opte por executar tudo de uma vez, recomenda-se que a sessão esteja limpa para evitar conflitos ou variáveis previamente carregadas.
Para garantir isso:
```
Runtime → Restart and run all
```
Isso assegura que o experimento será reproduzido do zero.

---

### 5️⃣ Geração do Arquivo de Submissão

Ao final da execução completa do notebook, será gerado automaticamente o arquivo:
```
submission.csv
```
O arquivo será salvo em dois locais:

* **`/content/submission.csv`**(diretório temporário do ambiente Colab)

* Dentro da pasta definida em PROJECT_DIR no Google Drive

O arquivo salvo corresponde exatamente ao utilizado para submissão na competição.

