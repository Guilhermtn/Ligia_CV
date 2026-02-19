# Pré-processamento e transformações de imagem.

# Como vimos no EDA, as imagens possuem resoluções e aspect ratios variados.
# Para alimentar redes neurais, padronizamos o tamanho.

# - **Treino:** inclui augmentations leves (ex.: rotação pequena e flip horizontal),
#   visando robustez a variações de aquisição.
# - **Validação/Teste:** apenas resize + normalização, para medir desempenho de forma estável.

# Como utilizaremos modelos pré-treinados, aplicamos normalização compatível com ImageNet.
# Além disso, aplicamos augmentation apenas no treino para melhorar generalização,
# mantendo validação/teste determinísticos.


from torchvision import transforms

# Constantes de normalização ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_tfms(img_size: int = 224) -> transforms.Compose:
    # Transformações do treino (com augmentation leve).

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),         # padroniza o tamanho
        transforms.RandomHorizontalFlip(p=0.5),          # variação leve (não altera anatomia verticalmente)
        transforms.RandomRotation(degrees=10),           # rotação pequena (simula variação de posicionamento)
        transforms.ToTensor(),                           # converte para tensor [0,1]
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # normalização ImageNet
    ])


def get_valid_tfms(img_size: int = 224) -> transforms.Compose:
    # Transformações para validação e teste (sem augmentation).

    # Aplica apenas resize + normalização ImageNet para medir desempenho de forma estável.

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
