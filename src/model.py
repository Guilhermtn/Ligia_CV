# Arquitetura do modelo para classificação de pneumonia.

# Usamos Transfer Learning com uma rede pré-treinada em ImageNet.
# Isso é útil pois o dataset não é gigantesco e o modelo pré-treinado já
# possui filtros visuais úteis (bordas, texturas, padrões), acelerando a
# convergência.

# A saída do modelo é um **score contínuo** (logit), que será convertido
# em probabilidade via sigmoide. A métrica de validação é a **ROC-AUC**,
# compatível com a avaliação da competição.


import torch.nn as nn
import torchvision.models as models


def build_model(pretrained: bool = True) -> nn.Module:
    if pretrained:
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    else:
        weights = None

    # Carrega EfficientNet-B0 pré-treinada
    model = models.efficientnet_b0(weights=weights)

    # Troca a camada final para saída binária (1 logit)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    return model
