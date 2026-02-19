# inference.py - Funções de inferência para classificação de pneumonia em raios-X.

# Módulo responsável por:
# - Predição de probabilidades (sigmoid sobre logits)
# - Carregamento de checkpoints salvos (.pth)
# - Ensemble de múltiplos folds (média das probabilidades)
# - Geração do arquivo de submissão para Kaggle

# Uso típico:
#     from src.inference import predict_proba, load_checkpoint, ensemble_predict, save_submission

#     # Predição simples com um modelo
#     probs = predict_proba(model, test_loader, device)

#     # Carregamento de checkpoint
#     model = load_checkpoint(model, "best_model_fold0.pth", device)

#     # Ensemble de 5 folds
#     probs = ensemble_predict(build_model, checkpoint_paths, test_loader, device)

#     # Salvar submissão
#     save_submission(df_test["id"], probs, "submission.csv")

import torch
import numpy as np
import pandas as pd


@torch.no_grad()
def predict_proba(model, loader, device):
    
    # Retorna probabilidades (sigmoid) para todos os itens do loader, na ordem.

    # Esta função aplica sigmoid nos logits de saída do modelo para obter
    # probabilidades de pneumonia (classe 1). Funciona tanto com loaders
    # de treino/validação (que retornam x, y) quanto loaders de teste
    # (que retornam apenas x).

    model.eval()
    all_probs = []

    for batch in loader:
        # batch pode ser (x, y) ou só x (teste)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, _ = batch
        else:
            x = batch

        x = x.to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        all_probs.append(probs)

    return np.concatenate(all_probs)


def load_checkpoint(model, checkpoint_path, device):
    
    # Carrega os pesos de um checkpoint salvo (.pth) no modelo.

    # Esta função carrega o state_dict de um arquivo .pth e aplica no modelo
    # fornecido. O modelo é movido para o dispositivo especificado e colocado
    # em modo de avaliação (eval).

    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    return model


def ensemble_predict(model_builder_fn, checkpoint_paths, loader, device):
    
    # Gera predições usando ensemble de múltiplos checkpoints (média das probabilidades).

    # Para cada checkpoint:
    # 1. Instancia um novo modelo
    # 2. Carrega os pesos do checkpoint
    # 3. Gera probabilidades com predict_proba
    # 4. Acumula as probabilidades

    # Ao final, retorna a média das probabilidades de todos os folds.

    all_probs = []

    for ckpt_path in checkpoint_paths:
        model = model_builder_fn()
        model = load_checkpoint(model, ckpt_path, device)
        probs = predict_proba(model, loader, device)
        all_probs.append(probs)

    # Média das probabilidades dos folds
    return np.mean(np.stack(all_probs, axis=0), axis=0)


def save_submission(ids, probs, output_path, validate=True):
    
    # Cria e salva o arquivo de submissão no formato exigido pela competição.

    # O arquivo contém duas colunas:
    # - id: identificador da amostra (como em test.csv)
    # - target: probabilidade de pneumonia (float entre 0 e 1)

    # Opcionalmente, realiza sanity checks antes de salvar:
    # - Verifica se não há IDs duplicados
    # - Verifica se valores de target estão em [0, 1]

    submission = pd.DataFrame({
        "id": ids,
        "target": np.asarray(probs).astype(float)
    })

    if validate:
        assert submission["id"].nunique() == len(submission), \
            "IDs duplicados na submissão."
        assert submission["target"].between(0, 1).all(), \
            "Existem valores de target fora de [0, 1]."
        assert len(ids) == len(probs), \
            "Quantidade de IDs diferente do número de predições."

    submission.to_csv(output_path, index=False)
    print(f"✅ {output_path} salvo com sucesso!")
    print(f"Shape: {submission.shape}")
    print(f"Target min/mean/max: {submission['target'].min():.4f} / "
          f"{submission['target'].mean():.4f} / {submission['target'].max():.4f}")

    return submission
