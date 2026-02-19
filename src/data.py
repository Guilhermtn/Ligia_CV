#Construção do Dataset e validações de estrutura.

# Este módulo concentra tudo relacionado a:
# - Leitura de CSV
# - Montagem de DataFrame com paths
# - Validação de arquivos
# - Dataset e DataLoaders

# Criado para:
# - Ler imagens a partir da coluna `path`
# - Retornar `(imagem, label)` no treino/validação
# - Retornar apenas `imagem` no teste

# A conversão para `RGB` é utilizada para compatibilidade direta com modelos
# pré-treinados em ImageNet.

import os
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset


def sanity_check(data_dir: str | Path) -> None:
    
    # Valida se os arquivos/pastas essenciais existem antes de seguir.

    # Isso evita erros comuns como:
    # - dataset na pasta errada
    # - dataset ainda zipado
    # - estrutura diferente do esperado

    data_dir = Path(data_dir)

    expected = {
        "train.csv": data_dir / "train.csv",
        "test.csv": data_dir / "test.csv",
        "NORMAL dir": data_dir / "train" / "train" / "NORMAL",
        "PNEUMONIA dir": data_dir / "train" / "train" / "PNEUMONIA",
        "test_images dir": data_dir / "test_images" / "test_images",
    }

    missing = [name for name, path in expected.items() if not path.exists()]
    if missing:
        details = ", ".join(missing)
        raise FileNotFoundError(f"Estrutura incompleta em '{data_dir}'. Ausentes: {details}")

    # Contagens rápidas
    n_norm = len(os.listdir(expected["NORMAL dir"]))
    n_pne = len(os.listdir(expected["PNEUMONIA dir"]))
    n_test = len(os.listdir(expected["test_images dir"]))

    print("✅ Estrutura mínima OK.")
    print(f"📊 Contagens: NORMAL={n_norm} | PNEUMONIA={n_pne} | TEST={n_test}")


def load_train_df(data_dir: str | Path, seed: int = 42) -> pd.DataFrame:
    
    # Constrói o DataFrame de treino a partir do train.csv.

    # Construímos `df_train` a partir do `train.csv`, garantindo que:
    # - cada `id` tenha um arquivo correspondente na pasta de imagens;
    # - o `label` utilizado é o rótulo oficial fornecido pela competição.

    # Também criamos a coluna `path` com o caminho completo da imagem para uso
    # no DataLoader.

    data_dir = Path(data_dir)

    train_csv_path = data_dir / "train.csv"
    pne_dir = data_dir / "train" / "train" / "PNEUMONIA"
    nor_dir = data_dir / "train" / "train" / "NORMAL"

    train_csv = pd.read_csv(train_csv_path)

    # Conjunto de arquivos existentes nas pastas (ids com extensão)
    files_pne = set(os.listdir(pne_dir))
    files_nor = set(os.listdir(nor_dir))
    files_all = files_pne | files_nor

    # Checagem: todo id do train.csv precisa existir nas pastas
    missing_train_files = set(train_csv["id"]) - files_all
    assert len(missing_train_files) == 0, "Existem ids no train.csv sem arquivo correspondente nas pastas."

    # Função para criar o path correto de cada id
    def build_train_path(img_id: str) -> str | None:
        # Decide se o arquivo está em PNE_DIR ou NOR_DIR
        if img_id in files_pne:
            return str(pne_dir / img_id)
        elif img_id in files_nor:
            return str(nor_dir / img_id)
        else:
            # Isso não deveria acontecer por causa do assert acima
            return None

    df_train = train_csv.copy()
    df_train["path"] = df_train["id"].apply(build_train_path)

    # Checagem final: nenhum path pode ficar nulo
    assert df_train["path"].isna().sum() == 0, "Alguns paths ficaram nulos. Verifique as pastas."

    # Embaralha para não ficar com blocos ordenados
    df_train = df_train.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_train


def load_test_df(data_dir: str | Path) -> pd.DataFrame:
    # Constrói o DataFrame de teste a partir do test.csv.

    # No conjunto de teste, os rótulos não são fornecidos. O objetivo é:
    # - associar cada `id` do `test.csv` ao respectivo arquivo na pasta test_images;
    # - garantir que os `ids` utilizados na submissão sigam exatamente o formato esperado.

    data_dir = Path(data_dir)

    test_csv_path = data_dir / "test.csv"
    test_img_dir = data_dir / "test_images" / "test_images"

    test_csv = pd.read_csv(test_csv_path)

    # Arquivos disponíveis na pasta de teste
    test_files_in_dir = set(os.listdir(test_img_dir))

    # Checagem: todo id do test.csv precisa existir na pasta
    missing_test_files = set(test_csv["id"]) - test_files_in_dir
    assert len(missing_test_files) == 0, "Existem ids no test.csv sem arquivo correspondente em test_images."

    df_test = test_csv.copy()  # mantém o id oficial
    df_test["path"] = df_test["id"].apply(lambda x: str(test_img_dir / x))

    return df_test


def add_folds(df_train: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    
	# Adiciona coluna 'fold' ao DataFrame de treino usando StratifiedKFold.

    # Como o dataset é desbalanceado, utilizamos StratifiedKFold para que cada
    # fold mantenha uma proporção de classes semelhante ao conjunto original.

    # Isso melhora a confiabilidade da validação e reduz variações artificiais
    # entre folds.
     
    df = df_train.copy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    df["fold"] = -1  # coluna para guardar o fold de cada amostra

    for fold, (_, val_idx) in enumerate(skf.split(df["path"], df["label"])):
        df.loc[val_idx, "fold"] = fold

    return df


class XRayDataset(Dataset):
    
    # Dataset customizado para imagens de Raio-X torácico.

    # Criado para:
    # - Ler imagens a partir da coluna `path`
    # - Retornar `(imagem, label)` no treino/validação
    # - Retornar apenas `imagem` no teste

    # A conversão para `RGB` é utilizada para compatibilidade direta com modelos
    # pré-treinados em ImageNet.

    # Args:
    #     df: DataFrame com colunas:
    #         - path (obrigatório)
    #         - label (se has_label=True)
    #     transform: Transformações a aplicar nas imagens.
    #     has_label: Se True, retorna (img, label); caso contrário, apenas img.
    

    def __init__(self, df: pd.DataFrame, transform: Callable | None = None, has_label: bool = True):
        # df: DataFrame com colunas:
        # - path (obrigatório)
        # - label (se has_label=True)
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_label = has_label

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        # Abre imagem e converte para RGB (compatível com modelos ImageNet)
        img = Image.open(row["path"]).convert("RGB")

        # Aplica transformações
        if self.transform:
            img = self.transform(img)

        # Se houver label (treino/validação), retorna (img, label)
        if self.has_label:
            y = torch.tensor(row["label"], dtype=torch.float32)
            return img, y

        # Caso teste, retorna apenas imagem
        return img


def make_loaders(
    df_train: pd.DataFrame,
    train_tfms: Callable,
    valid_tfms: Callable,
    fold: int = 0,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    
    # Cria DataLoaders de treino e validação com separação por fold.

    # Usamos a coluna `fold` para separar treino e validação.

    # Isso garante que:
    # - treino e validação são disjuntos
    # - a validação preserva a distribuição de classes (stratified)
    
    # Separa treino e validação
    df_tr = df_train[df_train["fold"] != fold].reset_index(drop=True)
    df_va = df_train[df_train["fold"] == fold].reset_index(drop=True)

    # Cria datasets
    ds_tr = XRayDataset(df_tr, transform=train_tfms, has_label=True)
    ds_va = XRayDataset(df_va, transform=valid_tfms, has_label=True)

    # DataLoaders
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return dl_tr, dl_va


def make_test_loader(
    df_test: pd.DataFrame,
    valid_tfms: Callable,
    batch_size: int = 32,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> DataLoader:
    
	# Cria DataLoader para o conjunto de teste.

    # Args:
    #     df_test: DataFrame de teste com coluna 'path'.
    #     valid_tfms: Transformações (sem augmentation).
    #     batch_size: Tamanho do batch (padrão: 32).
    #     num_workers: Número de workers para o DataLoader (padrão: 2).
    #     pin_memory: Se True, usa pin_memory para GPU (padrão: True).

    test_dataset = XRayDataset(df_test, transform=valid_tfms, has_label=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return test_loader
