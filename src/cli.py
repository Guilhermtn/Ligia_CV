#!/usr/bin/env python3

# CLI para geração de submission.csv sem abrir o notebook.

# Fluxo:
# 1. Lê o dataset de teste (test.csv + imagens)
# 2. Carrega os checkpoints dos 5 folds
# 3. Roda inferência com ensemble (média das probabilidades)
# 4. Salva submission.csv

# Uso:
#     # Básico (usa defaults)
#     python -m src.cli --data-dir /path/to/ligia-compviz --models-dir ./models

#     # Completo
#     python -m src.cli \
#         --data-dir /path/to/ligia-compviz \
#         --models-dir ./models \
#         --output submission.csv \
#         --batch-size 32 \
#         --num-folds 5 \
#         --device cuda

# Exemplo:
#     cd /home/jose/Documentos/Ligia_CV
#     python -m src.cli --data-dir ~/ligia-compviz --models-dir ./models


import argparse
import sys
from pathlib import Path

import torch

from src.data import load_test_df, make_test_loader
from src.inference import ensemble_predict, save_submission
from src.model import build_model
from src.transforms import get_valid_tfms


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gera submission.csv para a competição de pneumonia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplo de uso:
    python -m src.cli --data-dir ~/ligia-compviz --models-dir ./models

O arquivo submission.csv será salvo no diretório atual ou no caminho especificado.
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Diretório raiz do dataset (contém train.csv, test.csv, etc.)",
    )

    parser.add_argument(
        "--models-dir",
        type=str,
        required=True,
        help="Diretório com os checkpoints (best_model_fold0.pth, etc.)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="submission.csv",
        help="Caminho para salvar o arquivo de submissão (default: submission.csv)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Tamanho do batch para inferência (default: 32)",
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Número de folds/checkpoints para ensemble (default: 5)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Número de workers do DataLoader (default: 2)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cuda/cpu). Auto-detecta se não especificado.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Tamanho da imagem de entrada (default: 224)",
    )

    return parser.parse_args()


def get_device(device_arg: str | None) -> torch.device:
    """Determina o dispositivo a ser usado."""
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkpoint_paths(models_dir: Path, num_folds: int) -> list[str]:
    """Retorna lista de caminhos para os checkpoints."""
    paths = []
    for i in range(num_folds):
        ckpt_path = models_dir / f"best_model_fold{i}.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")
        paths.append(str(ckpt_path))
    return paths


def main():
    """Função principal do CLI."""
    args = parse_args()

    # Resolve paths
    data_dir = Path(args.data_dir).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    print("=" * 60)
    print("Geração de Submission - Classificação de Pneumonia")
    print("=" * 60)
    print(f"Data dir:    {data_dir}")
    print(f"Models dir:  {models_dir}")
    print(f"Output:      {output_path}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Num folds:   {args.num_folds}")
    print(f"Image size:  {args.img_size}")

    # Determina dispositivo
    device = get_device(args.device)
    print(f"Device:      {device}")
    print("=" * 60)

    # Valida checkpoints
    print("\n[1/4] Verificando checkpoints...")
    checkpoint_paths = get_checkpoint_paths(models_dir, args.num_folds)
    print(f"✅ {len(checkpoint_paths)} checkpoints encontrados")

    # Carrega dataset de teste
    print("\n[2/4] Carregando dataset de teste...")
    df_test = load_test_df(data_dir)
    print(f"✅ {len(df_test)} amostras de teste carregadas")

    # Cria DataLoader
    print("\n[3/4] Preparando DataLoader...")
    valid_tfms = get_valid_tfms(img_size=args.img_size)
    test_loader = make_test_loader(
        df_test,
        valid_tfms=valid_tfms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"✅ DataLoader pronto ({len(test_loader)} batches)")

    # Roda inferência com ensemble
    print(f"\n[4/4] Rodando inferência (ensemble de {args.num_folds} folds)...")
    test_probs = ensemble_predict(
        model_builder_fn=build_model,
        checkpoint_paths=checkpoint_paths,
        loader=test_loader,
        device=device,
    )
    print(f"✅ Inferência concluída ({len(test_probs)} predições)")

    # Salva submission
    print("\n" + "=" * 60)
    print("Salvando submissão...")
    save_submission(df_test["id"], test_probs, str(output_path))

    print("\n" + "=" * 60)
    print("✅ Processo concluído com sucesso!")
    print(f"Arquivo salvo em: {output_path}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
