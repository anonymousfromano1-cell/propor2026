"""
Script principal para Structured Sentiment Analysis com otimização via Optuna.
"""

from transformers import AutoTokenizer
import pandas as pd
from transformers import DataCollatorForTokenClassification
import numpy as np
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.model_selection import StratifiedKFold
from datasets import Dataset
import optuna
from optuna.storages import RDBStorage
import torch
import random
import os

# Importar funções auxiliares
from utils import (
    tokenize_and_align_labels,
    get_majority_label,
    compute_metrics,
    collect_fold_metrics,
    export_all_results,
)

# ============================================================================
# CONFIGURAÇÕES
# ============================================================================

SEED = 42
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
JSON_PATH = "./dataset.json"
N_FOLDS = 5
N_TRIALS = 20
RESULTS_DIR = "./results"

# Reprodutibilidade
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# Labels
label_list = [
    "O",
    "B-ASPECT", "I-ASPECT",
    "B-HOLDER", "I-HOLDER",
    "B-EXPRESSION-POS", "I-EXPRESSION-POS",
    "B-EXPRESSION-NEG", "I-EXPRESSION-NEG",
    "B-EXPRESSION-NEU", "I-EXPRESSION-NEU",
]

id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
labels_metric = ["HOLDER", "ASPECT", "EXP"]

# ============================================================================
# CARREGAMENTO DE DADOS
# ============================================================================

print("Carregando dados...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
data = pd.read_json(JSON_PATH)

# Processamento
records = data.to_dict("records")
processed = [tokenize_and_align_labels(r, tokenizer) for r in records]
df = pd.DataFrame(processed)
df["stratify_label"] = df["labels"].apply(get_majority_label)

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

print(f"Dataset carregado: {len(df)} amostras")

# ============================================================================
# CONFIGURAÇÃO DO EXPERIMENTO
# ============================================================================

CONFIG = {
    "model_name": MODEL_NAME,
    "n_folds": N_FOLDS,
    "seed": SEED,
    "dataset_size": len(df),
}


def create_compute_metrics():
    """Factory para criar função de métricas com closure."""
    def _compute_metrics(p):
        return compute_metrics(p, id2label, labels_metric)
    return _compute_metrics


def create_model():
    """Factory para criar modelo."""
    return AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )


# ============================================================================
# OPTUNA OBJECTIVE
# ============================================================================

def objective(trial):
    """Função objetivo do Optuna com cross-validation."""
    torch.cuda.empty_cache()

    # Hiperparâmetros
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [4, 8, 16])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    epochs = trial.suggest_int("num_train_epochs", 3, 10)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)

    # Cross-validation
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_scores = []
    fold_metrics = {key: [] for key in [
        "macro_span_f1", "holder_f1", "aspect_f1", "expression_f1",
        "holder_precision", "holder_recall",
        "aspect_precision", "aspect_recall",
        "expression_precision", "expression_recall",
        "acc_token_level", "f1_token_level",
        "precision_token_level", "recall_token_level",
        "F1_Span_NEG", "F1_Span_POS", "F1_Span_NEU"
    ]}

    X = df.drop(columns=["stratify_label"])
    y = df["stratify_label"]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
        print(f"\n{'='*50}")
        print(f"Trial {trial.number} - Fold {fold + 1}/{N_FOLDS}")
        print(f"{'='*50}")

        train_fold = X.iloc[train_idx].reset_index(drop=True)
        val_fold = X.iloc[val_idx].reset_index(drop=True)

        train_dataset = Dataset.from_pandas(train_fold)
        val_dataset = Dataset.from_pandas(val_fold)

        model = create_model()

        args = TrainingArguments(
            output_dir=f"./ssa_trial_{trial.number}_fold_{fold}",
            learning_rate=learning_rate,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="macro_span_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="none",
            logging_steps=50,
            seed=SEED,
            fp16=torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=create_compute_metrics(),
        )

        trainer.train()
        eval_results = trainer.evaluate()

        fold_score = eval_results["eval_macro_span_f1"]
        fold_scores.append(fold_score)

        # Armazenar todas as métricas de cada fold
        for key in fold_metrics:
            fold_metrics[key].append(eval_results.get(f"eval_{key}", 0.0))

        print(f"Fold {fold + 1} macro_span_f1: {fold_score:.4f}")

        # Limpar memória
        del model, trainer
        torch.cuda.empty_cache()

        # Early pruning
        current_mean = np.mean(fold_scores)
        trial.report(current_mean, step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Calcular e salvar médias e desvios
    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    print(f"\n{'='*50}")
    print(f"Trial {trial.number} Results:")
    print(f"Mean macro_span_f1: {mean_score:.4f} ± {std_score:.4f}")
    print(f"{'='*50}")

    # Salvar métricas no trial
    trial.set_user_attr("cv_mean_score", float(mean_score))
    trial.set_user_attr("cv_std_score", float(std_score))
    trial.set_user_attr("fold_scores", fold_scores)

    for key, values in fold_metrics.items():
        trial.set_user_attr(f"mean_{key}", float(np.mean(values)))
        trial.set_user_attr(f"std_{key}", float(np.std(values)))

    return mean_score


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print(f"Iniciando otimização com {N_FOLDS}-fold cross-validation")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Setup Optuna
    optuna_storage = RDBStorage("sqlite:///optuna_trials.db")
    
    study = optuna.create_study(
        study_name="ssa_cv",
        direction="maximize",
        storage=optuna_storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )

    # Otimização
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True
    )

    # Exportação de resultados
    print("\n" + "=" * 60)
    print("EXPORTANDO RESULTADOS PARA ARTIGO CIENTÍFICO")
    print("=" * 60)
    
    best_metrics = collect_fold_metrics(study.best_trial)
    results_df = pd.DataFrame([best_metrics])
    
    export_all_results(study, results_df, CONFIG, RESULTS_DIR)
    
    # Resumo final
    print("\n" + "=" * 60)
    print("MELHORES HIPERPARÂMETROS ENCONTRADOS:")
    print("=" * 60)
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best score: {study.best_value:.4f}")
    print(f"Parameters: {study.best_params}")
    print(f"CV Std: {study.best_trial.user_attrs.get('cv_std_score', 'N/A'):.4f}")
