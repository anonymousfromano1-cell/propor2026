"""
Funções auxiliares para exportação de resultados para artigo científico.
"""

import os
import json
import pandas as pd
from datetime import datetime


def format_metric(mean, std, precision=2):
    """
    Formata métrica como média ± desvio padrão em porcentagem.
    
    Args:
        mean: Valor médio (0-1)
        std: Desvio padrão (0-1)
        precision: Casas decimais
        
    Returns:
        String formatada (ex: "75.32 ± 2.14")
    """
    return f"{mean*100:.{precision}f} ± {std*100:.{precision}f}"


def export_latex_table(results_df, results_dir, n_folds, filename="results_table.tex"):
    """
    Exporta tabela de resultados em formato LaTeX.
    
    Args:
        results_df: DataFrame com métricas (mean e std)
        results_dir: Diretório de saída
        n_folds: Número de folds do CV
        filename: Nome do arquivo de saída
        
    Returns:
        String com conteúdo LaTeX
    """
    latex_content = r"""
\begin{table}[htbp]
\centering
\caption{Results of Structured Sentiment Analysis with %d-fold Cross-Validation}
\label{tab:results}
\begin{tabular}{lccc}
\toprule
\textbf{Component} & \textbf{Precision} & \textbf{Recall} & \textbf{F1-Score} \\
\midrule
""" % n_folds

    components = [
        ("Holder", "holder"),
        ("Aspect", "aspect"),
        ("Expression", "expression"),
    ]
    
    for name, key in components:
        p_mean = results_df[f"{key}_precision_mean"].iloc[0]
        p_std = results_df[f"{key}_precision_std"].iloc[0]
        r_mean = results_df[f"{key}_recall_mean"].iloc[0]
        r_std = results_df[f"{key}_recall_std"].iloc[0]
        f1_mean = results_df[f"{key}_f1_mean"].iloc[0]
        f1_std = results_df[f"{key}_f1_std"].iloc[0]
        
        latex_content += f"{name} & {format_metric(p_mean, p_std)} & {format_metric(r_mean, r_std)} & {format_metric(f1_mean, f1_std)} \\\\\n"

    macro_mean = results_df["macro_span_f1_mean"].iloc[0]
    macro_std = results_df["macro_span_f1_std"].iloc[0]
    
    latex_content += r"""\midrule
\textbf{Macro Avg (Span)} & -- & -- & %s \\
\bottomrule
\end{tabular}
\end{table}
""" % format_metric(macro_mean, macro_std)

    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"Tabela LaTeX salva em: {filepath}")
    return latex_content


def export_polarity_latex_table(results_df, results_dir, filename="polarity_table.tex"):
    """
    Exporta tabela de resultados de polaridade em formato LaTeX.
    
    Args:
        results_df: DataFrame com métricas de polaridade
        results_dir: Diretório de saída
        filename: Nome do arquivo de saída
        
    Returns:
        String com conteúdo LaTeX
    """
    latex_content = r"""
\begin{table}[htbp]
\centering
\caption{Expression Polarity Classification Results}
\label{tab:polarity}
\begin{tabular}{lc}
\toprule
\textbf{Polarity} & \textbf{F1-Score} \\
\midrule
"""
    
    polarities = [("Positive", "POS"), ("Negative", "NEG"), ("Neutral", "NEU")]
    
    for name, key in polarities:
        f1_mean = results_df[f"F1_Span_{key}_mean"].iloc[0]
        f1_std = results_df[f"F1_Span_{key}_std"].iloc[0]
        latex_content += f"{name} & {format_metric(f1_mean, f1_std)} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"Tabela de polaridade LaTeX salva em: {filepath}")
    return latex_content


def export_token_level_latex_table(results_df, results_dir, filename="token_level_table.tex"):
    """
    Exporta tabela de métricas token-level em formato LaTeX.
    
    Args:
        results_df: DataFrame com métricas token-level
        results_dir: Diretório de saída
        filename: Nome do arquivo de saída
        
    Returns:
        String com conteúdo LaTeX
    """
    latex_content = r"""
\begin{table}[htbp]
\centering
\caption{Token-level Classification Results}
\label{tab:token_level}
\begin{tabular}{lc}
\toprule
\textbf{Metric} & \textbf{Score} \\
\midrule
"""
    
    token_metrics = [
        ("Accuracy", "acc_token_level"),
        ("Precision", "precision_token_level"),
        ("Recall", "recall_token_level"),
        ("F1-Score", "f1_token_level"),
    ]
    
    for name, key in token_metrics:
        mean = results_df[f"{key}_mean"].iloc[0]
        std = results_df[f"{key}_std"].iloc[0]
        latex_content += f"{name} & {format_metric(mean, std)} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"Tabela token-level LaTeX salva em: {filepath}")
    return latex_content


def export_hyperparameters_table(best_params, results_dir, filename="hyperparameters.tex"):
    """
    Exporta tabela de hiperparâmetros em formato LaTeX.
    
    Args:
        best_params: Dicionário com melhores hiperparâmetros
        results_dir: Diretório de saída
        filename: Nome do arquivo de saída
        
    Returns:
        String com conteúdo LaTeX
    """
    param_names = {
        "learning_rate": "Learning Rate",
        "batch_size": "Batch Size",
        "weight_decay": "Weight Decay",
        "num_train_epochs": "Epochs",
        "warmup_ratio": "Warmup Ratio"
    }
    
    latex_content = r"""
\begin{table}[htbp]
\centering
\caption{Optimal Hyperparameters Found via Optuna}
\label{tab:hyperparameters}
\begin{tabular}{lc}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\
\midrule
"""
    
    for param, value in best_params.items():
        name = param_names.get(param, param)
        if isinstance(value, float):
            if value < 0.001:
                latex_content += f"{name} & {value:.2e} \\\\\n"
            else:
                latex_content += f"{name} & {value:.4f} \\\\\n"
        else:
            latex_content += f"{name} & {value} \\\\\n"

    latex_content += r"""\bottomrule
\end{tabular}
\end{table}
"""

    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(latex_content)
    
    print(f"Tabela de hiperparâmetros LaTeX salva em: {filepath}")
    return latex_content


def export_csv_results(results_df, results_dir, filename="detailed_results.csv"):
    """
    Exporta resultados detalhados em CSV.
    
    Args:
        results_df: DataFrame com todas as métricas
        results_dir: Diretório de saída
        filename: Nome do arquivo de saída
    """
    filepath = os.path.join(results_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"Resultados CSV salvos em: {filepath}")


def export_json_results(study, all_metrics, config, results_dir, filename="experiment_results.json"):
    """
    Exporta todos os resultados em JSON para reprodutibilidade.
    
    Args:
        study: Objeto Study do Optuna
        all_metrics: Dicionário com métricas
        config: Dicionário com configurações do experimento
        results_dir: Diretório de saída
        filename: Nome do arquivo de saída
    """
    results = {
        "experiment_info": {
            "model": config.get("model_name", "unknown"),
            "n_folds": config.get("n_folds", 5),
            "seed": config.get("seed", 42),
            "date": datetime.now().isoformat(),
            "dataset_size": config.get("dataset_size", 0),
            "n_trials": len(study.trials)
        },
        "best_trial": {
            "number": study.best_trial.number,
            "score": study.best_value,
            "params": study.best_params,
            "user_attrs": dict(study.best_trial.user_attrs)
        },
        "all_trials_summary": [
            {
                "trial": t.number,
                "score": t.value,
                "params": t.params,
                "state": str(t.state)
            }
            for t in study.trials if t.value is not None
        ],
        "metrics_summary": all_metrics
    }
    
    filepath = os.path.join(results_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Resultados JSON salvos em: {filepath}")


def generate_summary_report(study, results_df, config, results_dir):
    """
    Gera relatório resumido em texto.
    
    Args:
        study: Objeto Study do Optuna
        results_df: DataFrame com métricas
        config: Dicionário com configurações
        results_dir: Diretório de saída
        
    Returns:
        String com relatório completo
    """
    report = f"""
================================================================================
                    RELATÓRIO DE RESULTADOS - STRUCTURED SENTIMENT ANALYSIS
================================================================================

DATA DO EXPERIMENTO: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
MODELO: {config.get("model_name", "unknown")}
VALIDAÇÃO CRUZADA: {config.get("n_folds", 5)}-fold
SEED: {config.get("seed", 42)}
TAMANHO DO DATASET: {config.get("dataset_size", 0)} amostras
NÚMERO DE TRIALS: {len(study.trials)}

--------------------------------------------------------------------------------
                              MELHORES HIPERPARÂMETROS
--------------------------------------------------------------------------------
"""
    
    for param, value in study.best_params.items():
        if isinstance(value, float):
            report += f"  {param}: {value:.6f}\n"
        else:
            report += f"  {param}: {value}\n"

    report += f"""
--------------------------------------------------------------------------------
                              RESULTADOS PRINCIPAIS
--------------------------------------------------------------------------------

SPAN-LEVEL METRICS (Média ± Desvio Padrão):
"""
    
    metrics_to_report = [
        ("Macro Span F1", "macro_span_f1"),
        ("Holder F1", "holder_f1"),
        ("Aspect F1", "aspect_f1"),
        ("Expression F1", "expression_f1"),
    ]
    
    for name, key in metrics_to_report:
        mean = results_df[f"{key}_mean"].iloc[0]
        std = results_df[f"{key}_std"].iloc[0]
        report += f"  {name}: {mean*100:.2f}% ± {std*100:.2f}%\n"

    report += """
TOKEN-LEVEL METRICS:
"""
    
    token_metrics = [
        ("Accuracy", "acc_token_level"),
        ("F1-Score", "f1_token_level"),
        ("Precision", "precision_token_level"),
        ("Recall", "recall_token_level"),
    ]
    
    for name, key in token_metrics:
        mean = results_df[f"{key}_mean"].iloc[0]
        std = results_df[f"{key}_std"].iloc[0]
        report += f"  {name}: {mean*100:.2f}% ± {std*100:.2f}%\n"

    report += """
POLARITY CLASSIFICATION (F1-Score):
"""
    
    for pol in ["POS", "NEG", "NEU"]:
        mean = results_df[f"F1_Span_{pol}_mean"].iloc[0]
        std = results_df[f"F1_Span_{pol}_std"].iloc[0]
        report += f"  {pol}: {mean*100:.2f}% ± {std*100:.2f}%\n"

    report += """
================================================================================
"""

    filepath = os.path.join(results_dir, "summary_report.txt")
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print(f"Relatório salvo em: {filepath}")
    
    return report


def collect_fold_metrics(trial):
    """
    Coleta métricas de todos os folds de um trial.
    
    Args:
        trial: Objeto Trial do Optuna
        
    Returns:
        Dicionário com mean e std de todas as métricas
    """
    metrics_keys = [
        "macro_span_f1", "holder_f1", "aspect_f1", "expression_f1",
        "holder_precision", "holder_recall",
        "aspect_precision", "aspect_recall",
        "expression_precision", "expression_recall",
        "acc_token_level", "f1_token_level",
        "precision_token_level", "recall_token_level",
        "F1_Span_NEG", "F1_Span_POS", "F1_Span_NEU"
    ]
    
    results = {}
    for key in metrics_keys:
        mean_key = f"mean_{key}"
        std_key = f"std_{key}"
        results[f"{key}_mean"] = trial.user_attrs.get(mean_key, 0.0)
        results[f"{key}_std"] = trial.user_attrs.get(std_key, 0.0)
    
    return results


def export_all_results(study, results_df, config, results_dir):
    """
    Exporta todos os resultados de uma vez.
    
    Args:
        study: Objeto Study do Optuna
        results_df: DataFrame com métricas
        config: Dicionário com configurações
        results_dir: Diretório de saída
    """
    os.makedirs(results_dir, exist_ok=True)
    
    # Exportar em todos os formatos
    export_latex_table(results_df, results_dir, config.get("n_folds", 5))
    export_polarity_latex_table(results_df, results_dir)
    export_token_level_latex_table(results_df, results_dir)
    export_hyperparameters_table(study.best_params, results_dir)
    export_csv_results(results_df, results_dir)
    export_json_results(study, collect_fold_metrics(study.best_trial), config, results_dir)
    generate_summary_report(study, results_df, config, results_dir)
    
    print(f"\nTodos os resultados exportados para: {results_dir}/")