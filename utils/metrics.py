"""
Funções auxiliares para processamento de dados e cálculo de métricas.
"""

import copy
import numpy as np
from seqeval.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)


def tokenize_and_align_labels(example, tokenizer):
    """
    Tokeniza um exemplo e alinha os labels com os tokens.
    
    Args:
        example: Dicionário com 'tokens' e 'tags'
        tokenizer: Tokenizer do HuggingFace
        
    Returns:
        Dicionário tokenizado com labels alinhados
    """
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=512
    )

    word_ids = tokenized.word_ids()
    aligned_labels = []
    prev_word = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != prev_word:
            aligned_labels.append(example["tags"][word_id])
        else:
            aligned_labels.append(-100)
        prev_word = word_id

    tokenized["labels"] = aligned_labels
    return tokenized


def get_majority_label(tags):
    """
    Extrai o label majoritário para estratificação.
    
    Args:
        tags: Lista de tags numéricas
        
    Returns:
        Label mais frequente (excluindo 'O')
    """
    label_counts = {}
    for tag in tags:
        if tag != 0:  # Ignora 'O'
            label_counts[tag] = label_counts.get(tag, 0) + 1
    if label_counts:
        return max(label_counts, key=label_counts.get)
    return 0


def get_tuples(tags, target):
    """
    Extrai spans (tuplas de início, fim) para um target específico.
    
    Args:
        tags: Lista de tags BIO
        target: Tipo de entidade ('HOLDER', 'ASPECT', 'EXP')
        
    Returns:
        Conjunto de tuplas (início, fim, target)
    """
    spans = []
    start = None

    for i, tag in enumerate(tags):
        if tag.startswith("B-" + target):
            if start is not None:
                spans.append((start, i - 1, target))
            start = i
            if i + 1 < len(tags) and tags[i + 1].startswith("B-") and target not in tags[i + 1]:
                spans.append((start, start, target))
                start = None
                continue

        elif tag.startswith("I-" + target):
            continue

        else:
            if start is not None:
                spans.append((start, i - 1, target))
                start = None

    if start is not None:
        spans.append((start, len(tags) - 1, target))
    return set(spans)


def metrics(pred_set, gold_set, target, verbose=False):
    """
    Calcula métricas de span-level (precision, recall, F1).
    
    Args:
        pred_set: Lista de conjuntos de spans preditos
        gold_set: Lista de conjuntos de spans gold
        target: Nome do target para logging
        verbose: Se True, imprime resultados
        
    Returns:
        Dicionário com f1, precision, recall, tp, fp, fn
    """
    tp, fp, fn = 0, 0, 0

    for pred_spans, gold_spans in zip(pred_set, gold_set):
        pred_spans_xy = {(p[0], p[1]) for p in pred_spans}
        gold_spans_xy = {(g[0], g[1]) for g in gold_spans}

        tp += len(pred_spans_xy & gold_spans_xy)
        fp += len(pred_spans_xy - gold_spans_xy)
        fn += len(gold_spans_xy - pred_spans_xy)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = (2 * precision * recall) / (precision + recall + 1e-10)

    if verbose:
        print(f"Target: {target}")
        print(f"TP: {tp}, FP: {fp}, FN: {fn}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    return {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def seq_metrics(pred, gold):
    """
    Calcula métricas de token-level usando seqeval.
    
    Args:
        pred: Lista de listas de predições
        gold: Lista de listas de labels gold
        
    Returns:
        Dicionário com acc, f1, precision, recall
    """
    return {
        "acc": accuracy_score(gold, pred),
        "f1": f1_score(pred, gold),
        "precision": precision_score(gold, pred),
        "recall": recall_score(gold, pred)
    }


def group_by_label(pred, gold, label_storage, labels_metric=None):
    """
    Agrupa spans por tipo de label.
    
    Args:
        pred: Lista de listas de predições
        gold: Lista de listas de labels gold
        label_storage: Dicionário para armazenar os spans
        labels_metric: Lista de labels a processar (default: HOLDER, ASPECT, EXP)
    """
    if labels_metric is None:
        labels_metric = ["HOLDER", "ASPECT", "EXP"]
        
    for pred_, gold_ in zip(pred, gold):
        for target in labels_metric:
            pred_list = get_tuples(pred_, target)
            gold_list = get_tuples(gold_, target)

            label_storage[target]["PRED"].append(set(pred_list))
            label_storage[target]["GOLD"].append(set(gold_list))


def split_expression(data_list):
    """
    Converte labels de EXPRESSION-POS/NEG/NEU para POS/NEG/NEU.
    
    Args:
        data_list: Lista de listas de labels
        
    Returns:
        Cópia com labels de expressão simplificados
    """
    result = copy.deepcopy(data_list)
    for data_pos, labels in enumerate(result):
        for label_pos, label in enumerate(labels):
            if "EXP" in label:
                parts = label.split("-")
                if len(parts) >= 3:
                    result[data_pos][label_pos] = label[:2] + parts[2]
    return result


def class_report(pred, gold):
    """
    Gera relatório de classificação por polaridade.
    
    Args:
        pred: Lista de listas de predições
        gold: Lista de listas de labels gold
        
    Returns:
        Dicionário com F1-score para NEG, POS, NEU
    """
    pol_pred = split_expression(pred)
    pol_gold = split_expression(gold)

    report = classification_report(pol_gold, pol_pred, zero_division=0, output_dict=True)
    return {
        "NEG": report.get("NEG", {}).get("f1-score", 0.0),
        "POS": report.get("POS", {}).get("f1-score", 0.0),
        "NEU": report.get("NEU", {}).get("f1-score", 0.0)
    }


def compute_metrics(p, id2label, labels_metric=None, verbose=False):
    """
    Função principal de cálculo de métricas para o Trainer.
    
    Args:
        p: Tupla (predictions, labels) do Trainer
        id2label: Dicionário de id para label
        labels_metric: Lista de labels a avaliar
        verbose: Se True, imprime detalhes
        
    Returns:
        Dicionário com todas as métricas
    """
    if labels_metric is None:
        labels_metric = ["HOLDER", "ASPECT", "EXP"]
        
    predictions, labels = p
    pred_ids = np.argmax(predictions, axis=2)

    list_of_compared_labels = {
        "HOLDER": {"PRED": [], "GOLD": []},
        "ASPECT": {"PRED": [], "GOLD": []},
        "EXP": {"PRED": [], "GOLD": []},
    }

    true_preds = [
        [id2label[p] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(pred_ids, labels)
    ]

    true_labels = [
        [id2label[l] for (p, l) in zip(pred, lab) if l != -100]
        for pred, lab in zip(pred_ids, labels)
    ]

    group_by_label(true_preds, true_labels, list_of_compared_labels, labels_metric)

    f1_data = {}
    detailed_metrics = {}
    for label in labels_metric:
        pred_spans = list_of_compared_labels[label]["PRED"]
        true_spans = list_of_compared_labels[label]["GOLD"]
        result = metrics(pred_spans, true_spans, label, verbose=verbose)
        f1_data[label] = result["f1"]
        detailed_metrics[label] = result

    macro_span_f1 = sum(f1_data.values()) / len(f1_data)
    seq = seq_metrics(true_preds, true_labels)
    report = class_report(true_preds, true_labels)

    return {
        "macro_span_f1": macro_span_f1,
        "holder_f1": f1_data["HOLDER"],
        "aspect_f1": f1_data["ASPECT"],
        "expression_f1": f1_data["EXP"],
        "holder_precision": detailed_metrics["HOLDER"]["precision"],
        "holder_recall": detailed_metrics["HOLDER"]["recall"],
        "aspect_precision": detailed_metrics["ASPECT"]["precision"],
        "aspect_recall": detailed_metrics["ASPECT"]["recall"],
        "expression_precision": detailed_metrics["EXP"]["precision"],
        "expression_recall": detailed_metrics["EXP"]["recall"],
        "acc_token_level": seq["acc"],
        "f1_token_level": seq["f1"],
        "precision_token_level": seq["precision"],
        "recall_token_level": seq["recall"],
        "F1_Span_NEG": report["NEG"],
        "F1_Span_POS": report["POS"],
        "F1_Span_NEU": report["NEU"]
    }