from Levenshtein import distance as levenshtein_distance
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

def exact_match(prediction: str, ground_truth: str) -> bool:
    return prediction.strip() == ground_truth.strip()

def char_level_accuracy(prediction: str, ground_truth: str) -> float:
    matches = sum(p == g for p, g in zip(prediction, ground_truth))
    return matches / max(len(ground_truth), len(prediction))

def normalized_levenshtein(prediction: str, ground_truth: str) -> float:
    max_len = max(len(prediction), len(ground_truth))
    return 1 - (levenshtein_distance(prediction, ground_truth) / max_len)

def bleu_score(prediction: str, ground_truth: str) -> float:
    return sentence_bleu([ground_truth.split()], prediction.split())

def rouge_l(prediction: str, ground_truth: str) -> float:
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(prediction, ground_truth)['rougeL'].fmeasure

def evaluate_metrics(file1_path: str, file2_path: str) -> dict:
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        prediction = f1.read()
        ground_truth = f2.read()
    
    results = {
        'Exact Match': exact_match(prediction, ground_truth),
        'Character-Level Accuracy': char_level_accuracy(prediction, ground_truth),
        'Normalized Levenshtein Similarity': normalized_levenshtein(prediction, ground_truth),
        'BLEU Score': bleu_score(prediction, ground_truth),
        'ROUGE-L': rouge_l(prediction, ground_truth)
    }
    
    return results

truth = './book_input.txt'
pred = './gpt_output.txt'

# Running the evaluation
results = evaluate_metrics(truth, pred)

print(results)
