import re
import codecs
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_f1_scores(file_path):
    patterns = {
        'train': r'train epoch \d+ .*?f1 score: (\d*\.?\d*)',
        'val': r'val epoch \d+ .*?f1 score: (\d*\.?\d*)',
        # 'test': r'test epoch \d+ .*?f1 score: (\d*\.?\d*)'
    }
    
    scores = defaultdict(list)
    
    with codecs.open(file_path, 'r', encoding="latin-1") as file:
        content = file.read()
        
        for set_type, pattern in patterns.items():
            matches = re.finditer(pattern, content, re.IGNORECASE)
            set_scores = [float(match.group(1)) for match in matches]
            scores[set_type] = set_scores
            
    return dict(scores)

def make_plots(data):
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(data['train']) + 1)
    plt.plot(epochs, data['train'], 'b-', label='Train', linewidth=2)
    plt.plot(epochs, data['val'], 'r-', label='Validation', linewidth=2)
    # plt.plot(epochs, data['test'], 'g-', label='Test', linewidth=2)

    plt.title('F1 Scores', fontsize=14, pad=15)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('F1 Score', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)

    plt.grid(True, which='minor', linestyle=':', alpha=0.4)
    plt.minorticks_on()

    plt.tight_layout()

    plt.savefig("data/plots/f1_scores.png")

def main(filename):
    scores = extract_f1_scores(filename)
    make_plots(scores)

if __name__ == "__main__":
    main('data/raw_data/buf.txt')
