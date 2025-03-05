import sys
sys.path.append('/home/elicer/DaconAcc/')

from lora.sim_eval import Evaluater
import pandas as pd

if __name__ == '__main__':
    evaluater = Evaluater()
    
    output_path = "/home/elicer/DaconAcc/DBCMLAB_8bit_finetuned_result_valid_fewshot0_output_ngram1_re.txt"
    with open(output_path) as f:
        generated_texts = [line.strip() for line in f]
    
    val_path = "/home/elicer/DaconAcc/dataset/valid_prompt.csv"
    valid_answers = pd.read_csv(val_path)["answer"].tolist()

    score, _ = evaluater.evaluate(generated_texts, valid_answers)
    print(score)
    