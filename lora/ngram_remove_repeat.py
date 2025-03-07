def remove_repeated_ngrams(text, n=5):
    tokens = text.split()
    new_tokens = []
    seen_ngrams = set()
    
    i = 0
    while i < len(tokens):
        # 현재 위치에서 n-gram 추출 (문장의 끝이면 남은 단어 모두)
        current_ngram = ' '.join(tokens[i:i+n])
        if current_ngram in seen_ngrams and len(tokens[i:i+n]) == n:
            # 반복되는 n-gram이면 i를 n만큼 건너뜁니다.
            i += n
        else:
            seen_ngrams.add(current_ngram)
            new_tokens.append(tokens[i])
            i += 1
    return ' '.join(new_tokens)

# 사용 예시

if __name__ == '__main__':
    postprocessed = []
    # file_path = "/home/elicer/DaconAcc/lora/8bit_finetuned_result_valid_fewshot0_outputs.txt"
    # file_path = "/home/elicer/DaconAcc/DBCMLAB_8bit_fewshot0_output.txt"
    file_path = "/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant_Sim_output.txt"
    with open(file_path) as f:
        outputs = [line.strip() for line in f]
    for output in outputs:
        clean_text = remove_repeated_ngrams(output, n=1)
        postprocessed.append(clean_text)
    
    # save_path = "/home/elicer/DaconAcc/lora/8bit_finetuned_result_valid_fewshot0_outputs_post.txt"
    save_path = "/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant_Sim_output_ngram1.txt"
    with open(save_path, 'w') as f:
        for line in postprocessed:
            f.write(line + '\n')
    print("중복제거->제출")
