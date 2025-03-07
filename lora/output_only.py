import json


output = []
with open("/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant_Sim.txt") as f:
    for line in f:
        line = json.loads(line.strip())
        output.append(line["output"])
        

end_token = "<|im_end|>"
end_token2 = "<|"

with open("/home/elicer/DaconAcc/finetuned_juungwon_WO_Quant_Sim_output.txt", 'w') as f:
    for line in output:
        line = line.strip()
        if end_token in line:
            line = line.split(end_token)[0]
        if end_token2 in line:
            line = line.split(end_token)[0]
        f.write(line.strip() + "\n")
print("프롬프트 제거-> 중복제거")