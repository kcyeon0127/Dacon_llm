import gc
import psutil
import threading
import torch
import time
from typing import List
import pandas as pd
from transformers import LlamaTokenizer, pipeline
from tqdm import tqdm
from contextlib import nullcontext
from datetime import datetime
import json
import torch.distributed as dist
from transformers.pipelines.pt_utils import KeyDataset
from lora.sim_eval import Evaluater
from utils import get_valid_input_dataset



def save_to_json(output_filename, train_step_loss, train_epoch_loss, train_step_ppl, train_epoch_ppl, val_step_loss, val_epoch_loss, val_step_ppl, val_epoch_ppl):
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)

def byte2gb(x):
    return int(x / 2**30)

class MemoryTrace:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = byte2gb(torch.cuda.memory_allocated())
        self.process = psutil.Process()
        self.cpu_begin = byte2gb(self.cpu_mem_used())
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = byte2gb(torch.cuda.memory_allocated())
        self.peak = byte2gb(torch.cuda.max_memory_allocated())
        cuda_info = torch.cuda.memory_stats()
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.cuda_malloc_retires = cuda_info.get("num_alloc_retries", 0)
        self.peak_active_gb = byte2gb(cuda_info["active_bytes.all.peak"])
        self.m_cuda_ooms = cuda_info.get("num_ooms", 0)
        self.used = byte2gb(self.end - self.begin)
        self.peaked = byte2gb(self.peak - self.begin)
        self.max_reserved = byte2gb(torch.cuda.max_memory_reserved())

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = byte2gb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = byte2gb(self.cpu_peak - self.cpu_begin)
        # print(f"delta used/peak {self.used:4d}/{self.peaked:4d}")


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config):
    metrics_filename = f"{train_config.output_dir}/metrics_data-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    train_prep = []
    train_loss = []
    val_prep = []
    val_loss =[]
    train_step_perplexity = []
    train_step_loss = []
    val_step_loss = []
    val_step_perplexity = []
    
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_similarity = 0.0

    def eval_n_save(best_val_loss, best_val_similarity):
        eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity = evaluation(model, train_config, eval_dataloader, tokenizer)
        if eval_epoch_loss < best_val_loss:
            print("\n\n\n")
            model.save_pretrained(train_config.output_dir)
            print(f"PEFT modules are saved in {train_config.output_dir} directory")
            best_val_loss = eval_epoch_loss
            print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            print("\n\n\n")
            
        if train_config.save_best_similarity:
            similarity = generate_n_evaluate_similarity(model, tokenizer, train_config)
            print("Similarity:", similarity)
            if best_val_similarity < similarity:
                print("\n\n\n")
                model.save_pretrained(train_config.output_dir + "_similarity")
                print(f"PEFT modules are saved in {train_config.output_dir}_similarity directory")
                best_val_similarity = similarity
                print(f"best eval similarity on epoch {epoch+1} is {best_val_similarity}")
                print("\n\n\n")
        
        return best_val_loss, best_val_similarity, eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, similarity

    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:
            model.train()
            total_loss = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    batch[key] = batch[key].to('cuda:0')
                with nullcontext():
                    loss = model(**batch).loss
                loss = loss / gradient_accumulation_steps
                train_step_loss.append(loss.detach().float().item())
                train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                total_loss += loss.detach().float()
                loss.backward()
                if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    
                    if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.gradient_clipping_threshold)
                    optimizer.step()
                    isEvaluated = False
                    optimizer.zero_grad()
                    pbar.update(1)
                pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})")
                save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
                
                if train_config.eval_strategy == "step" and (step + 1) % train_config.eval_step == 0 and not isEvaluated:
                    best_val_loss, best_val_similarity, eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, similarity = eval_n_save(best_val_loss, best_val_similarity)
                    isEvaluated = True
                    
                    
                
            pbar.close()
                

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)

        train_epoch_loss = total_loss / len(train_dataloader)
        train_perplexity = torch.exp(train_epoch_loss)
        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))
        lr_scheduler.step()
        
        if not isEvaluated:
            best_val_loss, best_val_similarity, eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, similarity = eval_n_save(best_val_loss, best_val_similarity)
        
        val_step_loss.extend(temp_val_loss)
        val_step_perplexity.extend(temp_step_perplexity)

        checkpoint_start_time = time.perf_counter()
        
            
        
            
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)
            
        val_loss.append(float(best_val_loss))
        val_prep.append(float(eval_ppl))
        print(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        save_to_json(metrics_filename, train_step_loss, train_loss, train_step_perplexity, train_prep, val_step_loss, val_loss, val_step_perplexity, val_prep)
    
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_eval_prep = sum(val_prep)/len(val_prep)
    avg_eval_loss = sum(val_loss)/len(val_loss)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results['avg_eval_prep'] = avg_eval_prep
    results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    results["metrics_filename"] = metrics_filename


    return results

def extract_output(outputs):
    generated_texts = []
    for output in outputs:
        if "<|start_header_id|>assistant<|end_header_id|>" in output:
            output = output.split("<|start_header_id|>assistant<|end_header_id|>")[1].strip()
        if "<|eot_id|>" in output:
            output = output.split("<|eot_id|>")[0].strip()
        generated_texts.append(output)
    return generated_texts

def generate_n_evaluate_similarity(model, tokenizer, train_config):
    valid_input_dataset = get_valid_input_dataset(tokenizer)
    stop_word = "<|eot_id|>"
    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        return_full_text=False, 
        batch_size=train_config.val_batch_size,
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        max_new_tokens=128, 
        do_sample=False, 
        stop_sequence=stop_word
    )
    generated_texts = []
    for out in tqdm(pipe(KeyDataset(valid_input_dataset, "text")), total=1000):
        for line in out:
            clean_text = remove_repeated_ngrams(line['generated_text'].strip())
            generated_texts.append(clean_text.strip())
    
    return get_similarity(generated_texts)

def remove_repeated_ngrams(text, n=1):
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

def get_similarity(outputs: List[str]):

    generated_texts = extract_output(outputs)
    print("\n\n\n")
    print("Eval Generated Text")
    print(generated_texts[0])
    print("\n\n\n")
    evaluater = Evaluater()
    val_path = "/home/elicer/DaconAcc/dataset/valid_prompt.csv"
    valid_answers = pd.read_csv(val_path)["answer"].tolist()

    score, _ = evaluater.evaluate(generated_texts, valid_answers)
    return score

def evaluation(model,train_config, eval_dataloader, tokenizer):
    model.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    generated_texts = []
    with MemoryTrace() as memtrace:
        for step, batch in enumerate(tqdm(eval_dataloader,colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
            for key in batch.keys():
                batch[key] = batch[key].to('cuda:0')
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                outputs = model(**batch)
                loss = outputs.loss
                val_step_loss.append(loss.detach().float().item())
                val_step_perplexity.append(float(torch.exp(loss.detach().float())))

                eval_loss += loss.detach().float()
                
                # token_ids = torch.argmax(outputs.logits, dim=-1)
                # decoded_texts = tokenizer.batch_decode(token_ids, skip_special_tokens=False)
                # generated_texts += decoded_texts

            # Decode predictions and add to evaluation predictions list
            # preds = torch.argmax(outputs.logits, -1)
            # eval_preds.extend(
            #     tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            # )
    # If there's more than one CUDA device, reduce evaluation loss across all devices

    # similarity = eval_similarity(generated_texts)
    
    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_ppl = torch.exp(eval_epoch_loss)

    print(f" {eval_ppl=} {eval_epoch_loss=}")        
    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity