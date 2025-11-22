import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    
    def generate_stream(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
    ):
        """流式生成 - 逐个 token 返回"""
        # 添加单个请求
        self.add_request(prompt, sampling_params)
        
        # 获取序列ID（最新添加的）
        seq = self.scheduler.waiting[-1]
        target_seq_id = seq.seq_id
        
        # 记录已生成的token数
        generated_tokens = 0
        
        while not self.is_finished():
            # 执行一步推理
            outputs, _ = self.step()
            
            # 检查目标序列是否有新输出
            for seq_id, token_ids in outputs:
                if seq_id == target_seq_id:
                    # 计算新生成的token
                    new_tokens = token_ids[generated_tokens:]
                    generated_tokens = len(token_ids)
                    
                    # 解码新token
                    new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                    
                    # 返回新生成的文本和是否完成
                    yield {
                        "text": new_text,
                        "token_ids": new_tokens,
                        "finished": True  # 这个请求已完成
                    }
                    return  # 序列完成，退出
            
            # 如果这一步没有完成，检查是否生成了新token
            # 在decode阶段，每个序列都会生成一个token
            for seq in self.scheduler.running:
                if seq.seq_id == target_seq_id:
                    current_tokens = seq.completion_token_ids
                    if len(current_tokens) > generated_tokens:
                        new_tokens = current_tokens[generated_tokens:]
                        generated_tokens = len(current_tokens)
                        
                        # 解码新token
                        new_text = self.tokenizer.decode(new_tokens, skip_special_tokens=False)
                        
                        yield {
                            "text": new_text,
                            "token_ids": new_tokens,
                            "finished": False
                        }
                    break