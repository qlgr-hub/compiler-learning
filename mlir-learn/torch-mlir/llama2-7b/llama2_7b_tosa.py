# from transformers import AutoTokenizer, LlamaForCausalLM
# import torch
# import torch_mlir

# model = LlamaForCausalLM.from_pretrained("/workspace/mlir-learn/torch-mlir/llama2-7b/Llama-2-7b-hf")
# tokenizer = AutoTokenizer.from_pretrained("./Llama-2-7b-hf")

# prompt = "What is the capital of France?"
# inputs = tokenizer(prompt, return_tensors="pt")

# print(f"Input shape: {inputs['input_ids'].shape}")

# model.eval()
# randomLongTensor = torch.randint(low=0, high=100, size=(1, 8), dtype=torch.long)


# m = torch.export.export(model, (randomLongTensor,))
# print(m)
