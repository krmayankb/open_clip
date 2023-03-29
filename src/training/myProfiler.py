import torch
from torch.profiler import profile, record_function, ProfilerActivity

device = torch.device("cuda")
import open_clip
import csv 
import time

import argparse 
parser = argparse.ArgumentParser(description='My Profiler')

# benchmark specific args
parser.add_argument('--model', metavar='NAME', default='',
                    help='model to profile')

args = parser.parse_args()

# Define your model and data
model,_,preprocess = open_clip.create_model_and_transforms(args.model)
model = model.to(device)
data = torch.randn(1024, 3, 224, 224).to(device)

tokenizer = open_clip.get_tokenizer(args.model)
text = tokenizer(["a diagram", "a dog", "a cat"]).to(device)

results = []
# Set up the profiler
import csv

# Run inference with different batch sizes and collect profiling data
results = []
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as prof:
    for batch_size in [256,512]:
        input_data = data[:batch_size]
        with record_function(f"batch_size_{batch_size}"):
            torch.cuda.synchronize()
            start = time.time()
            output = model.encode_image(input_data)
            text_features = model.encode_text(text)

            torch.cuda.synchronize()
            end = time.time()
            
        print(f"{batch_size}:{end -start}")
        del output, text_features
        
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
