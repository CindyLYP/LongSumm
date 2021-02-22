import os


# os.environ["CUDA_VISIBLE_DEVICES"]
print(os.environ.keys())
if "CUDA_VISIBLE_DEVICES" in os.environ.keys():
    print("yes")
else:
    print("no")