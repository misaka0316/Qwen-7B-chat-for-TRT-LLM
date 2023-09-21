import numpy as np  


a = np.load('hidden_states.npy')
b = np.load('0layers.0.attention.hidden_states.npy')  
# print(a.shape) 
# print(b)
# print((a-b).sum())
# print((a-b).sum()/a.sum())
print(np.abs(a-b).max())
print(np.abs(a-b).min())
print(np.abs(a-b).sum()/a.size)

# a = np.load('mixed_x_layer.npy')
# b = np.load('0layers.0.attention.qkv.npy')  
# # print(a) 
# # print(b)
# # a = a[:, :, -3:]
# # b = b[:, :, -3:]
# print(a[:, :, (np.abs(a-b).argmax(axis=-1))])
# print((np.abs(a-b).argmax(axis=-1)))
# print(np.abs(a-b).max())
# print(np.abs(a-b).min())
# print(np.abs(a-b).sum()/a.sum())

X = np.load('0layers.0.attention.hidden_states.npy')
W = np.fromfile("./c-model/Qwen/1-gpu/model.layers.0.attention.query_key_value.weight.0.bin", dtype=np.float16).reshape(12288,4096)
B = np.fromfile("./c-model/Qwen/1-gpu/model.layers.0.attention.query_key_value.bias.0.bin", dtype=np.float16).reshape(1,12288)
Y = X @ W.T + B

Y_hat = np.load('0layers.0.attention.qkv.npy')