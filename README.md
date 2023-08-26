# Qwen-7B-chat-for-TRT-LLM
## 本项目是 NVIDIA TensorRT Hackathon 2023 的参赛题目
比赛链接为： [NVIDIA TensorRT Hackathon 2023](https://github.com/NVIDIA/trt-samples-for-hackathon-cn/tree/master/Hackathon2023) 

选题为以下的选项中的（2+4）：  
1.用TensorRT优化模型  
2.使用TensorRT-LLM实现新模型  
3.用TensorRT-LLM优化examples目录下的某个现有模型  
4.为TensorRT-LLM添加新feature，或者在模型上启用了现有feature  

原始模型为：[通义千问-7B-chat](https://huggingface.co/Qwen/Qwen-7B-Chat)  

通义千问-7B（Qwen-7B） 是阿里云研发的通义千问大模型系列的70亿参数规模的模型。Qwen-7B是基于Transformer的大语言模型, 在超大规模的预训练数据上进行训练得到。预训练数据类型多样，覆盖广泛，包括大量网络文本、专业书籍、代码等。同时，在Qwen-7B的基础上，我们使用对齐机制打造了基于大语言模型的AI助手Qwen-7B-Chat。Qwen-7B系列模型的特点包括：

1. **大规模高质量预训练数据**：我们使用了超过2.2万亿token的自建大规模预训练数据集进行语言模型的预训练。数据集包括文本和代码等多种数据类型，覆盖通用领域和专业领域。
2. **优秀的模型性能**：相比同规模的开源模型，Qwen-7B在多个评测数据集上具有显著优势，甚至超出12-13B等更大规模的模型。评测评估的能力范围包括自然语言理解与生成、数学运算解题、代码生成等。
3. **更好地支持多语言**：基于更大词表的分词器在分词上更高效，同时它对其他语言表现更加友好。用户可以在Qwen-7B的基础上更方便地训练特定语言的7B语言模型。
4. **8K的上下文长度**：Qwen-7B及Qwen-7B-Chat均能支持8K的上下文长度, 允许用户输入更长的prompt。
5. **支持插件调用**：Qwen-7B-Chat针对插件调用相关的对齐数据做了特定优化，当前模型能有效调用插件以及升级为Agent。
  
（注：* 2023年8月21日 阿里云了发布Qwen-7B-Chat的Int4量化模型，Qwen-7B-Chat-Int4。本项目先对fp16精度进行优化）
