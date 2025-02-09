import onnx
import json
import onnxruntime as ort
from tqdm import tqdm, trange
from Nexus import TextEmbedder, AbsInferenceArguments, BaseEmbedderInferenceEngine



model = TextEmbedder(model_name_or_path='/root/models/bge-base-zh-v1.5', use_fp16=True, devices=['cuda:0'])

onnx_model_path ='/root/models/bge-base-zh-v1.5/onnx/model_fp16.onnx'
providers = ['CUDAExecutionProvider', "CPUExecutionProvider"]
session=ort.InferenceSession(onnx_model_path, providers=providers)

with open('/root/datasets/test_inference.json','r', encoding='utf-8') as f:
    sentences = json.load(f)

def _inference_onnx(inputs, normalize = True, batch_size = None, *args, **kwargs):
    if isinstance(inputs, str):
        inputs=[inputs]
        
    tokenizer = model.tokenizer
    all_outputs=[]
    for i in trange(0, len(inputs), batch_size, desc='Batch Inference'):
        batch_inputs= inputs[i:i+batch_size]
        encoded_inputs = tokenizer(batch_inputs, return_tensors="np", padding=True,  truncation=True, max_length=512)
        # input_ids = encoded_inputs['input_ids']
        input_feed={
            'input_ids':encoded_inputs['input_ids'], #(bs, max_length)
            'attention_mask':encoded_inputs['attention_mask'],
            'token_type_ids':encoded_inputs['token_type_ids']
        }

        outputs = session.run(None, input_feed)
        embeddings = outputs[0] # (1, 9, 768)
        cls_emb=embeddings[:, 0, :]
        cls_emb=cls_emb.squeeze()
        all_outputs.extend(cls_emb)
    
    if normalize == True:
        all_outputs = all_outputs / np.linalg.norm(all_outputs, axis=-1, keepdims=True)
        return all_outputs
    
    return cls_emb

def main():
    # _inference_onnx(sentences, batch_size=16)
    tokenizer = model.tokenizer
    a='你好你好'
    import pdb
    pdb.set_trace()
    
main()