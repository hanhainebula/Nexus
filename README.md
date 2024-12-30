# UniRetrieval
Toolkit for Universal Retrieval, such as text retrieval, item recommendation, image retrieval, etc.

## 2024/12/29

1. Text Retrieval
    - Training 部分已经完成
    - Inference 加速部分 (onnx, tensorrt) 都已经实现
    - Evaluation 部分已经完成 (Optional: 加到 Trainer 里面供训练过程中评测使用，决定 early stop)
2. Recommendation
    - Training 部分已经完成
    - Inference 部分 embedder 的 onnx 和 tensorrt 加速都已经完成；reranker 的 onnx 已经完成，tensorrt 遇到一些问题
    - Evaluation 部分的代码已经完成还在 debug，然后整合到 Trainer 里面实现 early stop

TODO：
- Recommendation 部分：
    - reranker 的 tensorrt 加速：bug 解决
    - Evaluation 部分的代码 debug，然后整合到 Trainer 里面实现 early stop
- Text Retrieval 部分：
    - Optional: 把 Evaluation 部分的代码整合到 Trainer 里面实现 early stop
- Examples 编写，为后续的测试做准备：
    - Text Retrieval 部分的 examples 编写，每个功能都要有一个 example
    - Recommendation 部分的 examples 编写，每个功能都要有一个 example
- 合并代码 & 整理代码

## 2024/12/23

1. Training
    - Text Retrieval
        - modeling.py:
            - forward 函数: 改成一个通用的形式，放到 abc/training/embedder/AbsModeling.py 里面 (参数由原来 data_collator 返回的字典改为一个元组 -> 需要对应修改 data_collator 的 `__call__` 函数) - DONE
            - loss functions: 放到 modules/loss.py 里面实现 (包括 InfoNCE 损失和两个蒸馏损失，一个 loss_function，一个 distill_loss_function) - DONE
            - compute_score & compute_loss 函数的实现：把原来 forward 函数里面的代码实现到这两个函数里面 - DONE
            - 抽象 init_modules 到 abc/training/embedder/AbsModeling.py 里面，包括 loss_function 和 score_function - DONE
            - `__init__` 函数中的来自 model_args 的所有的参数都放到一个变量中，变量类型是 AbsModelArguments - DONE
            - forward 函数的返回值 (compute_loss 函数的返回值) 要是 EmbedderOutput 类型 (对应的 compute_loss 函数要同时返回 scores 和 loss)  DONE
        - datasets.py
            - 把两个 data_collator 的返回值由字典改成元组，即传给 model.forward 函数的 batch 参数 - DONE
            - abc/training/embedder/AbsDataset.py 中的 AbsEmbedderCollator 不再需要继承自 DataCollatorWithPadding，text retrieval 的子类中需要继承 - DONE
            - Callback 类挪到同级目录下的新建 callback.py 中，抽象类中的 AbsCallback 删掉 - DONE
    - Recommendation
        - arguments.py
            - TrainingArguments: 参考 HF 的 TrainingArguments，删掉/重命名已有的参数，保留推荐独有的参数 - Done
            - Statistics: 改成继承 AbsArguments - DONE
            - ModelArguments: 新增 data_config 参数，是 DataAttr4Model 类型 - DONE
            - RetrieverArguments -> ModelArguments - Done
            - DataArguments: 待检查，需要根据 RecStudio4Industry 的数据格式来修改，必要时可以添加 add_argument(self, name, value) 方法
            - 删去 178 行往后的冗余代码，需要 check，需要调用时从最外层的 modules 里面调用 - Done
        - datasets.py
            - 删除冗余的 DataAttr4Model 和 Statistics (arguments.py 中已有) - Done
            - 删除一些冗余的 read_* 代码，比如 read_json 和 read_yaml - Done
            - EarlyStopCallback 等 callback 类也挪到同级目录下的 callback.py 中，考虑删除掉一些不必要的 Callback，不再返回设定的 CallbackOutput，而是在 trainer 中 `__init__` 时设置一个 stop_training 的 flag，每个训练 batch 开始时检查 flag
        - modeling.py:
            - init_modules 方法根据 abc 中的代码进行修改，只保留推荐自己的 modules (保留 query_enoder, item_encoder, negative_sampler) - Done
            - compute_score 函数的 Item_loader 拉到 DailyDataset 里面, 实现一个 callback 函数 on_step_begin 传进去一个trainer_state，包含新的参数model以备动态负采样，通过Trainer类的args.model为dataset实现refresh_negative。
            - compute_score 的输入 batch 的内容多一些依赖于 ItemLoader 的参数 
            - BaseRetriever 里面的 get_item_feat 删掉，不依赖于self
            - forward 函数需要进行修改，删掉 cal_loss 这个参数，拆开对应的功能；RetrieverModelOutput 需要对应修改下来传入 scores 和 loss
            - eval_step 待定，后续可以拉到 evaluation 里面； predict 函数待定
        - trainer.py
            - 和 train 相关的函数，需要参考 HF 的 Trainer，把一些不必要的代码删掉 -
            - 和负采样相关的代码，需要 check 下是否需要再进行实现 (是否需要添加 callback) - 
            - 和 evaluation 相关的代码，可以先扔到 evaluation 里面
            - 删掉 modules/metrics.py 里面已经有的代码 (536行往后) - Done
            - get_train_loader 这个函数需要改成 get_train_dataloader 和 HF Trainer 名称保持一致，同时传入 data_collator, 在abc里面实现一个返回输入值的collator抽象类
            - train 的接口传入的 train_dataset 是 DailyDataset 类型的 - Done
            - 在 trainer 中 `__init__` 时设置一个 stop_training 的 flag，每个训练 batch 开始时检查 flag
            - self.args.model = self.model
        - runner.py
            - 实现 load_dataset 方法，返回 DailyDataset 类型 
            - 实现 load_data_collator 方法，返回实现的 DataCollator 类型


## 2024/12/12

Training 部分：
- Text Retrieval 部分：已经完成
- Recommendation 部分：已经完成
- TODO: 
    - 两部分的代码的整理，去掉一些冗余的代码，使得整体的代码更加一体化

Inference 部分：
- Text Retrieval 部分：已经完成了基本的代码
- TODO:
    - 在 Text Retrieval 部分集成进去 RecStudio4Industry 中用到的加速方法
    - Recommendation 部分的代码整理

Evaluation 部分：
- Text Retrieval 部分: 评测框架已经完成
- TODO:
    - Recommendation 部分: 评测框架的搭建
