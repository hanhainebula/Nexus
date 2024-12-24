# UniRetrieval
Toolkit for Universal Retrieval, such as text retrieval, item recommendation, image retrieval, etc.


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
            - TrainingArguments: 参考 HF 的 TrainingArguments，删掉/重命名已有的参数，保留推荐独有的参数
            - Statistics: 改成继承 AbsArguments - DONE
            - ModelArguments: 新增 data_config 参数，是 DataAttr4Model 类型 - DONE
            - RetrieverArguments -> ModelArguments
            - DataArguments: 待检查，需要根据 RecStudio4Industry 的数据格式来修改，必要时可以添加 add_argument(self, name, value) 方法
            - 删去 178 行往后的冗余代码，需要 check，需要调用时从最外层的 modules 里面调用
        - datasets.py
            - 把 DailyDataset 中的 `__iter__` 方法牵涉到的操作挪到 DataCollator 中
            - 删除冗余的 DataAttr4Model 和 Statistics (arguments.py 中已有)
            - 删除一些冗余的 read_* 代码，比如 read_json 和 read_yaml
            - EarlyStopCallback 等 callback 类也挪到同级目录下的 callback.py 中，考虑删除掉一些不必要的 Callback，不再返回设定的 CallbackOutput，而是在 trainer 中 `__init__` 时设置一个 stop_training 的 flag，每个训练 batch 开始时检查 flag
        - modeling.py:
            - init_modules 方法根据 abc 中的代码进行修改，只保留推荐自己的 modules (保留 query_enoder, item_encoder, negative_sampler)
            - forward 函数需要进行修改，删掉 cal_loss 这个参数，拆开对应的功能；RetrieverModelOutput 需要对应修改下来传入 scores 和 loss
            - eval_step 待定，后续可以拉到 evaluation 里面； predict 函数待定
        - trainer.py
            - 和 train 相关的函数，需要参考 HF 的 Trainer，把一些不必要的代码删掉
            - 和负采样相关的代码，需要 check 下是否需要再进行实现 (是否需要添加 callback)
            - 和 evaluation 相关的代码，可以先扔到 evaluation 里面
            - 删掉 modules/metrics.py 里面已经有的代码 (536行往后)
            - get_train_loader 这个函数需要改成 get_train_dataloader 和 HF Trainer 名称保持一致，同时传入 data_collator
            - train 的接口传入的 train_dataset 是 DailyDataset 类型的
            - 在 trainer 中 `__init__` 时设置一个 stop_training 的 flag，每个训练 batch 开始时检查 flag
        - runner.py
            - 实现 load_dataset 方法，返回 DailyDataset 类型
            - 实现 load_data_collator 方法，返回实现的 DataCollator 类型
