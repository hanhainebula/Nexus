import sys
import os

# 将项目的根目录添加到 sys.path 中
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from UniRetrieval.training.embedder.text_retrieval.__main__ import main



if __name__ == "__main__":
    main()  