import torch
from typing import OrderedDict
from UniRetrieval.modules.arguments import get_seq_data
from . import AverageAggregator, MLPModule, MultiFeatEmbedding


__all__ = ["BaseQueryEncoderWithSeq"]

class BaseQueryEncoderWithSeq(torch.nn.Module):
    def __init__(self, context_embedding, item_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_encoder = item_encoder
        self.context_embedding = context_embedding
        self.seq_aggragation = AverageAggregator(dim=1)


    def forward(self, batch):
        seq_data = get_seq_data(batch)
        seq_emb = self.item_encoder(seq_data)   # BxLxD1
        seq_emb = self.seq_aggragation(seq_emb) # BxD
        context_emb = self.context_embedding(batch) # BxD2
        cat_emb = torch.cat([seq_emb, context_emb], dim=-1)
        return cat_emb
        
class QueryEncoder(torch.nn.Module):
    def __init__(self, data_config, model_config, item_encoder):
        super(QueryEncoder, self).__init__()
        
        self.context_emb = MultiFeatEmbedding(
            features=data_config.context_features,
            stats=data_config.stats,
            embedding_dim=model_config.embedding_dim
        )
        
        self.base_encoder = BaseQueryEncoderWithSeq(
            context_embedding=self.context_emb,
            item_encoder=item_encoder
        )
        
        output_dim = model_config.mlp_layers[-1] + self.context_emb.total_embedding_dim
        
        self.mlp = MLPModule(
            mlp_layers=[output_dim] + model_config.mlp_layers,
            activation_func=model_config.activation,
            dropout=model_config.dropout,
            bias=True,
            batch_norm=model_config.batch_norm,
            last_activation=False,
            last_bn=False
        )
        
        self.encoder_mlp_sequence = torch.nn.Sequential(OrderedDict([
            ("encoder", self.base_encoder),
            ("mlp", self.mlp)
        ]))

    def forward(self, x):
        return self.encoder_mlp_sequence(x)
    
    