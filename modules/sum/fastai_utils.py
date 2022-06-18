from fastai.torch_core import params
            
def sum_splitter(sum_model): return [params(sum_model.models[0].vit_backbone), #params(sum_model.models[0][0]), 
                                     params(sum_model.models[1][0]), 
                                     params(sum_model.models[0].mlp),          #params(sum_model.models[0][1]), 
                                     params(sum_model.models[1][1]), 
                                     params(sum_model.models[2].embeds),
                                     params(sum_model.mixed_cls)] 