from fastai.vision.all import *

def rep_gen(model): return [params(model.visual_extractor), params(model.encoder_decoder)]
class SelectPred(Callback):
    def after_pred(self): 
        if self.pred[0].dtype==torch.float16: self.learn.pred[0] = to_float(self.pred[0]) #Since model returns tuple and only needs predictions
            
def multi_rep_gen(multiview_model): return [params(multiview_model.models[0].visual_extractor),
                                            params(multiview_model.models[1].visual_extractor),
                                            params(multiview_model.models[2].visual_extractor),
                                            params(multiview_model.models[3].visual_extractor),
                                            params(multiview_model.models[4].visual_extractor),
                                            params(multiview_model.models[5].visual_extractor),
                                            params(multiview_model.models[6].visual_extractor),
                                            params(multiview_model.models[7].visual_extractor),
                                            params(multiview_model.models[8].visual_extractor),
                                            params(multiview_model.models[9].visual_extractor),
                                            params(multiview_model.models[10].visual_extractor),
                                            params(multiview_model.models[11].visual_extractor),
                                            params(multiview_model.models[12].visual_extractor),
                                            params(multiview_model.models[13].visual_extractor),
                                            params(multiview_model.models[0].encoder_decoder),
                                            params(multiview_model.models[1].encoder_decoder),
                                            params(multiview_model.models[2].encoder_decoder),
                                            params(multiview_model.models[3].encoder_decoder),
                                            params(multiview_model.models[4].encoder_decoder),
                                            params(multiview_model.models[5].encoder_decoder),
                                            params(multiview_model.models[6].encoder_decoder),
                                            params(multiview_model.models[7].encoder_decoder),
                                            params(multiview_model.models[8].encoder_decoder),
                                            params(multiview_model.models[9].encoder_decoder),
                                            params(multiview_model.models[10].encoder_decoder),
                                            params(multiview_model.models[11].encoder_decoder),
                                            params(multiview_model.models[12].encoder_decoder),
                                            params(multiview_model.models[13].encoder_decoder),
                                            params(multiview_model.mixed_cls)] 