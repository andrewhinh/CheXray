global glb_vis_logits
global glb_txtcls_logits
global glb_tab_logits
    
def get_vis_logits(self, inp, out):
    global glb_vis_logits
    glb_vis_logits = inp
    #return None
def get_txtcls_logits(self, inp, out):
    global glb_txtcls_logits
    glb_txtcls_logits = inp
    #return None
def get_tab_logits(self, inp, out):
    global glb_tab_logits
    glb_tab_logits = inp
    #return None    