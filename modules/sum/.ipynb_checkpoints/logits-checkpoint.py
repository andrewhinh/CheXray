global glb_ap_logits
global glb_ap_axial_logits
global glb_ap_lld_logits
global glb_ap_rld_logits
global glb_pa_logits
global glb_pa_lld_logits
global glb_pa_rld_logits
global glb_lateral_logits
global glb_ll_logits
global glb_lao_logits
global glb_rao_logits
global glb_swimmers_logits
global glb_xtable_lateral_logits
global glb_lpo_logits
global glb_txtcls_logits
global glb_tab_logits
    
def get_ap_logits(self, inp, out):
    global glb_ap_logits
    glb_ap_logits = inp
    #return None
def get_ap_axial_logits(self, inp, out):
    global glb_ap_axial_logits
    glb_ap_axial_logits = inp
    #return None
def get_ap_lld_logits(self, inp, out):
    global glb_ap_lld_logits
    glb_ap_lld_logits = inp
    #return None
def get_ap_rld_logits(self, inp, out):
    global glb_ap_rld_logits
    glb_ap_rld_logits = inp
    #return None
def get_pa_logits(self, inp, out):
    global glb_pa_logits
    glb_pa_logits = inp
    #return None
def get_pa_lld_logits(self, inp, out):
    global glb_pa_lld_logits
    glb_pa_lld_logits = inp
    #return None
def get_pa_rld_logits(self, inp, out):
    global glb_pa_rld_logits
    glb_pa_rld_logits = inp
    #return None
def get_lateral_logits(self, inp, out):
    global glb_lateral_logits
    glb_lateral_logits = inp
    #return None
def get_ll_logits(self, inp, out):
    global glb_ll_logits
    glb_ll_logits = inp
    #return None
def get_lao_logits(self, inp, out):
    global glb_lao_logits
    glb_lao_logits = inp
    #return None
def get_rao_logits(self, inp, out):
    global glb_rao_logits
    glb_rao_logits = inp
    #return None
def get_swimmers_logits(self, inp, out):
    global glb_swimmers_logits
    glb_swimmers_logits = inp
    #return None
def get_xtable_lateral_logits(self, inp, out):
    global glb_xtable_lateral_logits
    glb_xtable_lateral_logits = inp
    #return None
def get_lpo_logits(self, inp, out):
    global glb_lpo_logits
    glb_lpo_logits = inp
    #return None
def get_txtcls_logits(self, inp, out):
    global glb_txtcls_logits
    glb_txtcls_logits = inp
    #return None
def get_tab_logits(self, inp, out):
    global glb_tab_logits
    glb_tab_logits = inp
    #return None    