from pydicom.filebase import DicomBytesIO
from numpy import int32
from PIL import Image
from pathlib import Path
from fastai.vision.core import PILBase
from fastai.medical.imaging import TensorDicom

def dcmread2(fn):
    dcm = Path(fn).dcmread(force=True)
    try: intercept=dcm[0x0028, 0x1052].value
    except: intercept=0
    try: slope=dcm[0x0028, 0x1053].value
    except: slope=1
    arr=dcm.pixel_array
    return intercept + arr*slope
class PILDicom2(PILBase):
    "same as PILDicom but changed pixel type to np.int32 as np.int16 cannot be handled by PIL"
    _open_args,_tensor_cls,_show_args = {},TensorDicom,TensorDicom._show_args
    @classmethod
    def create(cls, fn:(Path,str,bytes), mode=None)->None:
        "Open a `DICOM file` from path `fn` or bytes `fn` and load it as a `PIL Image`"
        # images are np.int16, but this cannont be handled by PIL. Will throw wrong mode error. 
        if isinstance(fn,bytes): im = Image.fromarray(dcmread2(DicomBytesIO(fn)))
        if isinstance(fn,(Path,str)): im = Image.fromarray(dcmread2(fn).astype(int32))
        im.load()
        im = im._new(im.im)
        return cls(im.convert(mode) if mode else im)