from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class LaRSDataset(BaseSegDataset):  
    METAINFO = {
        'classes': ('background', 'water', 'sky'),
        'palette': [(0,0,255), (255, 0, 0), (255, 0, 255)]

    }

    def __init__(self,
                 data_root,
                 data_prefix,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 **kwargs):
        super().__init__(
            data_root=data_root,
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            **kwargs
        )
