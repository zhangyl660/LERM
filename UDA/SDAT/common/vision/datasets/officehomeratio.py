import os
from typing import Optional
from .imagelistratio import ImageListRatio
from ._util import download as download_data, check_exits


class OfficeHomeratio(ImageListRatio):
    """`OfficeHome <http://hemanthdv.org/OfficeHome-Dataset/>`_ Dataset.

    Args:
        root (str): Root directory of dataset
        task (str): The task (domain) to create dataset. Choices include ``'Ar'``: Art, \
            ``'Cl'``: Clipart, ``'Pr'``: Product and ``'Rw'``: Real_World.
        download (bool, optional): If true, downloads the dataset from the internet and puts it \
            in root directory. If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image and returns a \
            transformed version. E.g, :class:`torchvision.transforms.RandomCrop`.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.

    .. note:: In `root`, there will exist following files after downloading.
        ::
            Art/
                Alarm_Clock/*.jpg
                ...
            Clipart/
            Product/
            Real_World/
            image_list/
                Art.txt
                Clipart.txt
                Product.txt
                Real_World.txt
    """
    download_list = [
        ("image_list", "image_list.zip", "https://cloud.tsinghua.edu.cn/f/ca3a3b6a8d554905b4cd/?dl=1"),
        ("Art", "Art.tgz", "https://cloud.tsinghua.edu.cn/f/4691878067d04755beab/?dl=1"),
        ("Clipart", "Clipart.tgz", "https://cloud.tsinghua.edu.cn/f/0d41e7da4558408ea5aa/?dl=1"),
        ("Product", "Product.tgz", "https://cloud.tsinghua.edu.cn/f/76186deacd7c4fa0a679/?dl=1"),
        ("Real_World", "Real_World.tgz", "https://cloud.tsinghua.edu.cn/f/dee961894cc64b1da1d7/?dl=1")
    ]
    image_list = {
        "Ar": "image_list/Art.txt",
        "Cl": "image_list/Clipart.txt",
        "Pr": "image_list/Product.txt",
        "Rw": "image_list/Real_World.txt",
        "A_C": "ArtoCl_256_20k.txt",
        "A_P": "ArtoPr_256_20k.txt",
        "A_R": "ArtoRw_256_20k.txt",
        "C_A": "CltoAr_256_20k.txt",
        "C_R": "CltoRw_256_20k.txt",
        "C_P": "CltoPr_256_20k.txt",
        "P_A": "PrtoAr_256_20k.txt",
        "P_C": "PrtoCl_256_20k.txt",
        "P_R": "PrtoRw_256_20k.txt",
        "R_A": "RwtoAr_256_20k.txt",
        "R_C": "RwtoCl_256_20k.txt",
        "R_P": "RwtoPr_256_20k.txt",
        "A_C_p": "Ar2Cl_p.txt",
        "A_P_p": "Ar2Pr_p.txt",
        "A_R_p": "Ar2Rw_p.txt",
        "C_A_p": "Cl2Ar_p.txt",
        "C_P_p": "Cl2Pr_p.txt",
        "C_R_p": "Cl2Rw_p.txt",
        "P_A_p": "Pr2Ar_p.txt",
        "P_C_p": "Pr2Cl_p.txt",
        "P_R_p": "Pl2Rw_p.txt",
        "R_A_p": "Rw2Ar_p.txt",
        "R_C_p": "Rw2Cl_p.txt",
        "R_P_p": "Rw2Pr_p.txt",
        "A_P_ws": "ArtoPr_256_20k_ws.txt",
        "A_C_ws": "ArtoCl_256_20k_ws.txt",
        "A_C_ws_A100": "ArtoCl_256_20k_A100.txt",
        "C_P_ws": "CltoPr_256_20k_ws.txt",
        "C_A_ws": "CltoAr_256_20k_ws.txt",
        "soft_A_P_temp1": "A_P_temp_1.txt",
        "soft_A_P_temp10": "A_P_temp_10.txt",
        "soft_A_P_temp1_mlp": "A_P_temp_1_mlp.txt",
        "soft_A_P_temp10_mlp": "A_P_temp_10_mlp.txt",
    }
    CLASSES = ['Drill', 'Exit_Sign', 'Bottle', 'Glasses', 'Computer', 'File_Cabinet', 'Shelf', 'Toys', 'Sink',
               'Laptop', 'Kettle', 'Folder', 'Keyboard', 'Flipflops', 'Pencil', 'Bed', 'Hammer', 'ToothBrush', 'Couch',
               'Bike', 'Postit_Notes', 'Mug', 'Webcam', 'Desk_Lamp', 'Telephone', 'Helmet', 'Mouse', 'Pen', 'Monitor',
               'Mop', 'Sneakers', 'Notebook', 'Backpack', 'Alarm_Clock', 'Push_Pin', 'Paper_Clip', 'Batteries', 'Radio',
               'Fan', 'Ruler', 'Pan', 'Screwdriver', 'Trash_Can', 'Printer', 'Speaker', 'Eraser', 'Bucket', 'Chair',
               'Calendar', 'Calculator', 'Flowers', 'Lamp_Shade', 'Spoon', 'Candles', 'Clipboards', 'Scissors', 'TV',
               'Curtains', 'Fork', 'Soda', 'Table', 'Knives', 'Oven', 'Refrigerator', 'Marker']
    # CLASSES = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
    #            'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
    #            'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
    #            'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse',
    #            'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin',
    #            'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda',
    #            'Speaker', 'Spoon', 'TV', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'Webcam']

    def __init__(self, root: str, task: str, baseDatasetLength, ratio , download: Optional[bool] = True,
                                              **kwargs):
        assert task in self.image_list
        dataset_length = ratio * baseDatasetLength
        data_list_file = os.path.join(root, self.image_list[task])

        if download:
            list(map(lambda args: download_data(root, *args), self.download_list))
        else:
            list(map(lambda file_name, _: check_exits(root, file_name), self.download_list))

        super(OfficeHomeratio, self).__init__(root, OfficeHomeratio.CLASSES, data_list_file=data_list_file, dataset_length=dataset_length,  **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())
