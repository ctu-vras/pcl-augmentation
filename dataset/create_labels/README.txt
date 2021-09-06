--- Model Cylinder3D - rank 2 on semnatic segmentation task on Semantic KITTI ---

github: https://github.com/xinge008/Cylinder3D
arxiv: https://arxiv.org/pdf/2011.10033.pdf


Codes were adjusted for out purpose.  


To create prediction of semantic KITTI (pseudo labels, which we used for insertion of cars) run:

 python3 main.py --demo-folder "str:path to velodyne data" --save-folder "str:path to file, where you want save predictions" --cuda "int:number of cude, which you want to connect to"