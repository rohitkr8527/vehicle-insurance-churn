#---------------------------
# from src.logger import logging
#------------------------
# from src.logger import logging
# from src.exception import MyException
# import sys

#-------------------------------------------
from src.pipline.training_pipeline import TrainPipeline
pipeline = TrainPipeline()
pipeline.run_pipeline()
