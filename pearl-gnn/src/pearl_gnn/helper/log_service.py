import logging
import uuid
import os

class LogService:
    
    def __init__(self):
        name = str(uuid.uuid4())
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        os.makedirs("out/baseline", exist_ok=True)
        handler = logging.FileHandler(os.path.join("out/baseline", "train_logs.txt"))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", "%Y/%m/%d %H:%M:%S")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.val_writer = open(os.path.join("out/baseline", "evaluate_logs.txt"), "a")