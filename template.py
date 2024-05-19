import os
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO,format="[%(asctime)s] %(message)s")
list_of_files=[
    "src/__init__.py",
    "src/helper.py",
    "src/prompts.py",
    "setup.py",
    "research/experiment.ipynb",
    "app.py",
    "store_index.py",
    "static/.gitkeep",
    "template/chat.html"
]
for filepath in list_of_files:
    path=Path(filepath)
    folderpath,file=os.path.split(path)
    if folderpath!="":
        os.makedirs(folderpath,exist_ok=True)
        logging.info(f"folder is created {folderpath} for the file{file}")
    if (not os.path.exists(file)) or (os.path.getsize(file)==0):
        with open(path,"w") as f:
            pass
        logging.info(f"file created with {file}")
    else:
        logging.info(f" file already created {file}")