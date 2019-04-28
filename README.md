# MogicPinyin
Pinyin input method based on 2-gram and 3-gram models

## Usage
Make sure you have grabbed all the trained data in `trained` folder. You can change locations of files in config.py. Note character list and pinyin list files in data folder is also indispensable.
### Build
In the root folder of the project
```
cd src
python3 setup.py build_ext --inplace  ## Build necessary modules
python3 main.py  ## Run main program
```
### Available arguments
```
python3 main.py -n 2 ## or 3 to use 3-gram model(slower but more accurate)
```
