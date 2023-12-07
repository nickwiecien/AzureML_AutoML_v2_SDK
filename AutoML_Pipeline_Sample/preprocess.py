Add inline comments to the code below. DO NOT CHANGE ANY OF THE CODE, ONLY ADD COMMENTS AND DOC STRINGS.  
  
## CODE: import argparse  
import datetime  
from pathlib import Path  
import yaml  
from mltable import load  
import time  
import mltable  
import random  
  
def parse_args():  
    # setup arg parser  
    parser = argparse.ArgumentParser()  
  
    # add arguments  
    parser.add_argument("--delta_table", type=str)  
    parser.add_argument("--preprocessed_train_data", type=str)  
    parser.add_argument("--preprocessed_test_data", type=str)  
    # parse args  
    args = parser.parse_args()  
    print("args received ", args)  
    # return args  
    return args  
  
def main(args):  
    """  
    Preprocessing of training/test data  
    """  
      
    current_timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  
    tbl = mltable.from_delta_lake(  
        delta_table_uri=args.delta_table, timestamp_as_of=current_timestamp  
    )  
    data = tbl.to_pandas_dataframe()  
    data = data.drop(columns=['Cycle_Index'])  
      
    unique_sources = list(data['Source'].unique())  
  
    test_keys = random.sample(unique_sources, 2)  
  
    train_keys = [x for x in unique_sources if x not in test_keys]  
  
    train_data = data[data['Source'].isin(train_keys)]  
    train_data = train_data.drop(columns=['Source'])  
  
    test_data = data[data['Source'].isin(test_keys)]  
    test_data = test_data.drop(columns=['Source'])  
  
    # write preprocessed train data in output path  
    train_data.to_csv(  
        args.preprocessed_train_data + "/train_data.csv",  
        index=False,  
        header=True,  
    )  
  
    # write preprocessed validation data in output path  
    test_data.to_csv(  
        args.preprocessed_test_data + "/test_data.csv",  
        index=False,  
        header=True,  
    )  
  
    # Write MLTable yaml file as well in output folder  
    # Since in this example we are not doing any preprocessing, we are just copying same yaml file from input,change it if needed  
  
    # read and write MLModel yaml file for train data  
    yaml_str = """  
    paths:  
      - file: ./train_data.csv  
    transformations:  
      - read_delimited:  
          delimiter: ','  
          encoding: 'ascii'  
          empty_as_string: false  
    """  
    with open(args.preprocessed_train_data + "/MLTable", "w") as file:  
        file.write(yaml_str)  
  
    # read and write MLModel yaml file for validation data  
    yaml_str = """  
    paths:  
      - file: ./test_data.csv  
    transformations:  
      - read_delimited:  
          delimiter: ','  
          encoding: 'ascii'  
          empty_as_string: false  
    """  
    with open(args.preprocessed_test_data + "/MLTable", "w") as file:  
        file.write(yaml_str)  
  
  
# run script  
if __name__ == "__main__":  
    # parse args  
    args = parse_args()  
  
    # run main function  
    main(args)  