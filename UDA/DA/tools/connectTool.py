import mysql.connector
import socket
import mysql.connector
import csv
import pandas as pd
import matplotlib.pyplot as plt


def query_to_df(db, query, is_print_query=False):
    if is_print_query:
        print("query: \n", query)
    cursor = db.cursor()
    cursor.execute(query)
    headers = [col[0] for col in cursor.description] # get headers
    rows = cursor.fetchall()
    df = pd.DataFrame(rows)
    df.columns = headers
    cursor.close()
    db.close()
    return df

def get_mysql_conn(cfg):
    return mysql.connector.connect(
        host="172.18.36.115",
        user="likesh",
        password="X!likesh123",
        database="ml_sh",
        auth_plugin='mysql_native_password'
    )

mysql_connect = get_mysql_conn()
def report_2_mysql(result_dict):
    cursor = mysql_connect.cursor()
    sql = """
    INSERT INTO ml_sh.t_meta_icl_selection VALUES 
        (%s, %s, %s, %s, %s,   %s, %s, %s, %s, %s,   %s, %s)
    """

    values = [(
        "11277",
        "ahsjdahs",
        "12873812",
        "zzz",
        2,

        "cifa10",
        "resnet18",
        "training",
        "saj",
        k,

        result_dict[k] * 100,
        ""

    ) for k in result_dict
    ]
    cursor.executemany(sql, values)
    mysql_connect.commit()
    cursor.close()


query = """
CREATE TABLE `CCL_Office31_classification` (
  `index_key` varchar(100) NOT NULL,
  `ts` varchar(100) DEFAULT NULL,
  `method` varchar(100) NOT NULL,
  `seed` int NOT NULL,
  `stage` varchar(10) NOT NULL,

  `epoch` int NOT NULL,
  `step` int NOT NULL,
  `ds` varchar(100) NOT NULL,
  `backbone` varchar(100) NOT NULL,
  `experiment` varchar(100) NOT NULL,

  `top1_acc` double DEFAULT NULL,
  `top5_acc` double DEFAULT NULL,
  `loss` double DEFAULT NULL,
  `ip` varchar(20) DEFAULT '0.0.0.0',
  `extra_text` text,
  `runing_time` int NOT NULL,
  PRIMARY KEY (`index_key`,`method`,`seed`,`stage`, `epoch`, `step`,`ds`,`backbone`,`experiment`)
) ENGINE=InnoDB DEFAULT CHARSET=latin1
"""


def create_table(db, is_print_query=False):
    if is_print_query:
        print("query: \n", query)
    cursor = db.cursor()
    cursor.execute(query)
    cursor.close()
    db.close()


result_dict = {"acc": 98, "loss": 0.12}

report_2_mysql(result_dict)


