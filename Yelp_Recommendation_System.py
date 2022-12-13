# Yelp Recommender System

import findspark
findspark.init()
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
import os
import sys
import math
import json
import time
import xgboost as xgb
import pickle




def mdl_params_func():
    params_dict = {}
    max_depth_val = 15
    eta_val = 0.3
    silent_val = 1
    booster_val = 'gbtree'
    objective_val = 'reg:linear'      
    
    params_dict['max-depth'] = max_depth_val
    params_dict['eta'] = eta_val
    params_dict['silent'] = silent_val
    params_dict['booster'] = booster_val
    params_dict['objective'] = objective_val
    return params_dict




def price_func(att, k):
    if att:
        if k in att.keys():
            return int(att.get(k))
    return 0


       
            
def output_mdl_based_func(output_file, opt, te_data):
    f = open(output_file, 'w')
    f.write("user_id, business_id, prediction\n")
    for j in range(0, len(opt)):
        f.write(te_data[j][0] + "," + te_data[j][1] + "," + str( max(1, min(5, opt[j]))) + "\n")

        

def output_func(output_file, pred_lst):
    f = open(output_file, 'w')
    f.write("user_id, business_id, prediction\n")
    for j in range(len(pred_lst)):
        f.write(str(pred_lst[j][0][0]) + "," + str(pred_lst[j][0][1]) + "," + str(pred_lst[j][1]) + "\n")
    f.close()

    

def bus_data_func(business):
    bus_data = []    
    bus_caters, bus_tv, bus_kid, bus_dog = 0.5, 0.5, 0.5, 0.5
    bus_delv = 0.5
    bus_bkpark, bus_outseat = 0.5, 0.5
    bus_credcards, bus_groups, bus_reserv, bus_takeout = 0.5, 0.5, 0.5, 0.5
    
    if 'attributes' in business and business['attributes']:
        if 'Caters' in business['attributes']:
            bus_caters = 1.0 if business['attributes']['Caters'] == 'True' else 0.0
        if 'HasTV' in business['attributes']:
            bus_tv = 1.0 if business['attributes']['HasTV'] == 'True' else 0.0
        if 'GoodForKids' in business['attributes']:
            bus_kid = 1.0 if business['attributes']['GoodForKids'] == 'True' else 0.0
        if 'DogsAllowed' in business['attributes']:
            bus_dog = 1.0 if business['attributes']['DogsAllowed'] == 'True' else 0.0
        if 'RestaurantsDelivery' in business['attributes']:
            bus_delv = 1.0 if business['attributes']['RestaurantsDelivery'] == 'True' else 0.0
        if 'BikeParking' in business['attributes']:
            bus_bkpark = 1.0 if business['attributes']['BikeParking'] == 'True' else 0.0
        if 'OutdoorSeating' in business['attributes']:
            bus_outseat = 1.0 if business['attributes']['OutdoorSeating'] == 'True' else 0.0        
        if 'BusinessAcceptsCreditCards' in business['attributes']:
            bus_credcards = 1.0 if business['attributes']['BusinessAcceptsCreditCards'] == 'True' else 0.0
        if 'RestaurantsGoodForGroups' in business['attributes']:
            bus_groups = 1.0 if business['attributes']['RestaurantsGoodForGroups'] == 'True' else 0.0
        if 'RestaurantsReservations' in business['attributes']:
            bus_reserv = 1.0 if business['attributes']['RestaurantsReservations'] == 'True' else 0.0
        if 'RestaurantsTakeOut' in business['attributes']:
            bus_takeout = 1.0 if business['attributes']['RestaurantsTakeOut'] == 'True' else 0.0
                    
    bus_data = bus_data + [bus_caters, bus_tv, bus_dog, bus_kid, bus_delv, bus_bkpark, bus_bkpark, bus_outseat, bus_credcards, bus_groups, bus_reserv, bus_takeout]
    return (business['business_id'], tuple(bus_data))





def bus_checkin_func(business_checkin):
    checkin_temp = business_checkin['time']
    checkin_cnt = sum(checkin_temp.values())
    return (business_checkin['business_id'], checkin_cnt)




def tip_func(tips_vector, b, u):
    set_key = (b,u)
    return tips_vector.get(set_key)




def user_data_func(user):
    usr = []
    compliment_hot, compliment_more, compliment_profile, compliment_cute = 0.5, 0.5, 0.5, 0.5
    compliment_list, compliment_note, compliment_plain = 0.5, 0.5, 0.5
    compliment_cool, compliment_funny, compliment_writer, compliment_photos = 0.5, 0.5, 0.5, 0.5

    if 'compliment_hot' in user:
        compliment_hot = user['compliment_hot']
    else:
        compliment_hot = 0.5
    
    if 'compliment_more' in user:
        compliment_more = user['compliment_more']
    else:
        compliment_more = 0.5

    if 'compliment_profile' in user:
        compliment_profile = user['compliment_profile']
    else:
        compliment_profile = 0.5
    
    if 'compliment_cute' in user:
        compliment_cute = user['compliment_cute']
    else:
        compliment_cute = 0.5

    if 'compliment_list' in user:
        compliment_list = user['compliment_list']
    else:
        compliment_list = 0.5
       
    if 'compliment_note' in user:
        compliment_note = user['compliment_note']
    else:
        compliment_note = 0.5

    if 'compliment_plain' in user:
        compliment_plain = user['compliment_plain']
    else:
        compliment_plain = 0.5
    
    if 'compliment_cool' in user:
        compliment_cool = user['compliment_cool']
    else:
        compliment_cool = 0.5
          
    if 'compliment_funny' in user:
        compliment_funny = user['compliment_funny']
    else:
        compliment_funny = 0.5

    if 'compliment_writer' in user:
        compliment_writer = user['compliment_writer']
    else:
        compliment_writer = 0.5
    
    if 'compliment_photos' in user:
        compliment_photos = user['compliment_photos']
    else:
        compliment_photos = 0.5

    usr = usr + [compliment_hot, compliment_more, compliment_profile, compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_photos]
    return (user['user_id'], tuple(usr))

    

    
    
bus_cat = {}
def bus_cat_func(cat):
    cate = str(cat).replace(" ", "") 
    for i in cate.split(','):
        if i not in bus_cat.keys():
            bus_cat[i] = 1
        else:
            bus_cat[i] = bus_cat[i] + 1
            
            
            
            
            
def bus_cat_gen_func(cat):
    cate = str(cat).replace(" ", "") 
    cate = cate.split(',')
    values = []
    for idx in cate:
        values.append(bus_cat[idx])
    return(max(values)/len(cate))
            
        
        
def usr_frnd_bus(usr, x):
    if x not in tips_user_bus.keys() or usr not in tips_user_bus.keys():
        return 0
    elif tips_user_bus[x] == tips_user_bus[usr]:
        return 1
    else:
        return 0
    
    
def usr_frnd(usr, usr_frn):
    lst = str(usr_frn).replace(" ", "").split(',')
    lst1 = list(map(lambda x: usr_frnd_bus(usr, x), lst)) 
    return sum(lst1)



def features_func(r):
    f = []
    f.extend(usr_json_map_rdd.get(r[0]))
    f.extend(bus_json_map_rdd.get(r[1]))
    f.extend([photo_count.get(r[1]) if photo_count.get(r[1]) is not None else 0])
    f.extend(bus_data.get(r[1]))
    f.extend([checkin_data.get(r[1]) if checkin_data.get(r[1]) is not None else 0])
    f.extend([tip_func(tips_data, r[1], r[0]) if tip_func(tips_data, r[1], r[0]) is not None else 0])
    f.extend(user_data.get(r[0]))
    f.extend([bus_city.get(r[1]) if bus_city.get(r[1]) is not None else 0])
    f.extend([bus_state.get(r[1]) if bus_state.get(r[1]) is not None else 0])
    f.extend([user_count.get(r[0]) if user_count.get(r[0]) is not None else 0])
    f.extend([bus_count.get(r[1]) if bus_count.get(r[1]) is not None else 0])
    f.extend([bus_category_data.get(r[1]) if bus_category_data.get(r[1]) is not None else 0])
    f.extend([user_frnds.get(r[0]) if user_frnds.get(r[0]) is not None else 0])
    return f




    
start_time = time.time()

# Train, Test, Output data files
tr_folder = sys.argv[1]
te_file = sys.argv[2]
output_file = sys.argv[3]


sc = SparkContext.getOrCreate();
spark = SparkSession(sc)
sc.setLogLevel('WARN')


tr_rdd = sc.textFile(os.path.join(tr_folder, 'yelp_train.csv'))
tr_header = tr_rdd.first()
tr_data = tr_rdd.filter(lambda x: x != tr_header).map(lambda x: x.split(','))


te_rdd = sc.textFile(te_file)
te_header = te_rdd.first()
te_rdd = te_rdd.filter(lambda x: x != te_header)




# Model based - Feature generation
usr_json_map_rdd = sc.textFile(os.path.join(tr_folder, 'user.json')).map(json.loads).map(lambda i: ((i["user_id"]), (i["review_count"], i["useful"], i["fans"], i["funny"], i["cool"], i["average_stars"]))).collectAsMap()
usr_json_map_rdd


bus_json_map_rdd = sc.textFile(os.path.join(tr_folder, 'business.json')).map(json.loads).map(lambda i: ((i['business_id']), (i['stars'], i['review_count'], price_func(i['attributes'], 'RestaurantsPriceRange2')))).collectAsMap()
bus_json_map_rdd


photo_count = sc.textFile(os.path.join(tr_folder, 'photo.json')).map(lambda x: (json.loads(x)['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda i: ((i[0]), (i[1]))).collectAsMap()
photo_count


bus_data = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(bus_data_func).collectAsMap()
bus_data


checkin_data = sc.textFile(os.path.join(tr_folder, 'checkin.json')).map(lambda x: json.loads(x)).map(bus_checkin_func).collectAsMap()
checkin_data


with open('tips_dict.pickle', 'rb') as fl:
    tips_data = pickle.load(fl)


user_data = sc.textFile(os.path.join(tr_folder, 'user.json')).map(lambda x: json.loads(x)).map(user_data_func).collectAsMap()
user_data


bus_city_temp = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['city'], 1)).reduceByKey(lambda x,y: x+y).collectAsMap()
bus_city_temp
bus_city = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], bus_city_temp[x['city']])).collectAsMap()
bus_city


bus_state_temp = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['state'], 1)).reduceByKey(lambda x,y: x+y).collectAsMap()
bus_state_temp
bus_state = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], bus_state_temp[x['state']])).collectAsMap()
bus_state


user_count = sc.textFile(os.path.join(tr_folder, 'review_train.json')).map(lambda x: (json.loads(x)['user_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda i: ((i[0]), (i[1]))).collectAsMap()
user_count


bus_count = sc.textFile(os.path.join(tr_folder, 'review_train.json')).map(lambda x: (json.loads(x)['business_id'], 1)).reduceByKey(lambda x,y: x+y).map(lambda i: ((i[0]), (i[1]))).collectAsMap()
bus_count


category_data = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x['categories'])).collectAsMap()
category_data
for i in list(category_data.keys()):
    bus_cat_func(category_data[i])
bus_category_data = sc.textFile(os.path.join(tr_folder, 'business.json')).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], bus_cat_gen_func(x['categories']))).collectAsMap()
bus_category_data


tips_user_bus = sc.textFile(os.path.join(tr_folder, 'tip.json')).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x['business_id'])).collectAsMap()
tips_user_bus
user_frnds = sc.textFile(os.path.join(tr_folder, 'user.json')).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], usr_frnd(x['user_id'], x['friends']))).sortBy(lambda x: x[1]).collectAsMap()
user_frnds



# Training
data_training = []
lbl = []
for tr_row in tr_data.collect():
    data_training.append(features_func(tr_row))
    lbl.append(float(tr_row[2]))
    
data_training = np.asarray(data_training)
lbl = np.asarray(lbl)
data_trained = xgb.DMatrix(data_training, label=lbl)
xgb_model = xgb.train(mdl_params_func(), data_trained, 100)



# Testing
val_te_data = te_rdd.map(lambda i: i.split(',')).map(lambda i: (i[0], i[1])).collect()
te_data = []
for te_row in val_te_data:
    te_data.append(features_func(te_row))
te_data = np.asarray(te_data)
pred = xgb_model.predict(xgb.DMatrix(te_data))
tmp_file = "mdlbased_output_temp.csv"
output_mdl_based_func(tmp_file, pred, val_te_data)




opt_rdd = sc.textFile(tmp_file)
opt_header = opt_rdd.first()
opt_data = opt_rdd.filter(lambda j: j != opt_header).map(lambda j: j.split(','))
modelbased_pred = opt_data.map(lambda j: (((j[0]), (j[1])), float(j[2])))

output_func(output_file, modelbased_pred.collect())
test_data_dict = te_rdd.map(lambda x: x.split(",")).map(lambda x: (((x[0]), (x[1])), float(x[2])))
joined_data = test_data_dict.join(modelbased_pred).map(lambda x: (abs(x[1][0] - x[1][1])))
rmse_rdd = joined_data.map(lambda x: x ** 2).reduce(lambda x, y: x + y)
rmse = math.sqrt(rmse_rdd / modelbased_pred.count())
print("Duration : ", time.time() - start_time, ", RMSE : ", rmse)