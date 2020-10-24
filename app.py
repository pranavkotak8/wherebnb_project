import pandas as pd
import numpy as np
import mapping as mp
import random
import os
import folium
import shutil
import time
from folium import plugins
import flask
from flask import Flask, render_template, request, redirect, session, url_for, make_response
import pickle

class MyFlask(flask.Flask):
    def get_send_file_max_age(self, name):
        if name.lower().endswith('map.html'):
            return 0
        return flask.Flask.get_send_file_max_age(self, name)

app = MyFlask(__name__,static_folder='static')

city_dict1={'Asheville': [35.57874453940702, -82.55658508002875],
 'Austin': [30.283220299244316, -97.75300299176298],
 'Boston': [42.33751953150349, -71.08297803906524],
 'Broward County': [26.100730532353005, -80.15030725174958],
 'Cambridge': [42.37262656948493, -71.10889725947523],
 'Chicago': [41.89904946884507, -87.66404224511494],
 'Clark County': [36.115950431555426, -115.1553760800795],
 'Columbus': [39.98893217819731, -82.99525612825397],
 'Denver': [39.74085208778584, -104.97419012635699],
 'Jersey City': [40.72694050241162, -74.05479675643095],
 'Los Angeles': [34.04613022869341, -118.31592329064955],
 'Nashville': [36.161298064831485, -86.76949305750145],
 'New Orleans': [29.95840226825378, -90.0748187240168],
 'New York City': [40.729633214975856, -73.95074746253856],
 'Oakland': [37.8118978308365, -122.23998523720358],
 'Pacific Grove': [36.621463016759776, -121.92083078212298],
 'Portland': [45.52818764540984, -122.65179200186857],
 'Rhode Island': [41.56768859734551, -71.41139883969366],
 'Salem': [44.92734410153466, -123.03917499306928],
 'San Clara Country': [37.352147836694485, -121.96639046255807],
 'San Diego': [32.77017793534333, -117.18311561189918],
 'San Francisco': [37.76716611693177, -122.42989077467759],
 'San Mateo County': [37.56024389842393, -122.33386482662019],
 'Santa Cruz County': [36.98839068109479, -121.9780671737747],
 'Seattle': [47.62583064067392, -122.33331940176427],
 'Twin Cities MSA': [44.972577104831544, -93.27391850699486],
 'Washington D.C.': [38.911354089688395, -77.01716847357274]}

app.secret_key = 'wherebnb'

@app.route('/')
def main():
    return render_template('index.html')

# @app.route('/tourist/price')
# def tourist():
#     return render_template('tourist.html')

@app.route('/Contact me')
def contactme():
    return render_template('contact.html')

@app.route('/About me')
def aboutme():
    return render_template('about.html')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html')


# @app.route('/host')
# def host():
#     session['customer'] = 'Host'
#     return render_template('host.html', customer=session['customer'])

# @app.route('/host/price', methods=['GET', 'POST'])
# def hostprice():
#     session['customer'] = 'Host'
#     return render_template('tourist.html', customer=session['customer'], city_target='Select', city_dict=mp.id_city_dict,
#                            popularity_dict=mp.pop_dict, popularity_target='Select',
#                            room_type_dict=mp.room_type_dict, room_type_target='Select')

# @app.route('/host/popularity', methods=['GET', 'POST'])
# def hostpopularity():
#     session['customer'] = 'Host'
#     return render_template('Popularity.html', customer=session['customer'], city_target='Select', city_dict=mp.id_city_dict,
#                            popularity_dict=mp.pop_dict, popularity_target='Select',
#                            room_type_dict=mp.room_type_dict, room_type_target='Select')

# @app.route('/host/title', methods=['GET', 'POST'])
# def hosttitle():
#     session['customer'] = 'Host'
#     return render_template('aibnbanalytics.html', customer=session['customer'], city_target='Select', city_dict=mp.id_city_dict,
#                            popularity_dict=mp.pop_dict, popularity_target='Select',
#                            room_type_dict=mp.room_type_dict, room_type_target='Select')


model= pickle.load(open('decision_tree_clf.pkl','rb'))

model1= pickle.load(open('LGBM_Final_Model.pkl','rb'))

@app.route('/poppredict')
def poppredict():
    
    
    return render_template('host.html')


@app.route('/poppredict1', methods=['POST'])
def poppredict1():
    if request.method == 'POST':
        d=request.form.to_dict(flat=False)
        print(d)
        l=[y for y in request.form.values()]
        print(l)

        city_id_dict = {'Asheville': 0,'Austin': 1,'Boston': 2,'Broward County': 3,
                'Cambridge': 4,'Chicago': 5,'Clark County': 6,'Columbus': 7,
                'Denver': 8,'Jersey City': 9,'Los Angeles': 10,
                'Nashville': 11,'New Orleans': 12,'New York City': 13,'Oakland': 14,
                'Pacific Grove': 15,'Portland': 16,'Rhode Island': 17,'Salem': 18,
                'San Clara Country': 19,'San Diego': 20,'San Francisco': 21,
                'San Mateo County': 22,'Santa Cruz County': 23,'Seattle': 24,
                'Twin Cities': 25,'Washington D.C.': 26}

        id_city_dict = {v:k for k,v in city_id_dict.items()}
        City=id_city_dict[int(d['from'][0])]
                

        df = pd.read_csv('Wherebnb Map.csv')

        dfs = df[df['city']==City].drop(['city'], axis=1)
        dfs = dfs.values.tolist()

        map = folium.Map(city_dict1[City], zoom_start=12,
                    width='100%',
                    height='100%')
        clusts = plugins.MarkerCluster().add_to(map)

        for lname, hood, lat, long, rtype, price, nrev, pop in dfs:
            pop_up = f'{lname}\nRoom Type: {rtype}\nPrice: {price}\nReviews: {nrev}\nPopularity: {pop}'
            folium.Marker([lat, long], 
                    icon=folium.Icon(color='black',icon='hotel', prefix='fa'),
                    popup=pop_up).add_to(clusts)

        map_path = app.root_path + '/' + 'static/map.html'
        map.save(map_path)

    
        pred=model.predict([l])
        if pred == 0:
            pred1="New Listing"
        elif pred == 1:
            pred1="Low"
        elif pred == 2:
            pred1="Average"
        elif pred == 3:
            pred1="High"
        else:
            pred1="Extremely Popular"
    k1=("The Predicted Popularity is "+pred1)
    
    return render_template('host.html',k1=k1)





@app.route('/pricepredict')
def pricepredict():

    
    return render_template('tourist.html')


@app.route('/pricepredict1', methods=['POST'])
def pricepredict1():
    if request.method == 'POST':
        print("HELLO")
        fd= request.form.to_dict(flat=False)
        print(fd)


    
        l1=[y for y in request.form.values()]
        print(l1)

        
        l1[1]=np.log1p(float(l1[1]))
        l1[2]=np.sqrt(float(l1[2]))


        city_id_dict = {'Asheville': 0,'Austin': 1,'Boston': 2,'Broward County': 3,
                'Cambridge': 4,'Chicago': 5,'Clark County': 6,'Columbus': 7,
                'Denver': 8,'Jersey City': 9,'Los Angeles': 10,
                'Nashville': 11,'New Orleans': 12,'New York City': 13,'Oakland': 14,
                'Pacific Grove': 15,'Portland': 16,'Rhode Island': 17,'Salem': 18,
                'San Clara Country': 19,'San Diego': 20,'San Francisco': 21,
                'San Mateo County': 22,'Santa Cruz County': 23,'Seattle': 24,
                'Twin Cities': 25,'Washington D.C.': 26}

        id_city_dict = {v:k for k,v in city_id_dict.items()}
    
        City=id_city_dict[int(fd['from'][0])]

        df = pd.read_csv('Wherebnb Map.csv')

        dfs = df[df['city']==City].drop(['city'], axis=1)
        dfs = dfs.values.tolist()

        map = folium.Map(city_dict1[City], zoom_start=12,
                    width='100%',
                    height='100%')
        clusts = plugins.MarkerCluster().add_to(map)

        for lname, hood, lat, long, rtype, price, nrev, pop in dfs:
            pop_up = f'{lname}\nRoom Type: {rtype}\nPrice: {price}\nReviews: {nrev}\nPopularity: {pop}'
            folium.Marker([lat, long], 
                    icon=folium.Icon(color='black',icon='hotel', prefix='fa'),
                    popup=pop_up).add_to(clusts)

        map_path = app.root_path + '/' + 'static/map.html'
        map.save(map_path)


        pred2=model1.predict([l1])

        pred3=np.expm1(float(pred2))
    
        pred3=np.around(pred3,2)

        k=("The Predicted Price is $"+str(pred3))
    
    return render_template('tourist.html',k=k,current_time=int(time.time()))

if __name__ == '__main__':
    app.run(debug=True)