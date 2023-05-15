from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application



@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('form.html')
    
    else:
        data=CustomData(
    Delivery_person_Age=int(request.form.get('delivery_person_age')),
    Delivery_person_Ratings=float(request.form.get('delivery_person_ratings')),
    Restaurant_latitude=float(request.form.get('restaurant_latitude')),
    Restaurant_longitude=float(request.form.get('restaurant_longitude')),
    Delivery_location_latitude=float(request.form.get('delivery_location_latitude')),
    Delivery_location_longitude=float(request.form.get('delivery_location_longitude')),
    Order_Date=request.form.get('order_date'),
    Time_Orderd=request.form.get('time_ordered'),
    Time_Order_picked	=request.form.get('time_picked'),
    Weather_conditions=request.form.get('weather_conditions'),
    Road_traffic_density=request.form.get('road_traffic_density'),
    Vehicle_condition=request.form.get('vehicle_condition'),
    Type_of_order=request.form.get('type_of_order'),
    Type_of_vehicle=request.form.get('type_of_vehicle'),
    multiple_deliveries=float(request.form.get('multiple_deliveries')),
    Festival=request.form.get('festival'),
    City=request.form.get('city')
        )


    final_new_data=data.get_data_as_dataframe()
    final_new_data.to_csv('new.csv')
    print(final_new_data)
    predict_pipeline=PredictPipeline()
    pred=predict_pipeline.predict(final_new_data)

    results=round(pred[0],2)
    

    return render_template('results.html',final_result=results)






if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)