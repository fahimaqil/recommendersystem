import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def recommend_books(predictions_df, user_id, rating, num_recommendations=5):
    
  # Get and sort the user's predictions

    rating.columns=["ID_USER","ID_HOTEL",'Trip Type',"Rating"]
    # print(rating.head)
    user_row_number = user_id - 1 # user_id starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)
    # Get the user's data and merge in the movie information.
    user_data =  rating[rating.ID_USER == (user_id)]
    user_data.columns=["ID_USER","ID_HOTEL","Trip Type","Rating"]
    print(sorted_user_predictions)
    # print(user_data.head)

    # print(sorted_user_predictions[:10])

    # user_full = (user_data.merge(rating, how = 'left', left_on = 'ID_HOTEL', right_on = 'ID_HOTEL').
    #                  sort_values(['Rating'], ascending=False)
    #              )
    
    # print ('User {0} has already rated {1} movies related to the context'.format(user_id, user_data.shape[0]))
    # print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))
    
    recommendations = (rating[~rating['ID_HOTEL'].isin(user_data['ID_HOTEL'])])
    recommendations=pd.merge(recommendations,sorted_user_predictions,on="ID_HOTEL")
    recommendations.columns=["ID_USER","ID_HOTEL","Trip Type","Rating","Predictions"]
    recommendations = recommendations.drop_duplicates(subset='ID_HOTEL', keep="first")
    recommendations = recommendations.sort_values('Predictions',ascending=False)
    recommendations = recommendations.drop(columns=['ID_USER','Rating',])
        #  merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
        #        left_on = 'ID_HOTEL',
        #        right_on = 'ID_HOTEL').
        #  rename(columns = {user_row_number: 'Predictions'}).
        #  sort_values('Predictions', ascending = False).
        #                iloc[:num_recommendations, :-1]
        #               )
    print(recommendations)
    return user_data, recommendations[:10],user_data
def createContextPD(contextType,data):
  data1=data.copy()
  data1 = data1[data1['Trip Type'] == contextType]
  data2=data.copy()
  data2=data2[data2['Trip Type'] != contextType]
  for _ in data2["Rating"]:
      data2["Rating"]=0
  return pd.concat([data1,data2],axis=0,join='outer')
def runRecommender(file1,file2,user_id,context):

    mainData = pd.read_json(file1)
    initData=pd.read_json(file2)
    print(mainData.columns)
        

    #mainData=mainData.drop(mainData.columns[0], axis=1)
    

    ##### Create another copy #####
    ##### Remove column #####
  

    mainData = mainData.drop_duplicates(['ID_USER','ID_HOTEL'])
    mainData=createContextPD(context,mainData)
    #pivot the data set
    ratingData = mainData.pivot(index = 'ID_USER', columns ='ID_HOTEL', values = 'Rating').fillna(0)
    
    R = ratingData.to_numpy()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = ratingData.columns)
    already_rated, predictions,rated = recommend_books(preds_df, user_id, mainData,10)
    #predictions=predictions.merge
    #print(predictions)
    predictions=predictions.merge(initData,on='ID_HOTEL')
    predictions = predictions.drop_duplicates(subset='ID_HOTEL', keep="first")
  
    #print(predictions[:10])
    predict=predictions.to_json(orient="records")
    rated=rated.to_json(orient="records")
    return predict,rated
def runRecommenderWithoutContext(file1,file2,user_id):

    mainData = pd.read_json(file1)
    initData=pd.read_json(file2)
        

    #mainData=mainData.drop(mainData.columns[0], axis=1)
    

    ##### Create another copy #####
    ##### Remove column #####
  

    mainData = mainData.drop_duplicates(['ID_USER','ID_HOTEL'])
    # mainData=createContextPD(context,mainData)
    #pivot the data set
    ratingData = mainData.pivot(index = 'ID_USER', columns ='ID_HOTEL', values = 'Rating').fillna(0)
    #trainset, testset = train_test_split(mainData, test_size=0.4)
  
    R = ratingData.to_numpy()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = ratingData.columns)
    already_rated, predictions,rated = recommend_books(preds_df, user_id, mainData,10)
    
    predictions=predictions.merge(initData,on='ID_HOTEL')
    rated=rated.merge(initData,on='ID_HOTEL')
    rated=rated.drop_duplicates(subset='ID_HOTEL', keep="first")
    predictions = predictions.drop_duplicates(subset='ID_HOTEL', keep="first")
  
    #print(predictions[:10])
    predict=predictions.to_json(orient="records")
    rated=rated.to_json(orient="records")
    return predict,rated
runRecommender('static/ratingHotel.json','static/hotel.json',1,3)

