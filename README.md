Yelp Recommender System
---------------------------------
Project Objective - To predict users preference for businesses using Yelp datasets and to build a recommendation system to predict the given (user, business) pairs by mining interesting and useful information using yelp datasets to support the recommendation system.

Methods Explored and Implemented - Item based, Model based and Weighted Hybrid Collaborative Filtering Algorithms for User-Business Recommendations  
Best Results - Model based Recommendation System
      
Dataset and Features - Yelp public dataset
Training data - yelp_train.csv - user_id, business_id, stars(rating)
Test data - yelp_val.csv - user_id, business_id, stars(rating)

business.json -
---------------------------------
review_count, 
stars, 
RestaurantsPriceRange2,
Caters, 
HasTV, 
GoodForKids, 
DogsAllowed, 
RestaurantsDelivery, 
BikeParking, 
OutdoorSeating, 
BusinessAcceptsCreditCards, 
RestaurantsGoodForGroups, 
RestaurantsReservations, 
RestaurantsTakeOut, 
bus_city(total count per city mapped to businesses), 
bus_state(total count per state mapped to businesses), 
bus_category(most appearing category of a business)

user.json - 
---------------------------------
review_count, 
average_stars, 
useful, 
funny, 
cool, 
fans, 
compliment_hot, 
compliment_more, 
compliment_profile, 
compliment_cute, 
compliment_list, 
compliment_note, 
compliment_plain, 
compliment_cool, 
compliment_funny, 
compliment_writer, 
compliment_photos, 
user_frnds(count of users friends linked to same business_id)

tip.json - 
---------------------------------
text sentiment(sentiment score of users to businesses)

photo.json -
--------------------------------
photo_count(count of photos per business_id)

checkin.json -
---------------------------------
checkin_count(count of checkins per business_id)

review_train.json -
---------------------------------
user_count(count of reviews of each user), 
bus_count(count of reviews of each business)

Merged these features with the training and validation data sets and used XGBoost model with hyperparameter tuning to predict the ratings(stars) of users to businesses.
  
**RMSE : 0.974**
