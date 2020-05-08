# Movie Recommendation System

A recommendation system provides suggestions to the users through a filtering process that is based on user preferences and browsing 
history. The information about the user is taken as an input. The information is taken from the input that is in the form of browsing 
data. This information reflects the prior usage of the product as well as the assigned ratings. A recommendation system is a platform 
that provides its users with various contents based on their preferences and likings. A recommendation system takes the information 
about the user as an input. The recommendation system is an implementation of the machine learning algorithms. Recommendation system are 
heavily utilised by Tech giants like Amazon to improve user experience.

A recommendation system also finds a similarity between the different products. For example, Netflix Recommendation System provides you 
with the recommendations of the movies that are similar to the ones that have been watched in the past. Furthermore, there is a 
collaborative content filtering that provides you with the recommendations in respect with the other users who might have a similar 
viewing history or preferences.

Single Domain recommendations are limited by the data present in that domain only.To further improve the accuracy of the system we can 
utilise the data from other domains as well. We can also collect social media data related to the user to obtain a virtual user profile 
and then give him suitable recommendations. Such techniques come under Cross domain recommendation system.



# Cross Domain Recommendation System

Recommender systems provide users with personalized online product and service recommendations and are a ubiquitous part of today's 
online entertainment smorgasbord. However, many suffer from cold-start problems due to a lack of sufficient preference data, and this is 
hindering their development. Cross-domain recommender systems have been proposed as one possible solution. These systems transfer 
knowledge from one domain that has adequate preference information to another domain that does not. The outlook for cross-domain 
recommendation is promising, but existing methods cannot ensure the knowledge extracted from the source domain is consistent with the 
target domain, which may impact the accuracy of the recommendations.

We have implemented both, the single domain recommendation system as well as the the cross domain recommendation system using 
collaborative filtering(memory based) and matrix factorization techniques and we have analyzed the accuracy of the recommendations in 
each case in terms of MAE(Mean Absolute Error), RMSD (Root Mean Squared Deviation), Precision and Recall measures.



# Matrix Factorization

In its basic form, matrix factorization characterizes both items and users by vectors of factors inferred from item rating patterns.  
High correspondence between item and user factors leads to a recommendation. These methods have become popular in recent years by 
combining good scalability with predictive accuracy. In addition, they offer much flexibility for modeling various real-life situations. 

Matrix factorization techniques are more effective because they allow us to discover the latent features underlying the interactions 
between users and items. Given that each user has rated some items in the system, we would like to predict how the user will rate the 
items that they have not yet rated, such that we can make recommendations to the users.

One strength of matrix factorization is that it allows incorporation of additional information. When explicit feedback is not available, 
recommender systems can infer user preferences using implicit feedback, which indirectly reflects opinion by observing user behavior 
including purchase history, browsing history, search patterns, or even mouse movements. Implicit feedback usually denotes the presence 
or absence of an event, so it is typically represented by a densely filled matrix.



# Collaborative Filtering

Collaborative filtering (CF) is a technique commonly used to build personalized recommendations on the Web. Some popular websites that 
make use of the collaborative filtering technology include Amazon, Netflix, iTunes, IMDB, LastFM, Delicious and StumbleUpon. In 
collaborative filtering, algorithms are used to make automatic predictions about a user's interests by compiling preferences from 
several users. 

Different types of collaborative filtering are as follows:

Memory Based: This method makes use of user rating information to calculate the likeness between the users or items. This calculated 
                likeness is then used to make recommendations.

Model Based: Models are created by using data mining, and the system learns algorithms to look for habits according to training data. 
             These models are then used to come up with predictions for actual data.

Hybrid: Various programs combine the model-based and memory-based CF algorithms.
