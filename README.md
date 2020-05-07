# Movie Recommendation System

A recommendation system is information filtering system that provides us some data prediction based on user’s data associated with the 
item.Recommendation system are heavily utilised by Tech giants like Amazon to improve user experience. Single Domain recommendations are 
limited by the data present in that domain only.To further improve the accuracy of the system we can utilise the data from other domains 
as well . We can also collect social media data related to the user to obtain a virtual user profile and then give him suitable 
recommendations.Such techniques come under Cross domain recommendation system. A cross domain rss aims to generate or enhance 
recommendations in a target domain by exploiting knowledge from source domains. We implemented cross domain by using matrix factorization. 
Matrix factorization techniques are more effective because they allow us to discover the latent features underlying the interactions 
between users and items. Given that each user has rated some items in the system, we would like to predict how the user will rate the 
items that they have not yet rated, such that we can make recommendations to the users.


# Cross Domain Recommendation System
Basic recommendation System faces problems of cold start and data sparsity. For overcoming the shortcomings of the basic recommendation 
system, cross domain recommendations are used. In cross domain recommendation System, the information from other domains(source domain) is 
used to predict the user’s behavior in the target domain.

We have implemented both, the single domain recommendation system as well as the the cross domain recommendation system using 
collaborative filtering(memory based) and matrix factorization techniques and we have analyzed the accuracy of the recommendations in each 
case in terms of MAE(Mean Absolute Error), RMSD (Root Mean Squared Deviation), Precision and Recall measures.

