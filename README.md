# Neural_Network_Charity_Analysis
Neural Networks with python sklearn and tensorflow

## Overview:
In this challenge we are given the task to use Scikit learn and TensorFlow to create a binary classification using neural networks.

## Purpose:
The purpose of this challenge is to use another machine learning model (neural network) to create a binary classification. In this module we compared and went over different implementations of Machine Learning and compared them to implementing a neural network.         

## Resources
* Data Source: (charity_data.csv)
* Software: 
\ Jupyter Notebook 6.3

## Analysis:
### Overview of Analysis:
With my knowledge of machine learning and neural networks, I took the features given in the charity_data.csv to help create a binary classification that can predict whether applicants can be successful if funded by Alphabet Soup. I first had to preprocess the data by changing categorical data into numerical data using the sklearn OneHotEncoder() module. Before the encoding I also had to bin some of the values to reduce the amount of features we had. After binning and encoding, I merged the new data together with the original data frame then scaled or normalized the data. From this point on, I implemented by compiling, training, and evaluated the neural network model. 

Determine Categorical Columns to transform into numerical
![Get cat columns](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/categorical_columns.PNG) 
After finding the categorical columns, determine how many unique values in each column and try to reduce using the plot density graph
![plot density graph](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/densityforbinning.PNG) 
Once the cut off is set, start binning the unique values to reduce.
![binning](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/binning%20process.PNG) 
Once the binning is done, encode using OneHotEncoder() module from sklearn Library
![one-hot](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/one_hot_encoder.PNG) 
Once encoded, merge with original data frame and implement your neural network. 
![neural_network](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/neural_network_model.PNG) 

### Results:
The results of this challenge was time consuming and confusing. For this challenge we were tasked initially with using only 2 layers with 80 and 30 nodes in each layer. After running and evaluating this model, we were given an accuracy of: 72.6%. 
![non-op params](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/non-op_sum.PNG) 
![non-op Accuracy](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/non-op.PNG) 

We were then tasked to raise the accuracy score from 72% up to 75% or at least try 3 different optimizations. Here is a summary of the results:

Optimization 1- Layers: 3   Nodes: 200 100 50
![opt1 params](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt1_3lay_sum.PNG)
![opt1 accuracy](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt1_3lay_200-100-50.PNG) 
We can clearly see that even by adding another layer from 2 to 3 and more nodes we are not improving our accuracy

Optimization 2- Layers: 4   Nodes: 200 150 100 50
![opt2 params](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt2_4lay_sum.PNG)
![opt2 accuracy](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt2_4lay_200-150-100-50.PNG) 
Similarly, to the 3 layers, I was trying to implement more layers in hope that the accuracy would increase since I ended up decreasing the nodes from optimization one to 3 layers with the node counts 20 10 5 and still had an indifferent accuracy. 

Optimization 3- Layers: 3   Nodes: 150 80 40 EPOCH: 150
![opt3 params](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt3_3lay_sum.PNG)
![opt3 accuracy](https://github.com/lo7kyle/Neural_Network_Charity_Analysis/blob/main/Resources/opt3_3lay_150-80-40.PNG) 
The idea here was to decrease the number of nodes to a smaller number and remove the fourth layer since I really didn't do anything. This again, didn't change our accuracy. I also increased the EPOCH by 20 but it didn't change anything. 

## Summary/Conclusion: 
In summary it seems like adding more layers or more nodes really don't affect the accuracy of the data. Of course, I could've ran the model with 500 nodes and 6 layers, but I feel like the 4-layer model when sharing similar accuracy to that of the 2 dissuaded me from creating an insanely complex model. My conclusion from the results is that the issue can only be solved by changing our data. Maybe I could bin the unique values more spread out or I can possibly not bin at all. Another conclusion can be that because we don't have that many features (44 were used in this challenge), the model had a harder time to increase its accuracy no matter how many layers or nodes we had. These are my only assumptions, and I can only prove it otherwise by changing my data. Lastly, we can always just use a different supervised machine learning model. In this case, we can implement a logistic regression since we are doing a simple binary classification or SVM.
