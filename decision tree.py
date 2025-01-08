#read the file with pandas
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

df = pandas.read_csv("data3.csv")
print(df)

#Change string values into numerical values
d ={"USA":1, "UK":0, "N":2  }#Means convert the values 'UK' to 0, 'USA' to 1, and 'N' to 2.
df['Nationality']=df['Nationality'].map(d)
d= {"YES":1, "NO":0}
df["Go"] = df["Go"].map(d)
print(df)

#Then we have to separate the feature columns from the target column.
# The feature columns are the columns that we try to predict from, 
# and the target column is the column with the values we try to predict.
#X is the feature columns, y is the target column:

features = ["Age", "Experience","Rank", "Nationality"]
X =df[features]
y =df["Go"]

print(X)
print(y)

#Now let us create a decision tree
dtree = DecisionTreeClassifier()
dtree = dtree.fit(X,y)
tree.plot_tree(dtree, feature_names=features)
print(dtree.predict([[35, 4, 4, 1]]))
plt.title("Decision tree for which comedian's show to attend")
plt.show()