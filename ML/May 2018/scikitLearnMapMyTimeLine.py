from sklearn import tree
import graphviz
x=[[913, 1539], [913, 1709], [905, 1705], [846, 1435], [920, 1716], [915, 1642]]
#monday 1 , tues 2 , wednesday 3 , thu 4 , friday 5
y=[2, 3, 4, 1, 2, 5]
classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(x,y)
print(classifier.predict([[923, 1308]]))
dot_data = tree.export_graphviz(classifier, out_file=None, 
                         feature_names=["in","out"],  
                         class_names=["mon","tue","wed","thr","fri"],  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = graphviz.Source(dot_data)
tree.export_graphviz(classifier,out_file='tree.dot')
graph