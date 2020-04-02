'''
Source codes for Python Machine Learning By Example 3rd Edition (Packt Publishing)
Chapter 3 Recognizing Faces with Support Vector Machine
Author: Yuxi (Hayden) Liu (yuxi.liu.ece@gmail.com)
'''


from sklearn.datasets import fetch_lfw_people


face_data = fetch_lfw_people(min_faces_per_person=80)

X = face_data.data
Y = face_data.target

print('Input data size :', X.shape)
print('Output data size :', Y.shape)
print('Label names:', face_data.target_names)

for i in range(5):
    print(f'Class {i} has {(Y == i).sum()} samples.')



import matplotlib.pyplot as plt

fig, ax = plt.subplots(3, 4)
for i, axi in enumerate(ax.flat):
    axi.imshow(face_data.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[],
            xlabel=face_data.target_names[face_data.target[i]])

plt.show()


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42)

from sklearn.svm import SVC
clf = SVC(class_weight='balanced', random_state=42)


from sklearn.model_selection import GridSearchCV
parameters = {'C': [0.1, 1, 10],
              'gamma': [1e-07, 1e-08, 1e-06],
              'kernel' : ['rbf', 'linear'] }

grid_search = GridSearchCV(clf, parameters, n_jobs=-1, cv=5)

grid_search.fit(X_train, Y_train)

print('The best model:\n', grid_search.best_params_)

print('The best averaged performance:', grid_search.best_score_)

clf_best = grid_search.best_estimator_

print(f'The accuracy is: {clf_best.score(X_test, Y_test)*100:.1f}%')

pred = clf_best.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(Y_test, pred, target_names=face_data.target_names))




from sklearn.decomposition import PCA
pca = PCA(n_components=100, whiten=True, random_state=42)
svc = SVC(class_weight='balanced', kernel='rbf', random_state=42)

from sklearn.pipeline import Pipeline
model = Pipeline([('pca', pca),
                  ('svc', svc)])

parameters_pipeline = {'svc__C': [1, 3, 10],
                       'svc__gamma': [0.001, 0.005]}
grid_search = GridSearchCV(model, parameters_pipeline)

grid_search.fit(X_train, Y_train)

print('The best model:\n', grid_search.best_params_)
print('The best averaged performance:', grid_search.best_score_)

model_best = grid_search.best_estimator_
print(f'The accuracy is: {model_best.score(X_test, Y_test)*100:.1f}%')
pred = model_best.predict(X_test)
print(classification_report(Y_test, pred, target_names=face_data.target_names))