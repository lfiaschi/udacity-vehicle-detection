from detectlib.features import get_samples, extract_features
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import time
from matplotlib import pyplot as plt
import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Define the paramters for the Features

options = dict(
    cspace='YCrCb',
    spatial=True,
    chist=True,
    hog=True,
    spatial_size=(32, 32),
    hist_bins=16,
    orient=9,
    pix_per_cell=8,
    cell_per_block=2,
    hog_channel='ALL',
)


# 1. Load the images
imgs, y = get_samples()

# 2. Extract features
X = extract_features(imgs, n_jobs=-1, **options)

scaler = StandardScaler()

# 3. Split / train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Scale
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print('Feature vector length:', len(X_train[0]))

#  5. Use a linear SVC
svc = LinearSVC(loss='hinge',dual=True, penalty='l2')

# 6. Optimize hyperparameters using cross validation
svc = GridSearchCV(svc,param_grid={'C': [100],
                                   'intercept_scaling': [100],
                                   # 'class_weight': [{0: 1, 1:2 }, {0:1, 1: 3}]
                                   }, n_jobs=None, verbose=10)

t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')

# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample

t=time.time()
n_predict = 10
print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: ', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
y_pred = svc.predict(X_test)
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test,y_pred)
plot_confusion_matrix(cm,['non-cars','cars'])

plt.savefig('output_images/confusion_matrix.png')
print(svc.best_estimator_)

# Final: Export trained model to file
import pickle

options['scaler'] = scaler
options['clf'] = svc.best_estimator_

with open('model.p', 'wb') as fh:
    pickle.dump(options,fh)