from datetime import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from time import time
from sklearn import metrics
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow import keras
import data_processor
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("expand_frame_repr", False)
np.set_printoptions(threshold=np.inf)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# Pearson Correlation of Features
data_processor.y_all = data_processor.y_all.map({"NH": 0, "H": 1})
train_data = pd.concat([data_processor.X_all, data_processor.y_all], axis=1)
colormap = plt.cm.RdBu
plt.figure(figsize=(21, 18))
plt.title("Pearson Correlation of Features", y=1.05, size=15)
sns.heatmap(
    train_data.astype(float).corr(),
    linewidths=0.1,
    vmax=1.0,
    square=True,
    cmap=colormap,
    linecolor="white",
    annot=False,
)
# plt.show()

# data_processor.X_all = data_processor.X_all.drop(["HTP", "ATP"], axis=1)

# FTR correlation matrix
plt.figure(figsize=(14, 12))
k = 10  # number of variables for heatmap
cols = abs(train_data.astype(float).corr()).nlargest(k, "FTR")["FTR"].index
cm = np.corrcoef(train_data[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)
# plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    data_processor.X_all,
    data_processor.y_all,
    test_size=0.2,
    random_state=42,
    stratify=data_processor.y_all,
)


def train_classifier(clf, X_train, y_train):
    # get training time
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    print("training time {:.4f} s".format(end - start))


def predict_labels(clf, features, target):
    # get predicting time
    start = time()
    y_pred = clf.predict(features)
    end = time()
    print("predicting time in {:.4f} s".format(end - start))
    return f1_score(target, y_pred, pos_label=1), np.sum(target == y_pred) / float(
        len(y_pred)
    )


def train_predict(clf, X_train, y_train, X_test, y_test):
    # Indicate the classifier and the training set size
    print(
        "train {} model，sample number {}。".format(clf.__class__.__name__, len(X_train))
    )
    train_classifier(clf, X_train, y_train)
    f1, acc = predict_labels(clf, X_train, y_train)
    print("F1 score and accuracy on training set: {:.4f} , {:.4f}".format(f1, acc))
    f1, acc = predict_labels(clf, X_test, y_test)
    print("F1 score and accuracy on testing set: {:.4f} , {:.4f}".format(f1, acc))


clf_A = LogisticRegression(random_state=42)
clf_B = SVC(random_state=42, kernel="rbf", gamma="auto")
clf_C = xgb.XGBClassifier(seed=20)

train_predict(clf_A, X_train, y_train, X_test, y_test)
print("")
train_predict(clf_B, X_train, y_train, X_test, y_test)
print("")
train_predict(clf_C, X_train, y_train, X_test, y_test)
print("")

# neural network
model = keras.Sequential(
    [
        keras.layers.Input(shape=(22,), name="input"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
# print the model summary
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("train {} model，sample number {}。".format(model.__class__.__name__, len(X_train)))

# Set up filepath for model
model_file_path = os.path.abspath(os.path.join("data", "model"))


# Helper function
def get_run_logdir():
    root_logdir = os.path.join(os.curdir, "data", "logs")
    run_id = datetime.now().strftime('run_%Y_%m_%d-%H_%M_%S')
    run_log_dir = os.path.join(root_logdir, run_id)
    return run_log_dir


# Get a file path to where to store log data
run_logdir = get_run_logdir()

# Set up a checkpoint that saves the best model parameters so far during training
checkpoint_cb = keras.callbacks.ModelCheckpoint(model_file_path)

# Set up a callback for early stopping (which stops the training when the model stops becoming better)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)

# Set up a callback for tensorboard which can be used for visualisation
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

start = time()
history = model.fit(X_train, y_train,
                    epochs=500,
                    validation_split=0.2,
                    callbacks=[checkpoint_cb, early_stopping_cb, tensorboard_cb])
end = time()
print("training time {:.4f} s".format(end - start))

# Roll back the best model
model = keras.models.load_model(model_file_path)

# Predictions for training set
y_train_predict_temp = model.predict(X_train)
y_train_predict = []
for i in range(len(y_train_predict_temp)):
    if y_train_predict_temp[i] >= 0.5:
        y_train_predict.append(1)
    else:
        y_train_predict.append(0)

# Predictions for test set
y_test_predict_temp = model.predict(X_test)
y_test_predict = []
for i in range(len(y_test_predict_temp)):
    if y_test_predict_temp[i] >= 0.5:
        y_test_predict.append(1)
    else:
        y_test_predict.append(0)

# Metrics of the model
f1_score_train = metrics.f1_score(y_train, y_train_predict).round(2)
f1_score_test = metrics.f1_score(y_test, y_test_predict).round(2)
acc_train = metrics.accuracy_score(y_train, y_train_predict).round(2)
acc_test = metrics.accuracy_score(y_test, y_test_predict).round(2)

print("F1 score and accuracy on training set: {:.4f} , {:.4f}".format(f1_score_train, acc_train))
print("F1 score and accuracy on testing set: {:.4f} , {:.4f}".format(f1_score_test, acc_test))
