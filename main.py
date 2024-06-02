import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from category_encoders import TargetEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

data = pd.read_csv('cardio_data.csv')
X = data.drop(["id","age","gender","height","weight"],axis=1)
#id is not necessary, age_years is best than age in days,heart disease is not relevant to gender ,height and weight is correlated to bmi
y = data["cardio"]

X_tr,X_te,y_tr,y_te = train_test_split(X,y,test_size=0.1)

convalues = ["ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","age_years","bmi"]
catvalues = ["bp_category"]
ct = ColumnTransformer(
[("categoricalEncoder", TargetEncoder(), catvalues ),
                       ('scaler', StandardScaler(), convalues)]
                       ,remainder="passthrough")

model = Pipeline(
    [("Columntransfer",ct),
     ("clf",MLPClassifier(max_iter=500,activation="tanh"))
])

model.fit(X_tr,y_tr)
yp = model.predict(X_te)

print("Accuracy: ",accuracy_score(y_te,yp))