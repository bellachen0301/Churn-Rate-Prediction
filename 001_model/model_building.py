import pickle

import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb

df = pd.read_csv("churn.csv")

# Generate features and target
num_cols =['points_in_wallet']
ordinal_cols = ['membership_category','feedback']
features = num_cols+ordinal_cols
target = 'churn_risk_score'

X = df[features]
y = df[target]

# Split DataFrame into train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                      y, 
                                                      test_size=0.2, 
                                                      random_state=2022)


# step1 pipeline
numeric_preprocessor = Pipeline(
    steps=[
        ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)


ordinal_categorical_preprocessor = Pipeline(
    steps=[
        (
            "imputation_constant",
            SimpleImputer(fill_value="missing", strategy="constant"),
        ),
        ("ordinal", OrdinalEncoder(categories=[['No Membership', 'Basic Membership', 'Silver Membership','Gold Membership', 'Premium Membership','Platinum Membership'],
                                            ['Poor Product Quality','Poor Customer Service','Poor Website','Too many ads','No reason specified','User Friendly Website',
                                            'Products always in Stock', 'Quality Customer Care','Reasonable Price'],
                                            ]
                                    ),
                                ),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("numerical", numeric_preprocessor, num_cols),
        ("ordinal_categorical", ordinal_categorical_preprocessor,ordinal_cols),
    ]
)

model = make_pipeline(preprocessor,
                    xgb.XGBClassifier(gamma= 1,learning_rate= 0.1,max_depth= 4,min_child_weight= 3,reg_lambda=1)
)
#step 2 fit 
model.fit(X_train, y_train)


with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
