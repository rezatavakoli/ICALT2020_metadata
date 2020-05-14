#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% load the OERs file
df = pd.read_csv("Data-set.csv", encoding = 'utf-8')

# %% add some features
df["title_length"] = df["title"].apply(lambda x: x.count(" ") + 1)
df['title_availability'] = np.where(df['title_length']>0, 'available', 'unavailable')

df["description_length"] = df["description"].apply(lambda x: str(x).count(" ") + 1)
df['description_availability'] = np.where(df['description_length']>0, 'available', 'unavailable')

df["subjects_length"] = df["subjects"].apply(lambda x: x.count(",") + 1 if len(x) > 2 else 0 )
df['subjects_availability'] = np.where(df['subjects_length']>0, 'available', 'unavailable')

df["number_of_languages"] = df["languages"].apply(lambda x: str(x).count(",") + 1 if len(str(x)) > 2 else 0 )
df['language_availability'] = np.where(df['number_of_languages']>0, 'available', 'unavailable')

df["number_of_accessibilities"] = df["accessibilities"].apply(lambda x: str(x).count(",") + 1 if len(str(x)) > 2 else 0 )
df['accessibility_availability'] = np.where(df['number_of_accessibilities']>0, 'available', 'unavailable')

df_check_0 = df.loc[df.quality_control == "Without Control"]
df_check_1 = df.loc[df.quality_control == "With Control"]


# %% convert strings to numbers to calculate Availability Score and Norm Score
df = df.replace(to_replace ="available", value =1)
df = df.replace(to_replace ="unavailable", value =0)
df["availability_score"] = (df["title_availability"] * 0.17) + (df["description_availability"] * 0.17) + (df["subjects_availability"] * 0.145) + (df["level_availability"] * 0.165) + (df["time_required_availability"] * 0.098) + (df["language_availability"] * 0.155) + (df["accessibility_availability"] * 0.099) 
df["norm_score"] = ((1/np.ceil(np.absolute(df["title_length"]-5.5)/2.5)) * 0.17) + ((1/np.ceil(np.absolute(df["description_length"]-54.5)/40)) * 0.17) + ((1/np.ceil(np.absolute(df["subjects_length"]-4.5)/3.5)) * 0.145) + (df["level_availability"] * 0.165) + (df["time_required_availability"] * 0.098) + (df["language_availability"] * 0.155) + (df["accessibility_availability"] * 0.099)

# %% build Train set and Test set
X = df.loc[:,['title_length','description_length','subjects_length', 'level_availability','availability_score', 'norm_score']]
Y = df.loc[:,'quality_control']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# %% model with a Random Forest Classifier
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# %% evaluate the result
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

# %% get Features' Importance
feature_importances = pd.DataFrame(classifier.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance', ascending=False)



# %%
