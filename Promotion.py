import streamlit as st
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn import over_sampling

df1=pd.read_csv('df1.csv')
df1.drop('Unnamed: 0',axis=1,inplace=True)

pro=LabelEncoder()
encpro1=pro.fit_transform(df1['department'])
df1['department'] = encpro1

pro1=LabelEncoder()
encpro2=pro1.fit_transform(df1['region'])
df1['region'] = encpro2

pro2=LabelEncoder()
encpro3=pro2.fit_transform(df1['education'])
df1['education'] = encpro3

pro3=LabelEncoder()
encpro4=pro3.fit_transform(df1['gender'])
df1['gender'] = encpro4

pro4=LabelEncoder()
encpro5=pro4.fit_transform(df1['recruitment_channel'])
df1['recruitment_channel'] = encpro5

X=df1.iloc[:,:11]
Y=df1.loc[:,df1.columns=='is_promoted']

X,Y= over_sampling.SMOTE().fit_resample(X,Y.values.ravel())

Y=pd.DataFrame(data=Y,columns=['is_promoted'])

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.10, random_state=42)

scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)


X_train=pd.DataFrame(X_train,columns=['department', 'region', 'education', 'gender', 'recruitment_channel',
       'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
       'awards_won', 'avg_training_score'])

X_train.reset_index(drop=True,inplace=True)
y_train.reset_index(drop=True,inplace=True)

entropy=DecisionTreeClassifier(criterion='entropy',random_state=0)
model=AdaBoostClassifier(base_estimator=entropy,n_estimators=20,random_state=42)
model.fit(X_train,y_train)


st.image("https://img.icons8.com/ios/50/FD7E14/google-ads.png", width=100)
st.title('Employee Promotion Prediction')

st.sidebar.header('user_input_features')
def user_input_features():
	department=st.sidebar.selectbox('Department of employee',('Sales & Marketing', 'Operations', 'Technology', 'Analytics','R&D', 'Procurement', 'Finance', 'HR', 'Legal'))
	region=st.sidebar.selectbox('Region of employment',('region_7', 'region_22', 'region_19', 'region_23', 'region_26','region_2', 'region_20', 'region_34', 'region_1', 'region_4','region_29', 'region_31', 'region_15', 'region_14', 'region_11',
       'region_5', 'region_28', 'region_17', 'region_13', 'region_16','region_25', 'region_10', 'region_27', 'region_30', 'region_12','region_21', 'region_8', 'region_32', 'region_6', 'region_33','region_24', 'region_3', 'region_9', 'region_18'))
	education=st.sidebar.selectbox('Education Level',("Master's & above", "Bachelor's",'Below Secondary'))
	gender=st.sidebar.selectbox('Gender',('f','m'))
	recruitment_channel=st.sidebar.selectbox('Recruitment_channel',('sourcing', 'other', 'referred'))
	no_of_trainings=st.sidebar.slider('Trainings',min_value=1,max_value=10)
	age=st.sidebar.number_input('Insert the Age')
	previous_year_rating=st.sidebar.radio('Rating',[1,2,3,4,5])
	length_of_service=st.sidebar.number_input('Length of service in years')
	awards_won=st.sidebar.radio('If awards won during the previous year then 1 else 0',[0,1])
	avg_training_score=st.sidebar.number_input('Average score in current training evaluations')
	data={'department':department,'region':region,'education':education,'gender':gender,'recruitment_channel':recruitment_channel,
	      'no_of_trainings':no_of_trainings,'age':age,'previous_year_rating':previous_year_rating,'length_of_service':length_of_service,
	      'awards_won':awards_won,'avg_training_score':avg_training_score}
	features = pd.DataFrame(data,index = [0])
	return features
df=user_input_features()
st.subheader('user_input_parameters')
st.write(df)

encpro=pro.transform(df['department'])
df['department'] = encpro

encpro=pro1.transform(df['region'])
df['region'] = encpro

encpro=pro2.transform(df['education'])
df['education'] = encpro

encpro=pro3.transform(df['gender'])
df['gender'] = encpro

encpro=pro4.transform(df['recruitment_channel'])
df['recruitment_channel'] = encpro

df=scaler.transform(df)
df=pd.DataFrame(df,columns=['department', 'region', 'education', 'gender', 'recruitment_channel',
       'no_of_trainings', 'age', 'previous_year_rating', 'length_of_service',
       'awards_won', 'avg_training_score'])


Prediction=model.predict(df)
prediction_proba=model.predict_proba(df)

st.subheader('Prediction Probability')
st.write(prediction_proba)


st.subheader('Prediction Result')
st.write('Yes' if prediction_proba[0][1]>0.5 else 'No')
st.balloons()

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image:url(https://images.unsplash.com/photo-1553095066-5014bc7b7f2d?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8d2FsbCUyMGJhY2tncm91bmR8ZW58MHx8MHx8&w=1000&q=80);
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
       )

add_bg_from_url() 

