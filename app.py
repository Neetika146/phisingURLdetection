# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import plotly.graph_objects as go
import seaborn as sns # data visualization
%matplotlib inline
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from keras.models import Sequential
from keras import Input
from keras.layers import Dense
import random
##from google.colab import drive
drive.mount('/content/drive') 
data= pd.read_csv("phishing.csv")
data
data.head()
data.tail()
data.shape
data.info()
df = data.drop(['length_hostname','ratio_intErrors','ratio_extErrors','empty_title','dns_record','domain_age','domain_registration_length','domain_with_copyright','right_clic','whois_registered_domain','onmouseover','domain_in_title','ratio_extMedia','sfh','ratio_intMedia','iframe','submit_email','login_form','external_favicon','links_in_tags','domain_in_brand','brand_in_subdomain','brand_in_path','suspecious_tld','statistical_report','nb_hyperlinks','ratio_intHyperlinks','ratio_extHyperlinks','ratio_nullHyperlinks','nb_extCSS','ratio_intRedirection','ratio_extRedirection','nb_dots','nb_hyphens','nb_at','nb_qm','nb_and','nb_or','nb_eq','nb_underscore','nb_tilde','nb_percent','nb_slash','nb_star','nb_colon','nb_comma','nb_semicolumn','nb_dollar','nb_space','nb_www','nb_com','nb_dslash','http_in_path','https_token','ratio_digits_url','ratio_digits_host','punycode','port','tld_in_path','tld_in_subdomain','abnormal_subdomain','nb_subdomains','prefix_suffix','random_domain','shortening_service','path_extension','nb_redirection','nb_external_redirection','length_words_raw','char_repeat','shortest_words_raw','shortest_word_host','shortest_word_path','longest_words_raw','longest_word_host','longest_word_path','avg_words_raw','avg_word_host','avg_word_path','phish_hints'],axis =1)
df.shape
df
df.describe()
df.isnull().sum()
df.status.value_counts()
sns.countplot(x="status",palette = "Reds", data = df)
plt.title('Count of URLs')
# distribution of ranking of page
chart = go.Figure(data = [go.Pie(labels = df['page_rank'].value_counts().index, values = df['page_rank'].value_counts(), hole =0)])
chart.update_layout(legend_title_text ='Page Rank')
chart.show()
sns.set(style ='whitegrid', color_codes = True)
sns.catplot(x='status', y ='web_traffic',height = 4, palette = 'PuOr', aspect = 1, kind ='bar',data = df)
plt.title('Engagement')
#describing each feature
df.describe()[1:].T.style.background_gradient(cmap = "Oranges")
plt.figure(figsize=(12.5,10.5))
sns.heatmap(df.corr(), annot = True, cmap = "Greens")
plt.show()
df_shuffled = shuffle(df, random_state= 42)
data_size = 8000
df_used = df_shuffled[:data_size].copy()
sns.catplot(x="status",height=4.5, aspect=1.5,palette = "coolwarm", kind="count",data=df_used)
plt.title('Count of URLs after shuffling')
df_used.info()
df_used.replace({'legitimate':0, 'phishing':1}, inplace=True)
df_used.status.value_counts()
# importing the required packages
from urllib.parse import urlparse,urlencode
import ipaddress
import re
# 1.Domain of the URL (Domain) 
def getDomain(url):  
  domain = urlparse(url).netloc
  if re.match(r"^www.",domain):
	       domain = domain.replace("www.","")
  return domain
# 2.Checks for IP address in URL (ip)
def havingIP(url):
  try:
    ipaddress.ip_address(url)
    ip = 1
  except:
    ip = 0
  return ip
# 3.Checks the presence of @ in URL (nb_at)
def haveAtSign(url):
  if "@" in url:
    at = 1    
  else:
    at = 0    
  return at
# 4.Checks the presence of . in URL (nb_dots)
def havedot(url):
  if "." in url:
    at = 1    
  else:
    at = 0    
  return at
# 5.Finding the length of URL and categorizing (length_url)
def getLength(url):
  if len(url) < 54:
    length = 0            
  else:
    length = 1            
  return length
# 6.Gives number of '/' in URL (nb_slash)
def getDepth(url):
  s = urlparse(url).path.split('/')
  depth = 0
  for j in range(len(s)):
    if len(s[j]) != 0:
      depth = depth+1
  return depth
# 7.Checking for Prefix or Suffix Separated by (-) in the Domain (Prefix/Suffix)
def prefixSuffix(url):
    if '-' in urlparse(url).netloc:
        return 1            # phishing
    else:
        return 0            # legitimate
X = df_used[['url']].copy()
y = df_used.status.copy()
tokenizer = RegexpTokenizer(r'[A-Za-z]+')
stemmer = SnowballStemmer("english")
cv = CountVectorizer()
def prepare_data(X) :
    X['text_tokenized'] = X.url.map(lambda t: tokenizer.tokenize(t))
    X['text_stemmed'] = X.text_tokenized.map(lambda t: [stemmer.stem(word) for word in t])
    X['text_sent'] = X.text_stemmed.map(lambda t: ' '.join(t))
    features = cv.fit_transform(X.text_sent)
    return X, features
X, features = prepare_data(X)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
logreg = LogisticRegression()
knn = KNeighborsClassifier()
dtree = DecisionTreeClassifier()
rfc = RandomForestClassifier()
svc = SVC()
def train_test_model(model, X, y, training_percentage) :
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=1-training_percentage, stratify=y, random_state=42)
    model.fit(trainX, trainY)
    predY = model.predict(testX)
    accuracy = accuracy_score(testY, predY)
    precision = precision_score(testY, predY, pos_label=1)
    recall = recall_score(testY, predY, pos_label=1)
    return accuracy, precision, recall  
training_sizes = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
def model_results(model) :
    results = []
    for p in training_sizes :
        results.append(train_test_model(model, features, y, p))
    return pd.DataFrame(results, columns=['Accuracy', 'Precision', 'Recall'])
logreg_results = model_results(logreg)
knn_results = model_results(knn)
dtree_results = model_results(dtree)
rfc_results = model_results(rfc)
svc_results = model_results(svc)
models = ['Logistic Regression', 'KNN', 'Decision Tree', 'Random Forest', 'SVM']
model_results = [logreg_results, knn_results, dtree_results, rfc_results, svc_results]
accuracies = []
precisions = []
recalls = []
for model in model_results :
    accuracies.append(model.Accuracy.values)
    precisions.append(model.Precision.values)
    recalls.append(model.Recall.values)
accuracies = pd.DataFrame(np.transpose(accuracies), columns=models, index=training_sizes*100)
precisions = pd.DataFrame(np.transpose(precisions), columns=models, index=training_sizes*100)
recalls = pd.DataFrame(np.transpose(recalls), columns=models, index=training_sizes*100)
accuracies
precisions
recalls
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = accuracies, markers= ['o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0.6,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0.6, 1, 0.05))
g.set_title("Accuracy vs Training Percentage for the Machine Learning Algorithms")
g.set_xlabel("Training Percentage")
g.set_ylabel("Accuracy")
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = precisions, markers= ['o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0.4,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0.4, 1, 0.05))
g.set_title("Precision vs Training Percentage for the Machine Learning Algorithms")
g.set_xlabel("Training Percentage")
g.set_ylabel("Precision")
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = recalls, markers= ['o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0, 1, 0.05))
g.set_title("Recall vs Training Percentage for the Machine Learning Algorithms")
g.set_xlabel("Training Percentage")
g.set_ylabel("Recall")
hidden_units = [2, 4, 6, 8, 10, 12, 14, 16, 18]
def train_test_nn(X, y, training_percentage, hidden_units) :
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=1-training_percentage, stratify=y, random_state=42)
    trainX = trainX.toarray()
    testX = testX.toarray()
    trainY = np.array(trainY)
    testY = np.array(testY)
    model = Sequential()
    model.add(Input(shape=(trainX.shape[1], ), name='Input-Layer'))
    model.add(Dense(hidden_units, activation='relu', name='Hidden-Layer'))
    model.add(Dense(1, activation='sigmoid', name='Output-Layer'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])
    model.fit(trainX, trainY, batch_size = 10, epochs = 10, verbose=0)
    score = model.evaluate(testX, testY, batch_size = 1, verbose=2)
    return score
accuracies_nn = []
precisions_nn = []
recalls_nn = []
for ts in training_sizes :
    a = []
    p = []
    r = []
    for hn in hidden_units :
        s = train_test_nn(features, y, ts, hn)
        a.append(s[1])
        p.append(s[2])
        r.append(s[3])
    accuracies_nn.append(a)
    precisions_nn.append(p)
    recalls_nn.append(r)
accuracies_nn_df = pd.DataFrame(accuracies_nn, columns=hidden_units, index=training_sizes*100)
precisions_nn_df = pd.DataFrame(precisions_nn, columns=hidden_units, index=training_sizes*100)
recalls_nn_df = pd.DataFrame(recalls_nn, columns=hidden_units, index=training_sizes*100)
accuracies_nn_df
precisions_nn_df
recalls_nn_df
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = accuracies_nn_df, markers= ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0.65,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0.65, 1, 0.05))
g.set_title("Accuracy vs Training Percentage for the Feed Forward Neural Network")
g.set_xlabel("Training Percentage")
g.set_ylabel("Accuracy")
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = precisions_nn_df, markers= ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0, 1, 0.05))
g.set_title("Precision vs Training Percentage for the Feed Forward Neural Network")
g.set_xlabel("Training Percentage")
g.set_ylabel("Precision")
sns.set(rc={'figure.figsize':(15,8)})
sns.set_style('whitegrid')
g = sns.lineplot(data = recalls_nn_df, markers= ['o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o'])
g.set(xlim = (0,100), ylim = (0,1), xticks = np.arange(0, 100, 10), yticks = np.arange(0, 1, 0.05))
g.set_title("Recall vs Training Percentage for the Feed Forward Neural Network")
g.set_xlabel("Training Percentage")
g.set_ylabel("Recall")

# #importing required libraries

# from flask import Flask, request, render_template
# import numpy as np
# import pandas as pd
# from sklearn import metrics 
# import warnings
# warnings.filterwarnings('ignore')
# from feature import generate_data_set
# # Gradient Boosting Classifier Model
# from sklearn.ensemble import GradientBoostingClassifier

# data = pd.read_csv("phishing.csv")
# #droping index column
# data = data.drop(['Index'],axis = 1)
# # Splitting the dataset into dependant and independant fetature

# X = data.drop(["class"],axis =1)
# y = data["class"]

# # instantiate the model
# gbc = GradientBoostingClassifier(max_depth=4,learning_rate=0.7)

# # fit the model 
# gbc.fit(X,y)

# app = Flask(__name__)


# @app.route("/")
# def index():
#     return render_template("index.html", xx= -1)


# @app.route("/predict", methods=["GET", "POST"])
# def predict():
#     if request.method == "POST":

#         url = request.form["url"]
#         x = np.array(generate_data_set(url)).reshape(1,30) 
#         y_pred =gbc.predict(x)[0]
#         #1 is safe       
#         #-1 is unsafe
#         y_pro_phishing = gbc.predict_proba(x)[0,0]
#         y_pro_non_phishing = gbc.predict_proba(x)[0,1]
#         # if(y_pred ==1 ):
#         pred = "It is {0:.2f} % safe to go ".format(y_pro_phishing*100)
#         return render_template('index.html',xx =round(y_pro_non_phishing,2),url=url )
#         # else:
#         #     pred = "It is {0:.2f} % unsafe to go ".format(y_pro_non_phishing*100)
#         #     return render_template('index.html',x =y_pro_non_phishing,url=url )
#     return render_template("index.html", xx =-1)


# if __name__ == "__main__":
#     app.run(debug=True)