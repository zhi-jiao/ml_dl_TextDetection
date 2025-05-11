import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.decomposition import PCA,IncrementalPCA
from scipy.sparse import csr_matrix
from sklearnex import patch_sklearn
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier


from tqdm import tqdm
patch_sklearn()


data = pd.read_csv('./paraphrasing_data.csv')

def preprocess_text(input):
    input = eval(input)
    sentence = ' '.join(input)
    return sentence


features = data['text'].apply(preprocess_text)
label = np.array( data['label'].map({'human':1,'bot':0}))

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(features)
X = X.astype(np.float32).toarray()


pca = PCA(n_components=512)
# pca = IncrementalPCA(n_components=512,batch_size=1000)
reduced_data = pca.fit_transform(X)  


# 定义PCA模型
X = X[0:679000,:]
pca = PCA(n_components=512)
n_samples, n_features = X.shape


batch_size = 1000

reduced_data = np.empty((0, 512))

# batch PCA
for i in tqdm(range(0, n_samples, batch_size)):
    batch = X[i:i+batch_size]
    batch_csr = csr_matrix(batch)
    batch_array = batch_csr.toarray()
    batch_reduced = pca.fit_transform(batch_array)
    reduced_data = np.vstack((reduced_data, batch_reduced))
    
    
# 划分数据集
X_train,X_test,y_train,y_test = train_test_split(reduced_data,label,test_size= 0.3,random_state=3407)
X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5,random_state=3407)




models = [
    (
        'SVM',
        SVC(random_state=3407),
        {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    ),
    (
        'RandomForest',
        RandomForestClassifier(random_state=3407),
        {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5]
        }
    ),
    (
        'LogisticRegression',
        LogisticRegression(random_state=3407, max_iter=1000),
        {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    ),
    (
        'GaussianNB',
        GaussianNB(),
        {
            'var_smoothing': [1e-9, 1e-8, 1e-7]
        }
    ),
    (
        'KNN',
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    )
]


for name, model, param_grid in models:
    print(f"\n=== {name} Grid Search and Classification Report ===")

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,  
        scoring='accuracy',
        n_jobs=-1  
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Parameters: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_

    print("\nTest Set:")
    y_pred_test = best_model.predict(X_test)
    print(classification_report(y_test, y_pred_test))
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(f"Test Set Accuracy: {test_accuracy:.4f}")