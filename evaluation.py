from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def scale_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def train_and_evaluate_naive_bayes(data):
    print("Inside NB")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    nb = GaussianNB()
    model = nb.fit(X_train_scaled, y_train)
    nb_predict = nb.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, nb_predict)
    conf_matrix = confusion_matrix(y_test, nb_predict)

    return accuracy, conf_matrix

def train_and_evaluate_decision_tree(data):
    print("Inside DT")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    dt = DecisionTreeClassifier()
    model = dt.fit(X_train_scaled, y_train)
    dt_predict = dt.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, dt_predict)
    conf_matrix = confusion_matrix(y_test, dt_predict)

    return accuracy, conf_matrix

def train_and_evaluate_random_forest(data):
    print("Inside RF")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    rf = RandomForestClassifier()
    model = rf.fit(X_train_scaled, y_train)
    rf_predict = rf.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, rf_predict)
    conf_matrix = confusion_matrix(y_test, rf_predict)

    return accuracy, conf_matrix

def train_and_evaluate_knn(data):
    print("Inside KNN")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    knn = KNeighborsClassifier()
    model = knn.fit(X_train_scaled, y_train)
    knn_predict = knn.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, knn_predict)
    conf_matrix = confusion_matrix(y_test, knn_predict)

    return accuracy, conf_matrix

def train_and_evaluate_svc(data):
    print("Inside SVC")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    svc = SVC()
    model = svc.fit(X_train_scaled, y_train)
    svc_predict = svc.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, svc_predict)
    conf_matrix = confusion_matrix(y_test, svc_predict)

    return accuracy, conf_matrix
