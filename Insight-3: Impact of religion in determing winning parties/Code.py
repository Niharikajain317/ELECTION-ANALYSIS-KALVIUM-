from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
winning_party_encoded = one_hot_encoder.fit_transform(data[['Winning Party']]).toarray()
X = data[['Hindus (%)', 'Muslims (%)', 'Christians (%)', 'Other (%)', 'Total Population (millions)']]
y = winning_party_encoded
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
importances = clf.feature_importances_
feature_names = X.columns
feature_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importances = feature_importances.sort_values(by='Importance', ascending=False)
feature_importances
