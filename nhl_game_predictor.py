# Recurrent Neural Network
# NHL Game Predictor

# Part 1 Data Preprocessing

#Import the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('NHL_Game_Stats_Train.csv')
training_set = dataset_train.loc[:, ['opponentTeamAbbrev','gameLocationCode','faceoffsLost','shotsAgainst','shootoutGamesWon','ppGoalsFor','shotsFor','otLosses','shootoutGamesLost','penaltyKillPctg','ppPctg','faceoffWinPctg','faceoffsWon','ppOpportunities','wins','goalsFor','losses','goalsAgainst','shNumTimes','ppGoalsAgainst','points']].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_1 = LabelEncoder()
training_set[:, 1] = labelencoder_1.fit_transform(training_set[:, 1])
labelencoder_2 = LabelEncoder()
training_set[:, 0] = labelencoder_2.fit_transform(training_set[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
training_set = onehotencoder.fit_transform(training_set).toarray()

# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

# Create a data structure with 60 game steps and one output
X_train = []
y_train = []
for i in range(20, 1056):
    X_train.append(training_set_scaled[i-20:i, :])
    y_train.append(training_set_scaled[i, len(training_set[0])-1])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Part 2 Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the training set
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Part 3 Making predictions and visualizing the results
X_test = []
X_test = X_train[1016:1036]
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))
predicted_win = regressor.predict(X_test)
predicted_win = sc.inverse_transform(predicted_win)

real_win = []
for i in training_set_scaled[1036:1056]:
    real_win.append(i[51])
# Visualizing the results
plt.plot(real_win, color = 'red', label = 'Actual Points')
plt.plot(predicted_win, color = 'blue', label = 'Predicted Points')
plt.title('Points Predictions')
plt.xlabel('Time')
plt.ylabel('Points')
plt.legend()
plt.show()

# New result for latest game
X_test = []
X_test.append(training_set_scaled[1036:1056, :])
X_test = np.array(X_test)

# Reshaping
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_win = regressor.predict(X_test)
predicted_win = sc.inverse_transform(predicted_win)