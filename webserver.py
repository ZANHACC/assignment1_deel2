from flask import Flask,render_template

app = Flask(__name__)

@app.route("/ram_pred")
def home1():
    import Helper
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #from numpy import loadtxt
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib.patches import Rectangle
    #from sklearn import metrics
    #from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score
    #range bepalen
    #normalisatie
    ramposition_train = pd.read_csv('features_1_1.csv', header = None)#hier lees je de X dit is de rampostion
    ramposition_test = pd.read_csv('features_1_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    Y = Helper.Y#dit is men persoonlijke data

    #nu gaan we de tabellen filteren van ramposition
    ramposition_train.drop(ramposition_train.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_train.drop(ramposition_train.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_train.reset_index(drop=True)

    ramposition_test.drop(ramposition_test.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_test.drop(ramposition_test.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_test.reset_index(drop=True)

    #dit i het model dat we een DecisionTree gaan gebruiken
    model1 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen

    #dit is voor de ramposition
    model1 = model1.fit(ramposition_train, Y)#hier trainen we het model
    ram_pred = model1.predict(ramposition_test)
    #print("dit is de ramposition predictions:",ram_pred)#dit printen we
    training_predictions_ram  = model1.predict(ramposition_train)

    plt.subplot(2, 2, 1)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(ram_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('ramposition')



    #injection_pressure
    injection_pressure_train = pd.read_csv('features_2_1.csv', header = None)#hier lees je de X dit is de rampostion
    injection_pressure_test = pd.read_csv('features_2_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #nu gaan we de tabellen filteren van injection_pressure
    injection_pressure_train.drop(injection_pressure_train.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_train.reset_index(drop=True)

    injection_pressure_test.drop(injection_pressure_test.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_test.reset_index(drop=True)


    #dit is voor de injection_pressure
    model2 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model2 = model2.fit(injection_pressure_train, Y)#hier trainen we het model
    inject_pred = model2.predict(injection_pressure_test)
    #print("dit is de injection pressure predictions:",inject_pred)#dit printen we

    training_predictions_injection  = model2.predict(injection_pressure_train)

    plt.subplot(2, 2, 2)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(inject_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('injection_pressure')


    #sensor_pressure
    sensor_pressure_train = pd.read_csv('features_3_1.csv', header = None)#hier lees je de X dit is de rampostion
    sensor_pressure_test = pd.read_csv('features_3_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #de tabellen van sensor_pressure moet niet gefiltert worden deze zijn allemaal anders

    #dit is voor de sensor_pressure
    model3 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model3 = model3.fit(sensor_pressure_train, Y)#hier trainen we het model
    sensor_pred = model3.predict(sensor_pressure_test)
    #print("dit is de sensor pressure predictions:",sensor_pred)#dit printen we

    training_predictions_pressure = model3.predict(sensor_pressure_train)

    plt.subplot(2, 2, 3)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('sensor_pressure')

    #dit is voor het totaal
    X_train = np.hstack((ramposition_train, injection_pressure_train, sensor_pressure_train))
    #print(X_train)
    X_test = np.hstack((ramposition_test , injection_pressure_test, sensor_pressure_test))

    model4 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model4 = model4.fit(X_train, Y)#hier trainen we het model
    sensor_pred = model4.predict(X_test)

    training_predictions_total = model4.predict(X_train)

    plt.subplot(2, 2, 4)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Total training data", "Total predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label') 
    plt.ylabel('Probability')   
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('Total')

    plt.subplots_adjust(left=0.1,#hoeveel plaats er links tussen zit
                      bottom=0.1, #hoeveel plaats er tussen de onderkant is
                      right=0.9, #hoeveel plaats er rechts tussen zit
                      top=0.9, #hoeveel plaats er tussen de top zit
                      wspace=0.4, #plaats tussen de diagrammen
                      hspace=0.6)#dit zorgt voor de afstand tussen de onderste en bovenste diagram(hier tussen label en injection_pressure)



    return str(ram_pred)#hier print je welke prediction je wilt hebben


@app.route("/inject_pred")
def home2():
    import Helper
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #from numpy import loadtxt
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib.patches import Rectangle
    #from sklearn import metrics
    #from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score
    #range bepalen
    #normalisatie
    ramposition_train = pd.read_csv('features_1_1.csv', header = None)#hier lees je de X dit is de rampostion
    ramposition_test = pd.read_csv('features_1_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    Y = Helper.Y#dit is men persoonlijke data

    #nu gaan we de tabellen filteren van ramposition
    ramposition_train.drop(ramposition_train.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_train.drop(ramposition_train.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_train.reset_index(drop=True)

    ramposition_test.drop(ramposition_test.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_test.drop(ramposition_test.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_test.reset_index(drop=True)

    #dit i het model dat we een DecisionTree gaan gebruiken
    model1 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen

    #dit is voor de ramposition
    model1 = model1.fit(ramposition_train, Y)#hier trainen we het model
    ram_pred = model1.predict(ramposition_test)
    #print("dit is de ramposition predictions:",ram_pred)#dit printen we
    training_predictions_ram  = model1.predict(ramposition_train)

    plt.subplot(2, 2, 1)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(ram_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('ramposition')



    #injection_pressure
    injection_pressure_train = pd.read_csv('features_2_1.csv', header = None)#hier lees je de X dit is de rampostion
    injection_pressure_test = pd.read_csv('features_2_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #nu gaan we de tabellen filteren van injection_pressure
    injection_pressure_train.drop(injection_pressure_train.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_train.reset_index(drop=True)

    injection_pressure_test.drop(injection_pressure_test.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_test.reset_index(drop=True)


    #dit is voor de injection_pressure
    model2 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model2 = model2.fit(injection_pressure_train, Y)#hier trainen we het model
    inject_pred = model2.predict(injection_pressure_test)
    #print("dit is de injection pressure predictions:",inject_pred)#dit printen we

    training_predictions_injection  = model2.predict(injection_pressure_train)

    plt.subplot(2, 2, 2)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(inject_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('injection_pressure')


    #sensor_pressure
    sensor_pressure_train = pd.read_csv('features_3_1.csv', header = None)#hier lees je de X dit is de rampostion
    sensor_pressure_test = pd.read_csv('features_3_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #de tabellen van sensor_pressure moet niet gefiltert worden deze zijn allemaal anders

    #dit is voor de sensor_pressure
    model3 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model3 = model3.fit(sensor_pressure_train, Y)#hier trainen we het model
    sensor_pred = model3.predict(sensor_pressure_test)
    #print("dit is de sensor pressure predictions:",sensor_pred)#dit printen we

    training_predictions_pressure = model3.predict(sensor_pressure_train)

    plt.subplot(2, 2, 3)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('sensor_pressure')

    #dit is voor het totaal
    X_train = np.hstack((ramposition_train, injection_pressure_train, sensor_pressure_train))
    #print(X_train)
    X_test = np.hstack((ramposition_test , injection_pressure_test, sensor_pressure_test))

    model4 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model4 = model4.fit(X_train, Y)#hier trainen we het model
    sensor_pred = model4.predict(X_test)

    training_predictions_total = model4.predict(X_train)

    plt.subplot(2, 2, 4)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Total training data", "Total predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label') 
    plt.ylabel('Probability')   
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('Total')

    plt.subplots_adjust(left=0.1,#hoeveel plaats er links tussen zit
                      bottom=0.1, #hoeveel plaats er tussen de onderkant is
                      right=0.9, #hoeveel plaats er rechts tussen zit
                      top=0.9, #hoeveel plaats er tussen de top zit
                      wspace=0.4, #plaats tussen de diagrammen
                      hspace=0.6)#dit zorgt voor de afstand tussen de onderste en bovenste diagram(hier tussen label en injection_pressure)



    return str(inject_pred)#hier print je welke prediction je wilt hebben




@app.route("/sensor_pred")
def home3():
    import Helper
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    #from numpy import loadtxt
    from sklearn.tree import DecisionTreeClassifier
    from matplotlib.patches import Rectangle
    #from sklearn import metrics
    #from sklearn.metrics import confusion_matrix 
    from sklearn.metrics import accuracy_score
    #range bepalen
    #normalisatie
    ramposition_train = pd.read_csv('features_1_1.csv', header = None)#hier lees je de X dit is de rampostion
    ramposition_test = pd.read_csv('features_1_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    Y = Helper.Y#dit is men persoonlijke data

    #nu gaan we de tabellen filteren van ramposition
    ramposition_train.drop(ramposition_train.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_train.drop(ramposition_train.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_train.reset_index(drop=True)

    ramposition_test.drop(ramposition_test.columns[[12, 20]], inplace=True, axis=1)#deze tabels zijn altijd hetzelfde
    ramposition_test.drop(ramposition_test.columns[1:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    ramposition_test.reset_index(drop=True)

    #dit i het model dat we een DecisionTree gaan gebruiken
    model1 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen

    #dit is voor de ramposition
    model1 = model1.fit(ramposition_train, Y)#hier trainen we het model
    ram_pred = model1.predict(ramposition_test)
    #print("dit is de ramposition predictions:",ram_pred)#dit printen we
    training_predictions_ram  = model1.predict(ramposition_train)

    plt.subplot(2, 2, 1)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(ram_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('ramposition')



    #injection_pressure
    injection_pressure_train = pd.read_csv('features_2_1.csv', header = None)#hier lees je de X dit is de rampostion
    injection_pressure_test = pd.read_csv('features_2_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #nu gaan we de tabellen filteren van injection_pressure
    injection_pressure_train.drop(injection_pressure_train.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_train.reset_index(drop=True)

    injection_pressure_test.drop(injection_pressure_test.columns[5:9], inplace=True, axis=1)#dit zijn de colommen die 0 zijn die ik weg heb gedaan
    injection_pressure_test.reset_index(drop=True)


    #dit is voor de injection_pressure
    model2 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model2 = model2.fit(injection_pressure_train, Y)#hier trainen we het model
    inject_pred = model2.predict(injection_pressure_test)
    #print("dit is de injection pressure predictions:",inject_pred)#dit printen we

    training_predictions_injection  = model2.predict(injection_pressure_train)

    plt.subplot(2, 2, 2)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(inject_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('injection_pressure')


    #sensor_pressure
    sensor_pressure_train = pd.read_csv('features_3_1.csv', header = None)#hier lees je de X dit is de rampostion
    sensor_pressure_test = pd.read_csv('features_3_2.csv', header = None)#hier lees je de X dit is de rampostion_test

    #de tabellen van sensor_pressure moet niet gefiltert worden deze zijn allemaal anders

    #dit is voor de sensor_pressure
    model3 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model3 = model3.fit(sensor_pressure_train, Y)#hier trainen we het model
    sensor_pred = model3.predict(sensor_pressure_test)
    #print("dit is de sensor pressure predictions:",sensor_pred)#dit printen we

    training_predictions_pressure = model3.predict(sensor_pressure_train)

    plt.subplot(2, 2, 3)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Training data", "Predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label')
    plt.ylabel('Probability')
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('sensor_pressure')

    #dit is voor het totaal
    X_train = np.hstack((ramposition_train, injection_pressure_train, sensor_pressure_train))
    #print(X_train)
    X_test = np.hstack((ramposition_test , injection_pressure_test, sensor_pressure_test))

    model4 = DecisionTreeClassifier(max_depth=10,random_state=42, max_features= 10, min_samples_leaf=2)#max_depth dit is voor te bepalen hvl keer de boom mag splitsen
    model4 = model4.fit(X_train, Y)#hier trainen we het model
    sensor_pred = model4.predict(X_test)

    training_predictions_total = model4.predict(X_train)

    plt.subplot(2, 2, 4)
    colors = ["red", "yellow"]
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["Total training data", "Total predicted test data"]
    plt.legend(handles, labels, loc='upper left')#dit is de locatie van de legende


    plt.xlabel('label') 
    plt.ylabel('Probability')   
    plt.hist(Y, color="red")#dit is voor het histogram te tekennen 
    plt.hist(sensor_pred,color="yellow")#dit is voor het histogram te tekennen 
    plt.title('Total')

    plt.subplots_adjust(left=0.1,#hoeveel plaats er links tussen zit
                      bottom=0.1, #hoeveel plaats er tussen de onderkant is
                      right=0.9, #hoeveel plaats er rechts tussen zit
                      top=0.9, #hoeveel plaats er tussen de top zit
                      wspace=0.4, #plaats tussen de diagrammen
                      hspace=0.6)#dit zorgt voor de afstand tussen de onderste en bovenste diagram(hier tussen label en injection_pressure)



    return str(sensor_pred)#hier print je welke prediction je wilt hebben


@app.route("/geen")
def salvador():
    return "Nu zie je het model niet"
    
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
