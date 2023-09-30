import pandas as pd

df = pd.read_csv('titanic.csv')

df.drop(['PassengerId','Name','Ticket','Cabin'],axis = 1, inplace=True)



df['Embarked'].fillna('S',inplace=True)


age_1 = df[df['Pclass'] ==1]['Age'].median()
age_2 = df[df['Pclass'] ==2]['Age'].median()
age_3 = df[df['Pclass'] ==3]['Age'].median()


def set_age(row):
    if pd.isnull(row['Age']):
        if row['Pclass'] ==1:
            return age_1
        elif row['Pclass'] ==2:
            return age_2
        elif row['Pclass'] ==3:
            return age_3
    return row['Age']


df['Age'] = df.apply(set_age,axis =1)

def set_sex (sex):
    if sex == 'male':
        return 1
    if sex == 'female':
        return 0

df['Sex']= df['Sex'].apply(set_sex)

df[list(pd.get_dummies(df['Embarked']).columns)] = pd.get_dummies(df['Embarked'])

df.drop('Embarked', axis = 1, inplace=True)

def is_alone(row):
    if row['SibSp'] + row['Parch'] == 0:
        return 1
    return 0

df['Alone'] = df.apply(is_alone, axis = 1)

print(df.pivot_table(values= 'Age', columns='Alone',
        index= 'Survived',aggfunc='count'))

#df.info()
#___________________________________________________________________________________________#scikit-learn


"""

aa = df[df['Parch'] > 0].Survived.value_counts()
bb = df[df['Parch'] == 0].Survived.value_counts()



print("З сім'ями")
print(aa)

print("Без сім'")
print(bb)
"""
"""
family = df[df['Parch'] > 0]
alone = df[df['Parch'] == 0]


Survived_family = family['Survived'].mean()
Survived_alone = alone['Survived'].mean()

if Survived_alone > Survived_family:
    print("Виживаність одиноких пасажирів більша.")
elif Survived_alone < Survived_family:
    print("Виживаність пасажирів з сім'єю більша.")

"""


#print(Survived_alone)
#print(Survived_family)

#________________________________________________________________________________________________________________

#Крок 0. Підключення потрібних модулів
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score 

# Крок 1. Розділяємо набір даних на тестування та навчання
X = df.drop('Survived',axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.25)


# Крок 2. Стандартизіція
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Крок 3. Стоворення об'єкту класифікатора KNN
classifiar = KNeighborsClassifier(n_neighbors = 3)



#Крок 4. Навчання моделі
classifiar.fit(X_train,y_train)



# Крок 5. Передбачення
y_pred = classifiar.predict(X_test)



#print(y_pred)


#Крок 6. Оцінка точності прогнозу

for p,t in zip(y_pred,y_test):
    print(f'p={p};t={t}')

percent = accuracy_score(y_test,y_pred) * 100 

print(f'Процент правильно передбачуваних результатів {percent}%')

print(confusion_matrix(y_test,y_pred))




