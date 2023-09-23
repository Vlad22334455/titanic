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

df.info()
#___________________________________________________________________________________________#scikit-learn


"""

aa = df[df['Parch'] > 0].Survived.value_counts()
bb = df[df['Parch'] == 0].Survived.value_counts()



print("З сім'ями")
print(aa)

print("Без сім'")
print(bb)
"""

family = df[df['Parch'] > 0]
alone = df[df['Parch'] == 0]


Survived_family = family['Survived'].mean()
Survived_alone = alone['Survived'].mean()

if Survived_alone > Survived_family:
    print("Виживаність одиноких пасажирів більша.")
elif Survived_alone < Survived_family:
    print("Виживаність пасажирів з сім'єю більша.")




#print(Survived_alone)
#print(Survived_family)