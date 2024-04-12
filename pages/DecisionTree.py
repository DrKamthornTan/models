import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import streamlit as st

df = pd.read_csv("data.csv")

df.dropna(inplace=True)

# Define the columns to drop
columns_to_drop = ["caseid","death_inhosp","dx","icu_days","intraop_epi", "preop_ph", "preop_be" ]

# Drop the specified columns from the DataFrame
df = df.drop(columns_to_drop, axis=1)

d = {'General surgery': 0, 'Thoracic surgery': 1, 'Gynecology': 2, 'Urology': 3}
df['department'] = df['department'].map(d)

d = {'Colorectal': 0, 'Biliary/Pancreas': 1, 'Others': 2, 'Stomach': 3, 'Major resection': 4, 'Breast': 5, 'Transplantation': 6,
    'Vascular': 7, 'Hepatic': 8, 'Thyroid': 9
}
df['optype'] = df['optype'].map(d)

d = {'General': 0, 'Spinal': 1, 'Sedationalgesia': 2}
df['ane_type'] = df['ane_type'].map(d)

d = {'Normal Sinus Rhythm': 0, 
    '1st degree A-V block': 1, 'Right bundle branch block': 1, 'Premature ventricular complexes': 1, 'Atrial fibrillation': 1,
    'Left anterior fascicular block': 1, 'Atrial fibrillation with slow ventricular response': 1, 'Atrial fibrillation with premature ventricular or aberrantly conducted complexes': 1, 
    '1st degree A-V block with Premature atrial complexes': 1, 'Complete right bundle branch block, occasional premature supraventricular complexes': 1, 
    'Right bundle branch block, Left anterior fascicular block': 1, 'AV sequential or dual chamber electronic pacemaker': 1, 
    'Electronic ventricular pacemaker': 1, 'Atrial flutter with 2:1 A-V conduction': 1, 'Atrial fibrillation with premature ventricular, Incomplete left bundle block': 1, 
    '1st degree A-V block with Premature supraventricular complexes, Left bundle branch block': 1, 'Left anterior hemiblock': 1,
    '1st degree A-V block, Left bundle branch block': 1, 'Premature supraventricular and ventricular complexes, Right bundle branch block': 1,
    'Atrial fibrillation, Right bundle branch block': 1, 'Atrial flutter with variable A-V block': 1 
}
df['preop_ecg'] = df['preop_ecg'].map(d)

d = {'Normal': 0, 'Mild obstructive': 1, 'Moderate obstructive': 2,
     'Mixed or pure obstructive': 3, 'Severe restrictive': 3
}
df['preop_pft'] = df['preop_pft'].map(d)

features = ['age', 'sex', 'bmi', 'asa', 'department','optype','ane_type','preop_ecg','preop_pft', 'preop_htn', 'preop_dm','emop','preop_pao2',
            'preop_na', 'preop_k', 'preop_hb','preop_sao2', 'preop_pao2','preop_hco3'
]

X = df[features]
y = df['risk']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X, y)

plt.figure(figsize=(12, 8))
tree.plot_tree(dtree, feature_names=features, filled=True)
plt.savefig('decision_tree.png')

st.image('decision_tree.png')
