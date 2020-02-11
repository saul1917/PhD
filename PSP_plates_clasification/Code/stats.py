import pandas as pd
from sklearn.metrics import confusion_matrix

df = pd.read_csv("usmmodel25epo224_2900norot.csv")

def get_stats(df): 
    expected_label_column = list(df["Expected_label"])
    given_label_column = list(df["Given_label"])

    confusion_matrix = confusion_matrix(expected_label_column, given_label_column, labels=["no", "yes"])
    tn, fp, fn, tp = confusion_matrix.ravel()
    total = tn + fp + fn + tp

    accuracy=(tn+tp)/total
    print ('Accuracy : ', accuracy)

    sensitivity = tp/(fn+tp)
    print('Sensitivity : ', sensitivity )

    specificity = tn/(tn+fp)
    print('Specificity : ', specificity)

    print((tn, fp, fn, tp))
