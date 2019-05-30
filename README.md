[Home](https://mgcodesandstats.github.io/) |
[Time Series Consulting](https://mgcodesandstats.github.io/time-series-consulting-python-r/) |
[Portfolio](https://mgcodesandstats.github.io/portfolio/) |
[Terms and Conditions](https://mgcodesandstats.github.io/terms/) |
[E-mail me](mailto:contact@michaeljgrogan.com) |
[LinkedIn](https://www.linkedin.com/in/michaeljgrogan/)

# Predicting Hotel Cancellations with ExtraTreesClassifier and Logistic Regression

Hotel cancellations can cause issues for managers. Not only is there the lost revenue as a result of the customer cancelling, but this can also cause difficulty in coordinating bookings and adjusting revenue management practices.

Data analytics can help to overcome this issue, in terms of identifying the customers who are most likely to cancel – allowing a hotel chain to adjust its marketing strategy accordingly.

To investigate how machine learning can aid in this task, I decided to generate a logistic regression in Python to determine whether cancellations can be accurately predicted with this model. The Algarve Hotel dataset available from [Science Direct](https://www.sciencedirect.com/science/article/pii/S2352340918315191) was used to train and validate the model, and then the logistic regression was used to generate predictions on a second dataset for a hotel in Lisbon. A 99% classification accuracy was achieved across the test set predictions.

## Data Processing

Firstly, the relevant libraries were imported and the relevant data type for each variable was classified:

```
import os
import csv
import random
import statsmodels.api as sm
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler

dtypes = {
        'IsCanceled':                                    'float64',
        'LeadTime':                                          'float64',
        'StaysInWeekendNights':                                     'float64',
        'StaysInWeekNights':                                     'float64',
        'Adults':                            'float64',
        'Children':                            'float64',
        'Babies':                                  'float64',
        'Meal':                                    'category',
        'Country':                                               'category',
        'MarketSegment':                                    'category',
        'DistributionChannel':                                       'category',
        'IsRepeatedGuest':                               'float64',
        'PreviousCancellations':                                    'float64',
        'PreviousBookingsNotCanceled':                          'float64',
        'ReservedRoomType':                                             'category',
        'AssignedRoomType':                                            'category',
        'BookingChanges':                                                'float64',
        'DepositType':                                              'category',
        'Agent':                                              'category',
        'Company':                                 'category',
        'DaysInWaitingList':                                           'float64',
        'CustomerType':                                           'category',
        'ADR':                                          'float64',
        'RequiredCarParkingSpaces':                                      'float64',
        'TotalOfSpecialRequests':                                              'float64',
        'ReservationStatus':                                                'category'
        }
```

As we can see, there are many variables that can potentially influence whether a customer is going to cancel or not, and not all of these variables will necessarily be relevant in determining this.

The data is imported, and then the data is factorized so as to express the categories in numerical format:

```
train_df = pd.read_csv('H1.csv', dtype=dtypes)
a=train_df.head()
b=train_df
b

data=b.apply(lambda col: pd.factorize(col, sort=True)[0])
data
```

![factorized](factorized.png)

The variables are then stacked together under the numpy format:

```
IsCanceled = data['IsCanceled']
y = IsCanceled
leadtime = data['LeadTime'] #1
staysweekendnights = data['StaysInWeekendNights'] #2
staysweeknights = data['StaysInWeekNights'] #3
adults = data['Adults'] #4
children = data['Children'] #5
babies = data['Babies'] #6
meal = data['Meal'] #7
country = data['Country'] #8
marketsegment = data['MarketSegment'] #9
distributionchannel = data['DistributionChannel'] #10
isrepeatedguest = data['IsRepeatedGuest'] #11
previouscancellations = data['PreviousCancellations'] #12
previousbookingsnotcanceled = data['PreviousBookingsNotCanceled'] #13
reservedroomtype = data['ReservedRoomType'] #14
assignedroomtype = data['AssignedRoomType'] #15
bookingchanges = data['BookingChanges'] #16
deptype = data['DepositType'] #17
agent = data['Agent'] #18
company = data['Company'] #19
dayswaitinglist = data['DaysInWaitingList'] #20
custype = data['CustomerType'] #21
adr = data['ADR'] #22
rcps = data['RequiredCarParkingSpaces'] #23
totalsqr = data['TotalOfSpecialRequests'] #24
reserv = data['ReservationStatus'] #25

x = np.column_stack((leadtime,staysweekendnights,staysweeknights,adults,children,babies,meal,country,marketsegment,distributionchannel,isrepeatedguest,previouscancellations,previousbookingsnotcanceled,reservedroomtype,assignedroomtype,bookingchanges,deptype,agent,company,dayswaitinglist,custype,adr,rcps,totalsqr,reserv))
x = sm.add_constant(x, prepend=True)
y=y.values
```

## Extra Trees Classifier

As mentioned, there are many variables included in the analysis – not all of them will be relevant in determining whether a customer is likely to cancel or not.

To solve this issue, the Extra Trees Classifier is used to print the importance of each variable in numerical format, and the variables with the highest ranked importance are then chosen accordingly.

```
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier()
model.fit(x, y)
print(model.feature_importances_)
```

Here are the generated readings:

```
[0.00000000e+00 2.80773623e-02 2.87638739e-03 5.72799343e-03
 1.72631728e-03 1.64816050e-03 2.23887272e-04 2.59074686e-03
 3.76929953e-02 2.09040742e-02 3.63074785e-03 3.40060409e-03
 9.81971773e-03 5.64723064e-04 3.11625309e-03 8.50316994e-03
 4.42794922e-03 4.86811986e-02 4.81089533e-03 1.07322025e-03
 8.82583550e-04 8.32794531e-03 8.61034596e-03 1.97387302e-02
 6.70055352e-03 7.66243438e-01]
```
From the above, the identified features of importance are lead time, country, market segment, deposit type, customer type, and reservation status.

The variables are redefined in the stack:

```
y1 = y
x1 = np.column_stack((leadtime,country,marketsegment,deptype,custype,reserv))
x1 = sm.add_constant(x1, prepend=True)
```

## Logistic Regression

Now, the logistic regression is generated:

```
x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=0)

logreg = LogisticRegression().fit(x1_train,y1_train)
logreg

print("Training set score: {:.3f}".format(logreg.score(x1_train,y1_train)))

print("Test set score: {:.3f}".format(logreg.score(x1_test,y1_test)))

import statsmodels.api as sm
logit_model=sm.Logit(y1,x1)
result=logit_model.fit()
print(result.summary())
```

Here are the results:

```
Training set score: 0.993
Test set score: 0.992
Optimization terminated successfully.
         Current function value: 0.100929
         Iterations 8
                           Logit Regression Results                           
==============================================================================
Dep. Variable:                      y   No. Observations:                40058
Model:                          Logit   Df Residuals:                    40051
Method:                           MLE   Df Model:                            6
Date:                Tue, 14 May 2019   Pseudo R-squ.:                  0.8291
Time:                        19:05:49   Log-Likelihood:                -4043.0
converged:                       True   LL-Null:                       -23662.
                                        LLR p-value:                     0.000
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const          1.5074      0.214      7.029      0.000       1.087       1.928
x1             0.0014      0.000      3.706      0.000       0.001       0.002
x2             0.0184      0.001     14.251      0.000       0.016       0.021
x3             0.1697      0.027      6.223      0.000       0.116       0.223
x4             1.1369      0.125      9.090      0.000       0.892       1.382
x5            -0.0812      0.060     -1.345      0.179      -0.199       0.037
x6            -7.2326      0.075    -96.874      0.000      -7.379      -7.086
==============================================================================
```

Now, the logistic regression is used to predict cancellations for the test data, and a confusion matrix is generated to determine the incidence of true/false positives and negatives:

```
pr = logreg.predict(x1_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y1_test,pr))
print(classification_report(y1_test,pr))
```

The confusion matrix is generated:

```
[[7267    0]
 [  76 2672]]
              precision    recall  f1-score   support

           0       0.99      1.00      0.99      7267
           1       1.00      0.97      0.99      2748

   micro avg       0.99      0.99      0.99     10015
   macro avg       0.99      0.99      0.99     10015
weighted avg       0.99      0.99      0.99     10015
```

From the above, we see that the accuracy in classification was quite high.

Here is an ROC curve illustrating the true vs false positive rate.

```
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
falsepos,truepos,thresholds=roc_curve(y1_test,logreg.decision_function(x1_test))
plt.plot(falsepos,truepos,label="ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
cutoff=np.argmin(np.abs(thresholds))
plt.plot(falsepos[cutoff],truepos[cutoff],'o',markersize=10,label="cutoff",fillstyle="none")
plt.show()
```

![roc-curve](roc-curve.png)

## Testing against unseen data

Now that the logistic regression has shown a high degree of classification accuracy against the validation dataset, another dataset H2.csv (also available from Science direct) is used for comparison purposes, i.e. the logistic regression generated using the last dataset is now used to predict classifications across this dataset (for a different hotel located in Lisbon, Portugal).

# Conclusion
This has been an illustration of how a logistic regression can be used to predict hotel cancellations. We have also seen how the Extra Trees Classifier can be used as a feature selection tool to identify the most reliable predictors of customer cancellations.
