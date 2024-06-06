Problem statement: 
Hospital Health+ has had management issues with their patients, especially those who stay longer. 
During the past few years, they have not been able to allocate their resources; namely, hospital 
beds, staff and doctors efficiently.
So they have sought the help of data scientists to find a solution to the length of stay problem that they have, 
and what managerial insights can be presented to prevent the hospital from overflowing. 

Findings:
The department of Gynecology, Anesthesia and Radiotherapy, age 31-50 and admission deposit are the most important determinants of LOS.
Also from EDA, the gynecology department is the most dominant deparment in the hospital.
Females are the most common patients and most of them are insured.
Bivariate analysis has shown also that gynecology department has specific wards.
Wards A, C and E have more lengthy days of stay.
Managerial implications mean to allocate more resources in these wards, mostly extra beds to avoid overcapacity.

Also extreme illnesses are taken in charge in wards A, E and C. Therefore, they need more rescouces allocated to them.

Last but not least, we were able to train a random forest model that was able to predict the LOS with the MAE of 0.86
For a new patient with all his data, we can predict how long he will stay in the hospital with a precision of 0.86 
and the hospital can plan for that particular patient accordingly.
