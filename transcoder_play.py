import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas



le = LabelEncoder()

gender = ["male", "female"]
region = ["from Europe", "from US", "from Asia"]
browser = ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"]

le.fit([gender, region, browser])
print le.classes_
# ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]



