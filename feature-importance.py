import matplotlib.pyplot as plt

features = [
    ('f*ck', 0.01819234011746802),
    ('yo', 0.01675511546272876),
    ('n**ga', 0.0165247486331531),
    ('em', 0.01635410396894181),
    ('sh*t', 0.016246600748652407),
    ('the', 0.012618444724575574),
    ('like', 0.011272452783229447),
    ('rap', 0.010985789300796196),
    ('up', 0.010945402730871969),
    ('in', 0.010912088860791964),
    ('on', 0.010846242796993151),
    ('a', 0.01048710573236125),
    ('man', 0.009974469686393135),
    ('got', 0.009608044485909949),
    ('off', 0.009017594883935696),
    ('yall', 0.008894512357477492),
    ('get', 0.008798489541578521),
    ('b*tch', 0.008778641777632702),
    ('duration', 0.008578890476081208),
    ('ass', 0.008557838095318716),
]

values = [pair[1] for pair in features]
words = [pair[0] for pair in features]

x = range(len(values))
markerline, stemlines, baseline = plt.stem(x, values, '-.', orientation='vertical')
plt.setp(baseline, 'color', 'r', 'linewidth', 2)
plt.xticks(x, words, rotation='vertical')
plt.yticks(rotation='vertical')
plt.show()
