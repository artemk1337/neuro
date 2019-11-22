from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, BaggingRegressor
import _pickle as cPickle
import json


def rfc(a, b, c, d, filename):
    g_perfect = [1, 1]
    k_fail = 0
    i_fail = 0
    k = 0
    g_max = 0
    g_max_te = 0
    for i in range(1, 301):
        with open('max_res.json', 'w') as file:
            json.dump({"i": g_perfect[0], "k": g_perfect[1], "g_max": g_max, "g_max_te": g_max_te}, file)
        print(f'i: {i}, k: {k},\n'
              f'g_max: {g_max}, g_max_te: {g_max_te}')
        if i_fail >= 30:
            break
        prev_good_te = 0
        for k in range(1, 501):
            if k_fail >= 30:
                k_fail = 0
                # i_fail += 1
                # k += 30
                break
            clf = RandomForestClassifier(n_estimators=i, max_depth=k, random_state=0)
            clf.fit(a, b)
            tmp = clf.score(a, b)
            tmp1 = clf.score(c, d)
            if tmp1 > prev_good_te:
                k_fail = 0
                prev_good_te = tmp1
            if g_max_te < tmp1 <= tmp:
                g_max = tmp
                g_max_te = tmp1
                i_fail = 0
                k_fail = 0
                g_perfect[0] = i
                g_perfect[1] = k
                with open(filename, 'wb') as f:
                    cPickle.dump(clf, f)
            else:
                k_fail += 1
        i_fail += 1
