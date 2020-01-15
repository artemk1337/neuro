from sklearn.datasets import fetch_20newsgroups
from word2vec_keras import Word2VecKeras

TTT = 0

if TTT == 1:
    # fetch the dataset using scikit-learn
    categories = ['alt.atheism', 'soc.religion.christian',
                  'comp.graphics', 'sci.med']

    train_b = fetch_20newsgroups(subset='train',
                                 categories=categories, shuffle=True, random_state=42)
    test_b = fetch_20newsgroups(subset='test',
                                categories=categories, shuffle=True, random_state=42)

    print('size of training set: %s' % (len(train_b['data'])))
    print('size of validation set: %s' % (len(test_b['data'])))
    print('classes: %s' % (train_b.target_names))

    x_train = train_b.data
    y_train = [train_b.target_names[idx] for idx in train_b.target]
    x_test = test_b.data
    y_test = [train_b.target_names[idx] for idx in test_b.target]

    print(x_train[-1])
    quit()
    model = Word2VecKeras()
    model.train(x_train, y_train)

    print(model.evaluate(x_test, y_test))

    model.save('./model.tar.gz')
else:
    model = Word2VecKeras()
    model.load('model.tar.gz')
    print(model.predict('hello'))


