# -*- coding: utf-8 -*-
import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.externals import joblib
import pickle


def snippets_to_list():
    folders = os.listdir(os.getcwd() + "/../DATA/train_db/")
    mega_list = []
    for folder in folders:
        folder_content = os.listdir(os.getcwd() + "/../DATA/train_db/" + folder)
        for json_query in folder_content:
            json_file = open(os.getcwd() + "/../DATA/train_db/" + folder + "/" + json_query)
            json_str  = json_file.read()
            json_data = json.loads(json_str)
            json_keys = json_data.keys()
            for json_key in json_keys:
                mega_list.append(json_data[json_key]["text"] + " " + json_data[json_key]["text"])
    return mega_list


def save_object(obj, filename):
    joblib.dump(obj, filename)
    # pickle.dump(obj, open(filename, "wb"))
    # with open(filename, 'wb') as output:  # Overwrites any existing file.
    #     pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def list_to_pickle_count_vectorizer():
    tf_vectorizer = CountVectorizer(min_df=10, stop_words='english')
    tf = tf_vectorizer.fit(snippets_to_list())
    save_object(tf, 'count_vectorizer.pkl')


def list_to_pickle_tfidf_vectorizer():
    tfidf_vectorizer = TfidfVectorizer(min_df=10, stop_words='english')
    tf = tfidf_vectorizer.fit(snippets_to_list())
    save_object(tf, 'tfidf_vectorizer.pkl')


def list_to_pickle_vectorizer(path):
    t_data = snippets_to_list()
    tf_vectorizer = CountVectorizer(min_df=10, stop_words='english')
    tf = tf_vectorizer.fit(t_data)
    save_object(tf, path + 'count_vectorizer.pkl')

    tfidf_vectorizer = TfidfVectorizer(min_df=10, stop_words='english')
    tf = tfidf_vectorizer.fit(t_data)
    save_object(tf, path + 'tfidf_vectorizer.pkl')


if __name__ == "__main__":
    # List all the folders of train_db
    # print(len(snippets_to_list()))
    # list_to_pickle_count_vectorizer()
    # list_to_pickle_tfidf_vectorizer()
    list_to_pickle_vectorizer("/../DATA/")


