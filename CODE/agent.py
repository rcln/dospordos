# -*- coding: utf-8 -*-


class Agent:

    def __init__(self, queries, current_query):
        self.queries = queries
        self.current_query = current_query
        self.current_snippet = None

    def next_snippet(self):
        self.current_snippet = self.queries[self.current_query].get()

    def change_query(self, query):
        self.current_query = query

    @staticmethod
    def stop():
        pass

    @staticmethod
    def change_db():
        pass

