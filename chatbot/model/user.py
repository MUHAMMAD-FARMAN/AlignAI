class User:
    def __init__(self, id, name, email):
        self.id = id
        self.name = name
        self.email = email

    def get_id(self):
        return self.id

    def get_name(self):
        return self.name

    def get_email(self):
        return self.email

    def set_name(self, name):
        self.name = name

    def set_email(self, email):
        self.email = email
