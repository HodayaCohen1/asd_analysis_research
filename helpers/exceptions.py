class FieldMissingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ValueMissingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ArgumentMissingException(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
