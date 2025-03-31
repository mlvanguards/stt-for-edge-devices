class STSServiceException(Exception):
    pass


class ImproperlyConfigured(STSServiceException):
    pass

class ExternalServiceAPIError(STSServiceException):
    """ Exception to mark network communication errors """

    def __init__(self, code, message):
        self.code = code
        super().__init__(message)
