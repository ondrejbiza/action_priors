class MockLogger:

    @staticmethod
    def info(message):

        print("mock logger, info: {:s}".format(message))

    @staticmethod
    def warning(message):
        print("mock logger, info: {:s}".format(message))

    @staticmethod
    def error(message):
        print("mock logger, info: {:s}".format(message))
