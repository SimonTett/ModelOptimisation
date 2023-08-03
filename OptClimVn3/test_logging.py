""" test that logging works"""
import logging

logger = logging.getLogger()
while logger.hasHandlers():
    logger.removeHandler(logger.handlers[0])

class fred:
    logger = logging.getLogger('OPTCLIM.fred')
    def test_method(self):
        self.logger.info('testing testing ')
        return 2

class fred2:
    logger = logging.getLogger('OPTCLIM.fred2')
    def test_method(self):
        self.logger.info('testing testing ')
        return 2

my_logger = logging.getLogger('OPTCLIM')
my_logger.handlers.clear()
my_logger.setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
fmt = '%(levelname)s:%(name)s:%(funcName)s: %(message)s'
formatter = logging.Formatter(fmt)
console_handler.setFormatter(formatter)
my_logger.addHandler(console_handler)

a=fred()
a.test_method()
a2=fred2()
a2.test_method()
my_logger.info("Hello World")

